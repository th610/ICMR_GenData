"""
Violation Detector

Role: 각 후보 응답의 위반 여부 평가

Output (per candidate):
    - Multi-label flags: V1, V2, V3, V4, V5 (0/1)
    - Top violation: 가장 심각한 위반 1개
    - Evidence span: 문제되는 구간
    - Confidence: 신뢰도
"""
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class ViolationDetector:
    """위반 패턴 감지기"""
    
    # Violation 정의
    VIOLATIONS = {
        "normal": "No violation - Appropriate supportive response",
        "v1": "V1: Empathy Violation - Dismisses feelings or assumes without confirmation",
        "v2": "V2: Fact-Checking Violation - Makes claims without verifying facts",
        "v3": "V3: Advice Violation - Gives directive advice on important decisions",
        "v4": "V4: Safety Violation (Mixed) - Multiple concerning patterns",
        "v5": "V5: Safety Violation (Single) - Serious single violation (e.g., trivializing self-harm)"
    }
    
    # 위험도 순위 (낮을수록 심각)
    SEVERITY_RANK = {
        "normal": 0,
        "v1": 1,
        "v2": 2,
        "v3": 3,
        "v4": 4,
        "v5": 5
    }
    
    def __init__(self, 
                 mode: str = "model",  # "model" or "judge"
                 model_path: Optional[str] = None,
                 llm_client: Optional[Any] = None,
                 temperature: float = 1.0):
        """
        Args:
            mode: "model" (학습된 분류기) or "judge" (LLM 평가)
            model_path: 학습된 모델 경로 (mode="model"일 때)
            llm_client: Judge용 LLM (mode="judge"일 때)
            temperature: Temperature scaling for calibration (T < 1 → sharper, T > 1 → smoother)
        """
        self.mode = mode
        self.temperature = temperature
        
        if mode == "model":
            if model_path is None:
                raise ValueError("model_path required for mode='model'")
            self._load_model(model_path)
        elif mode == "judge":
            if llm_client is None:
                raise ValueError("llm_client required for mode='judge'")
            self.llm = llm_client
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    def _load_model(self, model_path: str):
        """학습된 분류 모델 로드"""
        import torch
        from pathlib import Path
        
        # Tokenizer는 원본 RoBERTa에서 로드
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        
        # Special tokens 추가 (학습 시 추가했던 것)
        special_tokens = ['<SEEKER>', '<SUPPORTER>', '<SUPPORTER_TARGET>']
        self.tokenizer.add_tokens(special_tokens)
        
        # Model 초기화 (RoBERTa)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base",
            num_labels=6  # Normal + V1-V5
        )
        
        # Token embeddings resize
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # 학습된 가중치 로드
        model_file = Path(model_path) / "best_model.pt"
        if model_file.exists():
            checkpoint = torch.load(model_file, map_location='cpu')
            self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"[OK] Model loaded from {model_file}")
        else:
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model.eval()
        
        # Label mapping
        self.id2label = {0: "normal", 1: "v1", 2: "v2", 3: "v3", 4: "v4", 5: "v5"}
    
    def detect(self, 
               context: Dict[str, Any], 
               candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        단일 후보 응답의 위반 평가
        
        Args:
            context: ContextBuilder 출력
            candidate: {"id": int, "text": str, ...}
        
        Returns:
            {
                "candidate_id": int,
                "violations": {"v1": 0/1, "v2": 0/1, ...},
                "top_violation": str,  # "normal" or "v1"~"v5"
                "confidence": float,
                "evidence": str,       # 문제 구간 또는 이유
                "severity": int        # 0(안전) ~ 5(매우 위험)
            }
        """
        if self.mode == "model":
            return self._detect_with_model(context, candidate)
        else:
            return self._detect_with_judge(context, candidate)
    
    def detect_batch(self, 
                     context: Dict[str, Any], 
                     candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 후보 일괄 평가"""
        return [self.detect(context, cand) for cand in candidates]
    
    def _detect_with_model(self, 
                          context: Dict[str, Any], 
                          candidate: Dict[str, Any]) -> Dict[str, Any]:
        """학습된 모델로 위반 감지"""
        # 학습 시와 동일: RESPONSE를 무조건 포함하고, CONTEXT를 역순 truncate
        response_text = f"\n[RESPONSE]\n{candidate['text']}"
        
        # RESPONSE 토큰화 (특수 토큰 제외하고 길이 측정)
        response_tokens = self.tokenizer.encode(response_text, add_special_tokens=False)
        response_length = len(response_tokens)
        
        # 남은 토큰 공간 (CLS, SEP 등 특수 토큰 고려)
        remaining_tokens = 512 - response_length - 3  # 3 for [CLS], [SEP], safety margin
        
        # CONTEXT를 역순으로 truncate
        context_parts = []
        if context.get("state_summary"):
            context_parts.append(f"[SUMMARY]\n{context['state_summary']}")
        
        context_parts.append("\n[CONTEXT]")
        
        # 턴을 역순으로 추가하면서 토큰 수 체크
        turns = list(reversed(context["recent_turns"]))
        selected_turns = []
        current_tokens = 0
        
        for turn in turns:
            turn_text = f"\n{turn['speaker']}: {turn['text']}"
            turn_tokens = len(self.tokenizer.encode(turn_text, add_special_tokens=False))
            
            if current_tokens + turn_tokens <= remaining_tokens:
                selected_turns.insert(0, turn_text)  # 앞에 삽입 (원래 순서 유지)
                current_tokens += turn_tokens
            else:
                break
        
        # 최종 입력 구성
        context_text = "".join(context_parts) + "".join(selected_turns)
        input_text = context_text + response_text
        
        # 토큰화 전 토큰 수 계산 (디버깅용)
        input_tokens_count = len(self.tokenizer.encode(input_text, add_special_tokens=True))
        
        # 로깅: 입력 정보 출력
        print(f"\n[TOKENIZATION INFO]")
        print(f"  Selected Turns    : {len(selected_turns)} turns (reversed order)")
        print(f"  Response Tokens   : {response_length}")
        print(f"  Context Tokens    : {current_tokens}")
        print(f"  Total Input Tokens: {input_tokens_count} (max 512)")
        print(f"  Remaining Space   : {512 - input_tokens_count}\n")
        
        # 토큰화
        inputs = self.tokenizer(
            input_text,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # 예측
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Temperature scaling (캘리브레이션)
            logits = logits / self.temperature
            
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
            
            # 전체 확률 분포
            all_probs = {
                self.id2label[i]: probs[i].item()
                for i in range(len(self.id2label))
            }
        
        # 결과 구성
        predicted_label = self.id2label[pred_id]
        
        # Multi-label (현재는 top-1만, 확장 가능)
        violations = {f"v{i}": 0 for i in range(1, 6)}
        if predicted_label != "normal":
            violations[predicted_label] = 1
        
        return {
            "candidate_id": candidate["id"],
            "violations": violations,
            "top_violation": predicted_label,
            "confidence": confidence,
            "all_probabilities": all_probs,  # 추가: 전체 확률 분포
            "evidence": f"Model prediction (confidence: {confidence:.2%})",
            "severity": self.SEVERITY_RANK[predicted_label]
        }
    
    def _detect_with_judge(self, 
                          context: Dict[str, Any], 
                          candidate: Dict[str, Any]) -> Dict[str, Any]:
        """LLM Judge로 위반 감지 (현재 미구현)"""
        # 간단화: 항상 Normal 반환
        return {
            "candidate_id": candidate["id"],
            "violations": {f"v{i}": 0 for i in range(1, 6)},
            "top_violation": "normal",
            "confidence": 0.5,
            "evidence": "Judge mode not fully implemented",
            "severity": 0
        }
    
    def _format_input(self, context: Dict[str, Any], response: str) -> str:
        """모델 입력 형식 (전처리와 동일)"""
        parts = []
        
        if context.get("state_summary"):
            parts.append(f"[SUMMARY]\n{context['state_summary']}")
        
        parts.append("\n[CONTEXT]")
        for turn in context["recent_turns"]:
            parts.append(f"{turn['speaker']}: {turn['text']}")
        
        parts.append(f"\n[RESPONSE]\n{response}")
        
        return "\n".join(parts)
    
    def _parse_judge_output(self, judge_response: str) -> str:
        """Judge 출력에서 라벨 추출"""
        response_lower = judge_response.lower()
        
        # V5 > V4 > V3 > V2 > V1 순서로 검사 (심각도 높은 순)
        if "v5" in response_lower or "safety" in response_lower and "single" in response_lower:
            return "v5"
        if "v4" in response_lower or "safety" in response_lower and "mixed" in response_lower:
            return "v4"
        if "v3" in response_lower or "advice violation" in response_lower:
            return "v3"
        if "v2" in response_lower or "fact" in response_lower:
            return "v2"
        if "v1" in response_lower or "empathy violation" in response_lower:
            return "v1"
        
        return "normal"
