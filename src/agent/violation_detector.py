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
from src.llm.judge import build_judge_prompt
from src.llm.openai_client import OpenAIClient


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
                 llm_client: Optional[OpenAIClient] = None):
        """
        Args:
            mode: "model" (학습된 분류기) or "judge" (LLM 평가)
            model_path: 학습된 모델 경로 (mode="model"일 때)
            llm_client: Judge용 LLM (mode="judge"일 때)
        """
        self.mode = mode
        
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
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
        # 입력 텍스트 구성 (전처리 형식과 동일)
        input_text = self._format_input(context, candidate["text"])
        
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
            probs = torch.softmax(logits, dim=-1)[0]
            pred_id = torch.argmax(probs).item()
            confidence = probs[pred_id].item()
        
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
            "evidence": f"Model prediction (confidence: {confidence:.2%})",
            "severity": self.SEVERITY_RANK[predicted_label]
        }
    
    def _detect_with_judge(self, 
                          context: Dict[str, Any], 
                          candidate: Dict[str, Any]) -> Dict[str, Any]:
        """LLM Judge로 위반 감지"""
        # Judge 프롬프트 구성
        dialog = context["recent_turns"] + [{"speaker": "supporter", "text": candidate["text"]}]
        prompt = build_judge_prompt(dialog, context.get("situation"))
        
        # LLM 평가
        response = self.llm.generate(prompt, max_tokens=300, temperature=0.0)
        
        # 파싱 (간단한 버전, 실제로는 더 robust하게)
        predicted_label = self._parse_judge_output(response)
        
        # Multi-label
        violations = {f"v{i}": 0 for i in range(1, 6)}
        if predicted_label != "normal":
            violations[predicted_label] = 1
        
        return {
            "candidate_id": candidate["id"],
            "violations": violations,
            "top_violation": predicted_label,
            "confidence": 0.9,  # Judge는 confidence 없음
            "evidence": response,
            "severity": self.SEVERITY_RANK[predicted_label]
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
