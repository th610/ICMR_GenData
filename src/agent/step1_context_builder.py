"""
Context Builder

Role: 대화 히스토리 → 현재 턴 응답 생성/평가에 필요한 정보만 정리

Output:
    - recent_turns: 최근 N턴 대화 (토큰 제한 기반)
    - state_summary: 핵심 사실/감정/문제 압축
    - meta: 옵션 (감정 카테고리, 문제 유형 등)
"""
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
from src.llm.openai_client import OpenAIClient


class ContextBuilder:
    """대화 히스토리를 요약 + 최근 턴으로 압축 (Reverse Truncation)"""
    
    def __init__(self, 
                 llm_client: OpenAIClient,
                 window_size: int = 6,
                 max_tokens: int = 512,
                 use_summary: bool = True):
        """
        Args:
            llm_client: LLM summarization용 클라이언트
            window_size: 최대 턴 수 (soft limit)
            max_tokens: 최대 토큰 수 (hard limit, 512)
            use_summary: 긴 히스토리 요약 사용 여부
        """
        self.llm = llm_client
        self.window_size = window_size
        self.max_tokens = max_tokens
        self.use_summary = use_summary
        
        # Tokenizer 로드 (RoBERTa - 학습 시와 동일)
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        special_tokens = ["<SEEKER>", "<SUPPORTER>", "<SUPPORTER_TARGET>"]
        self.tokenizer.add_tokens(special_tokens)
    
    def build_context(self, 
                     dialog_history: List[Dict[str, str]],
                     situation: Optional[str] = None) -> Dict[str, Any]:
        """
        대화 히스토리를 현재 턴 컨텍스트로 변환 (Reverse Truncation)
        
        Args:
            dialog_history: [{"speaker": "seeker/supporter", "text": "..."}]
            situation: 대화 상황 설명 (옵션)
        
        Returns:
            {
                "recent_turns": [...],  # 토큰 제한 내 최근 턴들
                "state_summary": str,   # LLM 생성 요약 (긴 대화만)
                "situation": str,       # 원본 situation
                "meta": {...}           # 부가 정보
            }
        """
        # 토큰 기반 Reverse Truncation
        recent_turns = self._reverse_truncate(dialog_history)
        
        # 요약 생성 (긴 대화만)
        state_summary = None
        if self.use_summary and len(dialog_history) > self.window_size:
            # 요약 기능 스킵 (간단화)
            state_summary = None
        
        # Meta 정보 추출 (옵션)
        meta = self._extract_meta(dialog_history, situation)
        
        return {
            "recent_turns": recent_turns,
            "state_summary": state_summary,
            "situation": situation,
            "meta": meta,
            "total_turns": len(dialog_history)
        }
    
    def _reverse_truncate(self, dialog_history: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Reverse Truncation: 뒤에서부터 토큰 세면서 512 제한 내로 자르기
        
        학습 시와 동일한 방식:
        1. 대화를 역순으로 순회
        2. 각 턴의 토큰 수 누적
        3. 512 초과 시 중단
        4. 선택된 턴들을 원래 순서로 반환
        """
        if not dialog_history:
            return []
        
        selected_turns = []
        total_tokens = 0
        
        # 역순으로 순회 (최근 턴부터)
        for turn in reversed(dialog_history):
            # 턴 포맷: "<speaker> text" (학습 시와 동일)
            speaker_tag = f"<{turn['speaker'].upper()}>" if turn['speaker'] in ['seeker', 'supporter'] else turn['speaker']
            turn_text = f"{speaker_tag} {turn['text']}"
            
            # 토큰 수 계산
            turn_tokens = len(self.tokenizer.encode(turn_text, add_special_tokens=False))
            
            # 512 초과 체크 (hard limit)
            if total_tokens + turn_tokens > self.max_tokens:
                break
            
            selected_turns.insert(0, turn)  # 앞에 삽입 (원래 순서 유지)
            total_tokens += turn_tokens
            
            # NOTE: window_size는 삭제 - 토큰 제한까지 최대한 넣음
        
        return selected_turns
    
    def _extract_meta(self, 
                     dialog_history: List[Dict[str, str]], 
                     situation: Optional[str]) -> Dict[str, Any]:
        """
        메타 정보 추출 (감정 카테고리, 문제 유형 등)
        
        현재는 기본 정보만, 필요시 확장 가능
        """
        return {
            "turn_count": len(dialog_history),
            "has_situation": situation is not None,
            "last_speaker": dialog_history[-1]["speaker"] if dialog_history else None
        }
    
    def format_for_display(self, context: Dict[str, Any]) -> str:
        """컨텍스트를 읽기 쉬운 텍스트로 변환"""
        lines = []
        
        if context.get("situation"):
            lines.append(f"[SITUATION]\n{context['situation']}\n")
        
        if context.get("state_summary"):
            lines.append(f"[SUMMARY]\n{context['state_summary']}\n")
        
        lines.append(f"[RECENT CONVERSATION - Last {len(context['recent_turns'])} turns]")
        for i, turn in enumerate(context["recent_turns"], 1):
            lines.append(f"{i}. {turn['speaker']}: {turn['text']}")
        
        return "\n".join(lines)
