"""
Context Builder

Role: 대화 히스토리 → 현재 턴 응답 생성/평가에 필요한 정보만 정리

Output:
    - recent_turns: 최근 N턴 대화
    - state_summary: 핵심 사실/감정/문제 압축
    - meta: 옵션 (감정 카테고리, 문제 유형 등)
"""
from typing import List, Dict, Any, Optional
from src.llm.openai_client import OpenAIClient
from src.llm.summarize import build_summary_prompt


class ContextBuilder:
    """대화 히스토리를 요약 + 최근 턴으로 압축"""
    
    def __init__(self, 
                 llm_client: OpenAIClient,
                 window_size: int = 6,
                 use_summary: bool = True):
        """
        Args:
            llm_client: LLM summarization용 클라이언트
            window_size: 최근 N턴 유지
            use_summary: 긴 히스토리 요약 사용 여부
        """
        self.llm = llm_client
        self.window_size = window_size
        self.use_summary = use_summary
    
    def build_context(self, 
                     dialog_history: List[Dict[str, str]],
                     situation: Optional[str] = None) -> Dict[str, Any]:
        """
        대화 히스토리를 현재 턴 컨텍스트로 변환
        
        Args:
            dialog_history: [{"speaker": "seeker/supporter", "text": "..."}]
            situation: 대화 상황 설명 (옵션)
        
        Returns:
            {
                "recent_turns": [...],  # 최근 window_size 턴
                "state_summary": str,   # LLM 생성 요약 (긴 대화만)
                "situation": str,       # 원본 situation
                "meta": {...}           # 부가 정보
            }
        """
        # 최근 턴 추출
        recent_turns = dialog_history[-self.window_size:] if len(dialog_history) > self.window_size else dialog_history
        
        # 요약 생성 (긴 대화만)
        state_summary = None
        if self.use_summary and len(dialog_history) > self.window_size:
            # LLM으로 전체 대화 요약
            prompt = build_summary_prompt(dialog_history, situation)
            state_summary = self.llm.generate(prompt, max_tokens=150, temperature=0.3)
        
        # Meta 정보 추출 (옵션)
        meta = self._extract_meta(dialog_history, situation)
        
        return {
            "recent_turns": recent_turns,
            "state_summary": state_summary,
            "situation": situation,
            "meta": meta,
            "total_turns": len(dialog_history)
        }
    
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
