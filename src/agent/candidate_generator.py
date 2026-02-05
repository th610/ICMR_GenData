"""
Candidate Generator

Role: K개의 응답 후보 생성

Why K candidates?
    - 위반이 적은 대안 선택 가능
    - 표현/방향이 다른 후보 확보
    - 단일 생성보다 안정적 품질/안전 제어
"""
from typing import List, Dict, Any
from src.llm.openai_client import OpenAIClient


class CandidateGenerator:
    """K개 후보 응답 생성기"""
    
    def __init__(self, 
                 llm_client: OpenAIClient,
                 num_candidates: int = 3,
                 temperature: float = 0.8,
                 max_tokens: int = 150):
        """
        Args:
            llm_client: 응답 생성용 LLM
            num_candidates: 생성할 후보 개수 K
            temperature: 다양성 조절 (높을수록 다양)
            max_tokens: 응답 최대 길이
        """
        self.llm = llm_client
        self.K = num_candidates
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    def generate_candidates(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        컨텍스트를 받아 K개 후보 생성
        
        Args:
            context: ContextBuilder 출력
                - recent_turns: List[Dict]
                - state_summary: str
                - situation: str
        
        Returns:
            [
                {"id": 0, "text": "...", "generation_info": {...}},
                {"id": 1, "text": "...", "generation_info": {...}},
                ...
            ]
        """
        # 프롬프트 구성
        prompt = self._build_generation_prompt(context)
        
        # K개 후보 생성
        candidates = []
        for i in range(self.K):
            # OpenAI client로 텍스트 생성
            messages = [
                {"role": "system", "content": "You are an empathetic emotional support counselor."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            text = response.choices[0].message.content.strip()
            
            candidates.append({
                "id": i,
                "text": text,
                "generation_info": {
                    "temperature": self.temperature,
                    "prompt_version": "v1.0"
                }
            })
        
        return candidates
    
    def _build_generation_prompt(self, context: Dict[str, Any]) -> str:
        """
        응답 생성 프롬프트 구성
        
        Instruction:
            - 공감적 경청자 역할
            - 안전하고 지지적인 응답
            - 구체적 조언보다 탐색 우선
        """
        parts = []
        
        # System instruction
        parts.append("""You are an empathetic supporter in an emotional support conversation.

Guidelines:
- Show genuine empathy and validation
- Ask open-ended questions to understand better
- Avoid giving direct advice unless the seeker explicitly asks
- Never make assumptions about facts you don't know
- Be especially careful with sensitive topics (health, relationships, mental health)
- If the seeker mentions self-harm or suicide, express concern and suggest professional help

Generate a supportive response based on the conversation below.
""")
        
        # Context
        if context.get("situation"):
            parts.append(f"\n[Initial Situation]\n{context['situation']}\n")
        
        if context.get("state_summary"):
            parts.append(f"\n[Conversation Summary]\n{context['state_summary']}\n")
        
        parts.append("\n[Recent Conversation]")
        for turn in context["recent_turns"]:
            speaker = turn["speaker"].capitalize()
            parts.append(f"{speaker}: {turn['text']}")
        
        parts.append("\nSupporter:")
        
        return "\n".join(parts)
