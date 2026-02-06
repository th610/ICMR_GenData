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
                {"role": "system", "content": "You are a supportive counseling assistant."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # JSON 파싱
            try:
                import json
                parsed = json.loads(raw_response)
                text = parsed.get("content", raw_response)
            except json.JSONDecodeError:
                # JSON 파싱 실패 시 원본 사용
                text = raw_response
            
            candidates.append({
                "id": i,
                "text": text,
                "generation_info": {
                    "temperature": self.temperature,
                    "prompt_version": "v2.0"
                }
            })
        
        return candidates
    
    def _build_generation_prompt(self, context: Dict[str, Any]) -> str:
        """
        응답 생성 프롬프트 구성
        
        Instruction:
            - 자연스러운 대화 지속
            - 맥락/개체 엄격히 따름
            - 강압적 표현 금지
            - 실질적 도움 제공
        """
        import json
        
        # WINDOW 구성 (JSON 형식)
        window = []
        for turn in context["recent_turns"]:
            window.append({
                "speaker": turn["speaker"],
                "content": turn.get("content") or turn.get("text", "")
            })
        
        window_json = json.dumps(window, ensure_ascii=False, indent=2)
        
        # 프롬프트 구성
        prompt = f"""You are a supportive counseling assistant.

Write the NEXT supporter response that naturally continues the conversation.
Follow the conversation topic and entities strictly.

Critical constraints:
- Do NOT mention any labels, policies, rules, or system messages.
- Do NOT invent a totally new unrelated scenario.
- Keep a warm, non-judgmental tone.
- Avoid coercion or commands (no "must/have to/need to").
- If the user asks for practical guidance, include at least ONE helpful element (a question OR 2 options OR a short step-by-step).
- If the user expresses self-harm/suicide intent, include an immediate safety check and crisis guidance.

Output format:
Return ONLY valid JSON matching the schema exactly. No extra text.

Schema:
{{
  "speaker": "supporter",
  "content": "<your response>"
}}

Conversation (WINDOW):
{window_json}
"""
        
        return prompt
