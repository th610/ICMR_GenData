"""
Controller V2 - Policy Layer with Rewrite & Fail-safe

Priority: V5 > V4 > V2 > V1 > V3 > Normal

Actions:
    - accept: Normal, 그대로 통과
    - rewrite: V1-V5, 위반 수정
    - failsafe: 재시도 실패, 안전 응답
"""
from typing import Dict, Any, Optional


class ControllerV2:
    """정책 레이어: 위반 탐지 결과에 따른 행동 결정"""
    
    # 위반 우선순위 (높을수록 위험)
    PRIORITY = {
        'V5': 5,  # 위기 안전 (최우선)
        'V4': 4,  # 현실 왜곡
        'V2': 3,  # 자율성 침해
        'V1': 2,  # 맥락 이해 부족
        'V3': 1,  # 공감 품질
        'Normal': 0
    }
    
    # Fail-safe 템플릿 (위반 타입별)
    FAILSAFE_TEMPLATES = {
        'V5': "지금 당장 스스로를 해칠 위험이 있나요? 위험이 있다면 지금 바로 112/119에 연락하거나 가까운 사람에게 도움을 요청해주세요. 1393(자살예방상담전화), 1577-0199(정신건강상담전화)도 24시간 운영됩니다.",
        'general': "I'm really sorry you're going through this. To help in a practical way, what feels most urgent right now, and what outcome are you hoping for?",
        'V4': "말씀하신 내용을 이해했습니다. 제가 제대로 이해하고 있는지 확인하고 싶은데, 조금 더 자세히 말씀해주시겠어요?",
        'V2': "그런 상황이시군요. 어떻게 하고 싶으신가요? 제가 도움이 될 수 있는 부분이 있을까요?",
        'V1': "제가 상황을 제대로 이해했는지 확인하고 싶습니다. 조금 더 자세히 설명해주실 수 있나요?",
        'V3': "그런 일이 있으셨군요. 더 말씀해주시겠어요?",
        'Unknown': "제가 어떻게 도움을 드릴 수 있을까요?"
    }
    
    # Rewrite 프롬프트 (위반 타입별) - Reasoning 포함
    REWRITE_PROMPTS = {
        'V1': """You are fixing the LAST supporter response to remove a counseling violation (V1: hasty conclusion without clarifying questions).

Conversation (WINDOW):
{window}

Original last supporter response:
{original}

Fix requirements:
- In 1 sentence, acknowledge the user's new key information or uncertainty.
- Ask exactly ONE clarifying question (must contain a "?").
- Then offer 1–2 gentle next steps WITHOUT assuming missing facts.
- Do NOT use commands ("must/have to/need to").

Output format (JSON):
{{
  "reasoning": "Explain what violation you detected and how you fixed it (1-2 sentences)",
  "corrected_response": "<your corrected response>"
}}""",
        
        'V2': """You are fixing the LAST supporter response to remove a counseling violation (V2: coercion/command that removes user autonomy).

Conversation (WINDOW):
{window}

Original last supporter response:
{original}

Fix requirements:
- Remove any commanding language ("must/have to/need to") and ultimatums.
- Offer TWO realistic options (A/B) in autonomy-supportive language.
- Ask exactly ONE preference question (must contain a "?").
- Keep it warm and non-judgmental.

Output format (JSON):
{{
  "reasoning": "Explain what violation you detected and how you fixed it (1-2 sentences)",
  "corrected_response": "<your corrected response>"
}}""",
        
        'V3': """You are fixing the LAST supporter response to remove a counseling violation (V3: empathy-only despite an explicit request for practical help).

Conversation (WINDOW):
{window}

Original last supporter response:
{original}

Fix requirements:
- Keep ONE brief empathy/validation sentence.
- Add exactly ONE concrete help element (choose only one):
  (a) ONE clarifying question, OR
  (b) TWO options (A/B), OR
  (c) a short 2–3 step guide.
- Do NOT add multiple questions (at most one "?").

Output format (JSON):
{{
  "reasoning": "Explain what violation you detected and how you fixed it (1-2 sentences)",
  "corrected_response": "<your corrected response>"
}}""",
        
        'V4': """You are fixing the LAST supporter response to remove a counseling violation (V4: reinforcing an unrealistic/extreme belief as fact).

Conversation (WINDOW):
{window}

Original last supporter response:
{original}

Fix requirements:
- Do NOT confirm the extreme belief as true.
- Validate the feeling without validating the belief.
- Ask exactly ONE gentle reality-check question (must contain a "?").
- Add one grounded alternative framing (e.g., uncertainty, another possible interpretation).
- Do NOT give step-by-step advice or commands.

Output format (JSON):
{{
  "reasoning": "Explain what violation you detected and how you fixed it (1-2 sentences)",
  "corrected_response": "<your corrected response>"
}}""",
        
        'V5': """You are fixing the LAST supporter response to remove a counseling violation (V5: failing to respond safely after explicit self-harm/suicide ideation).

Conversation (WINDOW):
{window}

Original last supporter response:
{original}

Fix requirements (must follow):
- Ask ONE direct safety question (must contain a "?") about immediate danger.
- Encourage urgent help: emergency services or a trusted person nearby.
- Provide crisis resources in Korea: 1393, 1577-0199, and emergency 112/119.
- Suggest one short safety step (e.g., do not stay alone).
- Keep calm, supportive, and concise.

Output format (JSON):
{{
  "reasoning": "Explain what violation you detected and how you fixed it (1-2 sentences)",
  "corrected_response": "<your corrected response>"
}}"""
    }
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Args:
            llm_client: Rewrite용 LLM (OpenAIClient 등)
        """
        self.llm = llm_client
    
    def decide(self, detection: Dict[str, Any]) -> Dict[str, str]:
        """
        위반 탐지 결과에 따른 행동 결정
        
        Args:
            detection: {'label': str, 'confidence': float}
        
        Returns:
            {'type': 'accept' | 'rewrite' | 'failsafe'}
        """
        label = detection['label']
        confidence = detection['confidence']
        
        # 대소문자 구분 없이 Normal 체크
        if label.lower() == 'normal':
            return {'type': 'accept'}
        
        # Confidence가 너무 낮으면 (borderline case) accept
        # 예: 25% 미만이면 모델이 확신하지 못하는 것
        # if confidence < 0.25:
        #     return {'type': 'accept'}
        
        # 위반 발견 → rewrite 시도
        return {'type': 'rewrite', 'violation': label}
    
    def rewrite(self, 
                original: str, 
                violation_type: str, 
                context: Dict[str, Any]) -> str:
        """
        위반 타입별 응답 수정
        
        Args:
            original: 원본 응답
            violation_type: v1, v2, v3, v4, v5 (소문자)
            context: 대화 맥락
        
        Returns:
            수정된 응답
        """
        import json
        
        # 대문자로 변환
        violation_key = violation_type.upper()
        
        if not self.llm:
            # LLM 없으면 규칙 기반 간단 수정
            return self._rule_based_rewrite(original, violation_key)
        
        # LLM 기반 수정
        prompt = self.REWRITE_PROMPTS.get(violation_key, "")
        if not prompt:
            print(f"   ⚠️  No rewrite prompt for {violation_key}")
            return self._rule_based_rewrite(original, violation_key)
        
        # WINDOW 구성 (JSON 형식)
        window_turns = context.get('recent_turns', [])
        window_list = []
        for turn in window_turns:
            window_list.append({
                "speaker": turn["speaker"],
                "content": turn.get("content") or turn.get("text", "")
            })
        window_json = json.dumps(window_list, ensure_ascii=False, indent=2)
        
        prompt = prompt.format(
            original=original,
            window=window_json
        )
        
        try:
            # OpenAI client로 rewrite
            messages = [
                {"role": "system", "content": "You are an expert counselor fixing violations."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                max_tokens=300,
                temperature=0.7
            )
            
            raw_response = response.choices[0].message.content.strip()
            
            # JSON 파싱 시도
            try:
                import json as json_lib
                parsed = json_lib.loads(raw_response)
                reasoning = parsed.get("reasoning", "")
                rewritten = parsed.get("corrected_response", raw_response)
                
                # Reasoning 반환 (디버깅용)
                return {
                    "text": rewritten,
                    "reasoning": reasoning,
                    "raw": raw_response
                }
            except json_lib.JSONDecodeError:
                # JSON 파싱 실패 시 원본 사용
                return {
                    "text": raw_response,
                    "reasoning": "",
                    "raw": raw_response
                }
        except Exception as e:
            print(f"   ⚠️  Rewrite failed: {e}")
            fallback = self._rule_based_rewrite(original, violation_type)
            return {
                "text": fallback,
                "reasoning": f"LLM failed, used rule-based: {e}",
                "raw": fallback
            }
    
    def _rule_based_rewrite(self, original: str, violation_type: str) -> str:
        """규칙 기반 간단 수정"""
        if violation_type == 'V2':
            # 명령형 → 질문형
            if '하세요' in original or '해야' in original:
                return "어떻게 하고 싶으신가요?"
        
        elif violation_type == 'V1':
            # 맥락 부족 → 재확인
            return "제가 제대로 이해했는지 확인하고 싶어요. 조금 더 말씀해주시겠어요?"
        
        return original
    
    def failsafe(self, violation_type: str, context: Dict[str, Any]) -> str:
        """
        재시도 실패 시 안전 응답 반환
        
        Args:
            violation_type: 마지막 위반 타입
            context: 대화 맥락
        
        Returns:
            안전한 템플릿 응답
        """
        return self.FAILSAFE_TEMPLATES.get(
            violation_type, 
            self.FAILSAFE_TEMPLATES['Unknown']
        )
