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
        'V5': "제가 들으니 정말 힘드실 것 같습니다. 이런 상황에서는 전문가의 도움이 필요할 수 있습니다. 상담 전문가나 관련 기관에 연락하시는 것을 권장드립니다.",
        'V4': "말씀하신 내용을 이해했습니다. 제가 제대로 이해하고 있는지 확인하고 싶은데, 조금 더 자세히 말씀해주시겠어요?",
        'V2': "그런 상황이시군요. 어떻게 하고 싶으신가요? 제가 도움이 될 수 있는 부분이 있을까요?",
        'V1': "제가 상황을 제대로 이해했는지 확인하고 싶습니다. 조금 더 자세히 설명해주실 수 있나요?",
        'V3': "그런 일이 있으셨군요. 더 말씀해주시겠어요?",
        'Unknown': "제가 어떻게 도움을 드릴 수 있을까요?"
    }
    
    # Rewrite 프롬프트 (위반 타입별)
    REWRITE_PROMPTS = {
        'V1': """다음 상담 응답을 수정하세요.

문제: 맥락 이해 부족 - 이전 대화 내용을 충분히 반영하지 못했습니다.

수정 지침:
1. 이전 대화에서 언급된 구체적인 내용을 참조하세요
2. 사용자가 이미 말한 정보를 재확인하는 질문은 피하세요
3. 대화 흐름에 자연스럽게 이어지도록 하세요

원본 응답: {original}

대화 맥락:
{context}

수정된 응답 (한 문장으로):""",
        
        'V2': """다음 상담 응답을 수정하세요.

문제: 자율성 침해 - 강요하거나 일방적으로 결정을 내리려 했습니다.

수정 지침:
1. 강요하는 표현('~해야 한다', '~하세요')을 제거하세요
2. 선택지를 제공하거나 질문 형태로 바꾸세요
3. 사용자의 의견을 존중하는 태도를 보이세요

원본 응답: {original}

수정된 응답 (한 문장으로):""",
        
        'V3': """다음 상담 응답을 수정하세요.

문제: 공감 품질 저하 - 기계적이거나 피상적인 반응입니다.

수정 지침:
1. 사용자의 감정을 구체적으로 언급하세요
2. 일반적인 위로보다 상황에 맞는 공감을 표현하세요
3. 진정성 있고 따뜻한 어조를 사용하세요

원본 응답: {original}

수정된 응답 (한 문장으로):""",
        
        'V4': """다음 상담 응답을 수정하세요.

문제: 현실 왜곡 - 사실이 아니거나 과장된 내용이 있습니다.

수정 지침:
1. 검증되지 않은 사실 주장을 제거하세요
2. 불확실한 내용은 조심스럽게 표현하세요
3. 사용자가 직접 말한 내용만 참조하세요

원본 응답: {original}

수정된 응답 (한 문장으로):""",
        
        'V5': """다음 상담 응답을 수정하세요.

문제: 위기 상황 - 전문가 개입이 필요한 상황입니다.

수정 지침:
1. 전문가 상담을 권유하세요
2. 구체적인 도움 방법(기관, 핫라인 등)을 안내하세요
3. 공감과 함께 즉각적인 조치를 독려하세요

원본 응답: {original}

수정된 응답 (한 문장으로):"""
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
        
        if label == 'Normal':
            return {'type': 'accept'}
        
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
        
        prompt = prompt.format(
            original=original,
            context=context.get('window', '')
        )
        
        try:
            # OpenAI client로 rewrite
            messages = [
                {"role": "system", "content": "당신은 정신건강 상담 전문가입니다."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm.client.chat.completions.create(
                model=self.llm.model,
                messages=messages,
                max_tokens=200,
                temperature=0.7
            )
            
            rewritten = response.choices[0].message.content.strip()
            return rewritten
        except Exception as e:
            print(f"   ⚠️  Rewrite failed: {e}")
            return self._rule_based_rewrite(original, violation_type)
    
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
