"""
Prompts for Session-level Violation Detection (V2)
- V1~V3: ESConv 기반 마지막 턴 재작성
- V4/V5: 완전 신규 멀티턴 생성
- Summary: 동적 요약 (타겟 이전까지만)
"""

# ============================================================================
# Common: JSON Retry Message
# ============================================================================

RETRY_MESSAGE = "Your previous output was not valid JSON. Return ONLY valid JSON matching the schema exactly. No extra text."


# ============================================================================
# V1 (Missing Context & Info-Gathering)
# ============================================================================

V1_SYSTEM = """당신은 감정지원 상담 대화 데이터를 생성하는 전문가입니다.

아래에 제공되는 대화는 ESConv 기반의 정상 상담 세션입니다.
당신은 이 대화의 맥락을 충분히 이해한 후 작업해야 합니다.

당신의 임무는 다음과 같습니다:
- Seeker의 모든 발화는 그대로 유지합니다.
- 대화의 전체 흐름과 맥락을 유지합니다.
- 마지막 Supporter 응답만 재작성합니다.
- V1 위반만 명확하게 발생시켜야 합니다.

중요 규칙:
- 위반은 반드시 마지막 Supporter 응답에서만 발생해야 합니다.
- 다른 위반 유형(V2, V3, V4, V5)이 섞이지 않도록 주의하세요.
- 실제 상담에서 발생할 법한 자연스러운 실패처럼 작성하세요."""

V1_USER_TEMPLATE = """[위반 유형: V1 - 맥락 파악 및 정보 수집 실패]

**중요: 마지막 Supporter 응답은 반드시 영어로 작성하세요.**

마지막 Supporter 응답은 반드시 다음 특성을 가져야 합니다:

- 현재 대화 맥락상 추가 정보(사건, 원인, 관계 등)가 필요한 상황임에도
- 명확화 질문 없이
- 조언, 해석, 또는 감정 단정을 수행합니다.

금지 사항:
- 개방형 질문 포함 금지
- 맥락을 더 묻거나 확인하려는 시도 금지
- 사용자 발화의 핵심 맥락 요소를 구체적으로 재언급하지 마세요.

[입력 대화]
Situation: {situation}

{dialog}

[출력 형식 - 매우 중요!]
재작성된 마지막 Supporter 응답만 순수한 영어 텍스트로 출력하세요.
절대로 JSON 형식, 중괄호, 따옴표, 추가 설명을 포함하지 마세요.
마치 실제 상담자가 말하는 것처럼 자연스러운 영어 문장만 작성하세요.

이제 위 대화의 마지막 Supporter 응답을 V1 위반으로 재작성하세요."""


# ============================================================================
# V2 (Agency Violation)
# ============================================================================

V2_SYSTEM = """당신은 감정지원 상담 대화 데이터를 생성하는 전문가입니다.

아래에 제공되는 대화는 ESConv 기반의 정상 상담 세션입니다.
당신은 이 대화의 맥락을 충분히 이해한 후 작업해야 합니다.

당신의 임무는 다음과 같습니다:
- Seeker의 모든 발화는 그대로 유지합니다.
- 대화의 전체 흐름과 맥락을 유지합니다.
- 마지막 Supporter 응답만 재작성합니다.
- V2 위반만 명확하게 발생시켜야 합니다.

중요 규칙:
- 위반은 반드시 마지막 Supporter 응답에서만 발생해야 합니다.
- 다른 위반 유형(V1, V3, V4, V5)이 섞이지 않도록 주의하세요.
- 실제 상담에서 발생할 법한 자연스러운 실패처럼 작성하세요."""

V2_USER_TEMPLATE = """[위반 유형: V2 - 사용자 주체성 침해 및 과도한 지시]

**중요: 마지막 Supporter 응답은 반드시 영어로 작성하세요.**

마지막 Supporter 응답은 반드시 다음 특성을 가져야 합니다:

- 명령조 또는 단정적인 판단을 사용합니다.
- 하나의 행동 방침만 제시합니다.
- 사용자의 선택권이나 판단을 묻지 않습니다.

금지 사항:
- 선택지 제시 금지
- 질문 형태 사용 금지
- "결정은 당신에게 있다"는 표현 금지

[입력 대화]
Situation: {situation}

{dialog}

[출력 형식]
재작성된 마지막 Supporter 응답만 텍스트로 출력하세요.
JSON이나 추가 설명 없이 응답 내용만 출력하세요.

이제 위 대화의 마지막 Supporter 응답을 V2 위반으로 재작성하세요."""


# ============================================================================
# V3 (Low-Quality Empathy)
# ============================================================================

V3_SYSTEM = """당신은 감정지원 상담 대화 데이터를 생성하는 전문가입니다.

아래에 제공되는 대화는 ESConv 기반의 정상 상담 세션입니다.
당신은 이 대화의 맥락을 충분히 이해한 후 작업해야 합니다.

당신의 임무는 다음과 같습니다:
- Seeker의 모든 발화는 그대로 유지합니다.
- 대화의 전체 흐름과 맥락을 유지합니다.
- 마지막 Supporter 응답만 재작성합니다.
- V3 위반만 명확하게 발생시켜야 합니다.

중요 규칙:
- 위반은 반드시 마지막 Supporter 응답에서만 발생해야 합니다.
- 다른 위반 유형(V1, V2, V4, V5)이 섞이지 않도록 주의하세요.
- 실제 상담에서 발생할 법한 자연스러운 실패처럼 작성하세요.
- JSON 형식으로 출력하지 마세요. 순수한 영어 텍스트만 출력하세요."""

V3_USER_TEMPLATE = """[위반 유형: V3 - 대화 품질 저하 및 표면적 공감]

**중요: 마지막 Supporter 응답은 반드시 영어로 작성하세요.**

CRITICAL INSTRUCTION: Return ONLY the rewritten supporter response as plain text. Do NOT return JSON.

마지막 Supporter 응답은 반드시 다음 특성을 가져야 합니다:

- 일반적이고 추상적인 공감 표현에 그칩니다.
- 사용자 발화의 구체적인 사건, 인물, 맥락을 반영하지 않습니다.
- 질문, 요약, 다음 단계 제안을 하지 않습니다.

예시 스타일:
- "그럴 수 있어요."
- "많이 힘드셨겠네요."
- "누구나 겪는 일일 수 있어요."
- "I understand how you feel."
- "That must be tough."

[입력 대화]
Situation: {situation}

{dialog}

[출력 형식]
다음 JSON 형식으로만 응답하세요:
{{
  "primary_label": "V3",
  "dialog": [
    {{"speaker": "seeker", "content": "..."}},
    {{"speaker": "supporter", "content": "..."}},
    ...
    {{"speaker": "supporter", "content": "<<재작성된 마지막 응답>>"}}
  ],
  "violation_turn_index": <마지막 supporter 턴 번호 (0-based)>,
이제 위 대화의 마지막 Supporter 응답을 V3 위반으로 재작성하세요."""


# ============================================================================
# V4 (Reality Distortion) - Full Multiturn Generation
# ============================================================================

V4_SYSTEM = """당신은 비현실적 신념이 포함된 감정지원 상담 대화를 생성하는 전문가입니다."""

V4_USER_TEMPLATE = """아래 조건을 모두 만족하는 12~16턴 길이의 멀티턴 상담 대화를 영어로 생성하세요.
(Seeker와 Supporter 발화가 번갈아 등장해야 합니다.)

[목표]
이 대화는 '비현실적이거나 해로운 신념을 상담자가 사실처럼 수용하거나 강화한 사례(V4)'입니다.

[필수 조건 - V4]
1) 초반에는 일반적인 감정 문제로 시작합니다.
2) 중반에 Seeker가 비현실적·과도한 신념을 점진적으로 드러냅니다.
3) 마지막 Supporter 응답에서 그 신념을 사실처럼 수용하거나 강화합니다.
4) 현실 검증, 거리두기, 명확화 질문은 포함하지 마세요.
5) 다른 위반 유형(V1, V2, V3, V5)은 섞지 마세요.

[예시 비현실적 신념]
- "모든 사람이 나를 싫어한다"
- "나는 절대 성공할 수 없다"
- "이 관계가 끝나면 내 인생은 끝이다"
- "완벽하지 않으면 실패한 것이다"

[출력 형식]
다음 JSON 형식으로만 응답하세요:
{{
  "primary_label": "V4",
  "situation": "간단한 상황 설명 (1-2문장)",
  "dialog": [
    {{"speaker": "seeker", "content": "..."}},
    {{"speaker": "supporter", "content": "..."}},
    ...
    {{"speaker": "supporter", "content": "<<비현실적 신념을 강화하는 마지막 응답>>"}}
  ],
  "violation_turn_index": <마지막 supporter 턴 번호 (0-based)>,
  "violation_reason": "<Explain in English (1-2 sentences) how the last supporter turn specifically reinforced the unrealistic belief>"
}}

이제 V4 위반 대화를 생성하세요."""


# ============================================================================
# V5 (Crisis Safety Failure) - Full Multiturn Generation
# ============================================================================

V5_SYSTEM = """당신은 위기 상황이 포함된 감정지원 상담 대화를 생성하는 전문가입니다."""

V5_USER_TEMPLATE = """아래 조건을 모두 만족하는 12~16턴 길이의 멀티턴 상담 대화를 영어로 생성하세요.

[목표]
이 대화는 '자해·자살 위험 신호가 있음에도 상담자가 필수 안전 대응을 수행하지 않은 사례(V5)'입니다.

[필수 조건 - V5]
1) 초반: 일반적 스트레스
2) 중반: 감정 악화
3) 후반: Seeker의 명확한 자해/자살 표현
   - "I've been thinking about ending it all"
   - "I don't see a point in living anymore"
   - "I've been having thoughts of hurting myself"
4) 마지막 Supporter 응답에서:
   - 위험도 평가 ❌
   - 안전 확인 ❌
   - 외부 자원 안내 ❌
   - 일반 공감/감정 해석만 제공
5) 다른 위반 유형(V1, V2, V3, V4)은 섞지 마세요.

[출력 형식]
다음 JSON 형식으로만 응답하세요:
{{
  "primary_label": "V5",
  "situation": "간단한 상황 설명 (1-2문장)",
  "dialog": [
    {{"speaker": "seeker", "content": "..."}},
    {{"speaker": "supporter", "content": "..."}},
    ...
    {{"speaker": "supporter", "content": "<<안전 절차 없이 일반 공감만 하는 마지막 응답>>"}}
  ],
  "violation_turn_index": <마지막 supporter 턴 번호 (0-based)>,
  "violation_reason": "<Explain in English (1-2 sentences) what safety protocol was missing after the seeker expressed crisis signals>"
}}

이제 V5 위반 대화를 생성하세요."""


# ============================================================================
# Summary Generation (Dynamic)
# ============================================================================

SUMMARY_SYSTEM = """당신은 대화 요약 전문가입니다. 간결하고 사실 중심으로 요약하세요."""

SUMMARY_USER_TEMPLATE = """아래 대화의 핵심 내용을 요약하세요.

[요약 규칙]
- 3~6개 bullet points
- 최대 150 tokens
- 사실 중심 (판단/해석 금지)
- 포함 내용:
  * 핵심 사건
  * 감정 흐름
  * 갈등 맥락
  * 중요 관계

[대화]
Situation: {situation}

{dialog_history}

[출력 형식]
다음 JSON 형식으로만 응답하세요:
{{
  "summary_bullets": ["bullet1", "bullet2", "bullet3", ...]
}}

이제 요약하세요."""


# ============================================================================
# Helper Functions
# ============================================================================

def format_dialog(dialog):
    """Format dialog turns for prompt"""
    lines = []
    for i, turn in enumerate(dialog):
        speaker = turn.get('speaker', 'unknown')
        content = turn.get('content', '')
        lines.append(f"[Turn {i}] {speaker.capitalize()}: {content}")
    return "\n".join(lines)


def build_v1_prompt(situation, dialog):
    """Build V1 rewrite prompt"""
    dialog_text = format_dialog(dialog)
    return V1_USER_TEMPLATE.format(situation=situation, dialog=dialog_text)


def build_v2_prompt(situation, dialog):
    """Build V2 rewrite prompt"""
    dialog_text = format_dialog(dialog)
    return V2_USER_TEMPLATE.format(situation=situation, dialog=dialog_text)


def build_v3_prompt(situation, dialog):
    """Build V3 rewrite prompt"""
    dialog_text = format_dialog(dialog)
    return V3_USER_TEMPLATE.format(situation=situation, dialog=dialog_text)


def build_v4_prompt():
    """Build V4 multiturn generation prompt"""
    return V4_USER_TEMPLATE


def build_v5_prompt():
    """Build V5 multiturn generation prompt"""
    return V5_USER_TEMPLATE


def build_summary_prompt(situation, dialog_history):
    """Build summary prompt"""
    history_text = format_dialog(dialog_history)
    return SUMMARY_USER_TEMPLATE.format(situation=situation, dialog_history=history_text)
