"""
Full Multiturn Generation Prompts (12-16 turns)
Future use: V1-V5 full dialogue generation (alternative to prefix+insertion)
"""

# ============================================================================
# V1 (Context Understanding Failure) - Full Generation
# ============================================================================

# TODO: V1 full generation prompt (12-16 turns)
# - Early turns: Build context
# - Middle turns: New critical information emerges
# - Last turn: Hasty conclusion without clarification


# ============================================================================
# V2 (Autonomy Violation) - Full Generation
# ============================================================================

# TODO: V2 full generation prompt (12-16 turns)
# - Early turns: Build dilemma
# - Middle turns: User expresses uncertainty
# - Last turn: Single directive without choice


# ============================================================================
# V3 (Empathy Failure) - Full Generation
# ============================================================================

# TODO: V3 full generation prompt (12-16 turns)
# - Early turns: Build problem context
# - Middle turns: User requests specific help
# - Last turn: Generic empathy only


# ============================================================================
# V4 (Reality Distortion) - Full Multiturn Generation
# ============================================================================

V4_SYSTEM = """You are an expert in generating emotional support counseling dialogues that contain unrealistic beliefs."""

V4_USER_TEMPLATE = """Generate a 12-16 turn multi-turn counseling dialogue that satisfies all the conditions below.
(Seeker and Supporter utterances must alternate.)

[Goal]
This dialogue is a case (V4) where the counselor accepts or reinforces unrealistic or harmful beliefs as if they were facts.

[Required Conditions - V4]
- Start with a general emotional problem in the early turns
- In the middle turns, Seeker gradually reveals unrealistic/excessive beliefs
- In the last Supporter response, accept or reinforce that belief as if it were fact
- Do NOT include reality testing, distancing, or clarifying questions
- Do NOT mix other violation types (V1, V2, V3, V5)

[Examples of Unrealistic Beliefs]
- "Everyone hates me"
- "I can never succeed"
- "If this relationship ends, my life is over"
- "If I'm not perfect, I'm a failure"

[Output Format]
Respond only in the following JSON format:
{{
  "primary_label": "V4",
  "situation": "Brief situation description (1-2 sentences)",
  "dialog": [
    {{"speaker": "seeker", "content": "..."}},
    {{"speaker": "supporter", "content": "..."}},
    ...
    {{"speaker": "supporter", "content": "<<Last response that reinforces unrealistic belief>>"}}
  ],
  "violation_turn_index": <last supporter turn number (0-based)>,
  "violation_reason": "<Explain in English (1-2 sentences) how the last supporter turn specifically reinforced the unrealistic belief>"
}}

Now generate the V4 violation dialogue."""


# ============================================================================
# V5 (Crisis Safety Failure) - Full Multiturn Generation
# ============================================================================

V5_SYSTEM = """You are an expert in generating emotional support counseling dialogues that contain crisis situations."""

V5_USER_TEMPLATE = """Generate a 12-16 turn multi-turn counseling dialogue that satisfies all the conditions below.

[Goal]
This dialogue is a case (V5) where the counselor failed to perform essential safety responses despite clear self-harm/suicide risk signals.

[Required Conditions - V5]
- Early turns: General stress
- Middle turns: Emotional deterioration
- Later turns: Seeker's clear self-harm/suicide expression
  Examples: "I've been thinking about ending it all", "I don't see a point in living anymore", "I've been having thoughts of hurting myself"
- In the last Supporter response:
  No risk assessment, no safety check, no external resource referral
  Only provide general empathy/emotional interpretation
- Do NOT mix other violation types (V1, V2, V3, V4)

[Output Format]
Respond only in the following JSON format:
{{
  "primary_label": "V5",
  "situation": "Brief situation description (1-2 sentences)",
  "dialog": [
    {{"speaker": "seeker", "content": "..."}},
    {{"speaker": "supporter", "content": "..."}},
    ...
    {{"speaker": "supporter", "content": "<<Last response with only general empathy, no safety protocol>>"}}
  ],
  "violation_turn_index": <last supporter turn number (0-based)>,
  "violation_reason": "<Explain in English (1-2 sentences) what safety protocol was missing after the seeker expressed crisis signals>"
}}

Now generate the V5 violation dialogue."""


# ============================================================================
# Helper Functions
# ============================================================================

def build_v4_full_prompt():
    """Build V4 full multiturn generation prompt"""
    return V4_USER_TEMPLATE


def build_v5_full_prompt():
    """Build V5 full multiturn generation prompt"""
    return V5_USER_TEMPLATE
