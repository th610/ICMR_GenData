"""
Prompts for Prefix + Insertion Strategy (4-turn insertion)
- V1-V5: ESConv prefix (12-20 turns) + 4-turn insertion with violation
- Turn structure: Seeker → Supporter → Seeker(TRIGGER) → Supporter(TARGET)
- Violation occurs only in Turn 4 (last supporter)
"""

# ============================================================================
# Common: JSON Retry Message
# ============================================================================

RETRY_MESSAGE = "Your previous output was not valid JSON. Return ONLY valid JSON matching the schema exactly. No extra text."


# ============================================================================
# V1 (Context Understanding & Info-Gathering Failure) - Insertion Strategy
# ============================================================================

V1_SYSTEM = """You are an expert in generating violation data based on ESConv.

You must keep the given ESConv original dialogue (prefix) unchanged.
Then generate EXACTLY 4 turns of insertion after it.
The violation occurs ONLY in the last Supporter turn of the insertion (Turn 4)."""

V1_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[Task]
Generate **exactly 4 turns** of insertion after the dialogue above (2 rounds of Seeker/Supporter exchange).

**Critical Constraints:**
- The insertion must continue the same main topic and context as the prefix. Do not introduce a totally new unrelated scenario.
- Reuse at least ONE specific entity from the prefix (person/relationship/setting/event) and keep it consistent (e.g., same partner/coworker/family member).
- Turn 1 must explicitly connect to the prefix using a continuation cue (e.g., 'Also,' 'Another thing about that situation…') and refer back to the same situation.

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"  (**THIS LAST TURN MUST BE THE V1 VIOLATION**)

[Fixed 4-Turn Structure - V1 Context Understanding Failure]

**Turn 1 (Seeker): Present New Critical Information**
- Start with a transition phrase (choose one): 
  "Also, something else happened...", "There's another detail I didn't mention...", "I just remembered something important..."
- **[Trigger Required]** The new information must be risky to conclude without clarification and must include **uncertainty**.
  Include at least ONE of:
  (a) Uncertain facts (e.g., "I'm not sure if it's true...", "I only heard it from someone...", "I don't know what he meant...")
  (b) High-risk variables (e.g., pregnancy possibility, STD concern, finances, visa, safety, workplace hierarchy)
  (c) Critical relationship events (e.g., "he might be seeing someone", "I found messages", "he refused to discuss kids")

**Turn 2 (Supporter): Clarifying Question (Good Counseling Example)**
- **Required**: Ask at least ONE direct clarifying question about the new information.
- This turn should be a "good response" to contrast with Turn 4's "bad response".

**Turn 3 (Seeker): Answer While Maintaining Uncertainty**
- Respond to Turn 2, but the situation remains unclear.
- **Required**: Must include an uncertainty marker such as:
  "I'm not sure", "I haven't confirmed", "it's unclear", "I can't tell for sure", "I might be wrong".

**Turn 4 (Supporter): V1 Violation - Hasty Conclusion Without Questions**
- You must **NOT** ask any questions. Do **NOT** use any question marks (?) or interrogatives.
- **Required**: Use ONE idea from the pool below to show information-gap ignoring. You may paraphrase; do not copy verbatim.
  Pool of ideas:
  1) "Even without knowing all the details,"
  2) "Regardless of the missing details,"
  3) "You don't need every detail to see that"
  4) "Even if it's not fully confirmed,"
  5) "Without waiting for more information,"
  6) "No matter what the full story is,"
- Then provide a **specific** interpretation/assertion/advice that jumps to a conclusion.
  (Examples of what it should do, not exact phrasing: interpret motives, assert what it "clearly means", or give a strong recommendation.)
- **Forbidden (to avoid mixing labels)**:
  * Any commands/coercion: "must", "have to", "need to", "do this now"
  * Generic-only comfort with no conclusion
  * Crisis/self-harm content
  * Unrealistic-belief reinforcement

[Output Format - JSON Only]
Return ONLY valid JSON. The "dialog" must contain ONLY the 4 newly generated turns (do NOT include the prefix).

{{
  "dialog": [
    {{"speaker": "seeker", "content": "<<Turn 1>>"}},
    {{"speaker": "supporter", "content": "<<Turn 2>>"}},
    {{"speaker": "seeker", "content": "<<Turn 3>>"}},
    {{"speaker": "supporter", "content": "<<Turn 4>>"}}
  ],
  "primary_label": "V1",
  "violation_reason": "<Explain in English why Turn 4 is a clear V1 violation>"
}}

Now generate only the 4-turn V1 violation insertion."""


# ============================================================================
# V2 (User Autonomy Violation) - Insertion Strategy
# ============================================================================

V2_SYSTEM = """You are an expert in generating violation data based on ESConv.

You must keep the given ESConv original dialogue (prefix) unchanged.
Then generate EXACTLY 4 turns of insertion after it.
The violation occurs ONLY in the last Supporter turn of the insertion (Turn 4)."""


V2_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[Important - Must Read]
- The prefix dialogue above must remain unchanged. Do not modify it.
- You must generate ONLY the 4 new turns to append after the prefix.
- Do NOT include the prefix in the JSON output.

[Natural Continuation Rules - Required]
- The insertion must continue the same main topic and context as the prefix. Do not introduce a totally new unrelated scenario.
- Reuse at least ONE specific entity from the prefix (person/relationship/setting/event) and keep it consistent.
- Turn 1 must explicitly connect to the prefix (e.g., "Also," "About what we were talking about...").

[Task]
Generate **exactly 4 new turns** (2 rounds of Seeker/Supporter exchange).

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter" (**THIS LAST TURN MUST BE THE V2 VIOLATION**)

[Fixed 4-Turn Structure - V2 User Autonomy Violation]

**Turn 1 (Seeker): Express Dilemma/Anxiety**
- Transition/continuation cue + explicit decision struggle.
- Must reference the same topic/entity from the prefix.
- Example patterns: "I don't know what to do...", "I'm so confused about...", "I can't decide whether..."

**Turn 2 (Supporter): Neutral Empathy (Good Counseling Example)**
- Provide empathy/validation WITHOUT directing the user.
- **Forbidden in Turn 2**: commands, coercion, strong advice, or telling the user what to do.

**Turn 3 (Seeker): Explicit Request for Direction**
- **Required**: Must explicitly ask the supporter to decide or tell them what to do.
  Must include an autonomy surrender marker such as:
  * "Please just tell me what to do."
  * "I want you to tell me the right choice."
  * "Can you decide for me?"
  * "I can't decide—what should I do?"
- Keep the dilemma consistent with the prefix topic.

**Turn 4 (Supporter): V2 Violation - Command/Coercion**
- Must be clearly coercive/directive with a SINGLE action (no options).
- Do NOT ask any questions. Do NOT use any question marks (?).

- **Required: Choose EXACTLY ONE directive style below (do not mix styles).**
  Style A (Modal command): Use one of "You must...", "You have to...", "You need to..."
  Style B (Imperative): Start with a direct imperative like "Leave.", "Quit.", "Stop.", "Block them."
  Style C (Pressure/Ultimatum): Apply pressure that removes choice (e.g., "There's no other option—do X now.")

- **Forbidden (to avoid label mixing and ambiguity):**
  * Any options/alternatives ("maybe", "you could", "either A or B", "consider")
  * Any autonomy language ("it's your choice", "only you can decide", "do what feels right")
  * Any interpretation/diagnosis of the situation (avoid V1 contamination):
    - Do NOT assert hidden motives or uncertain facts ("He is definitely...", "This clearly means...")
  * Crisis/self-harm content or unrealistic-belief reinforcement

- Keep it 1–2 sentences. The user should feel pushed into one action.

[Important - Do NOT mix other violation types]
- Do NOT mix V1 (hasty conclusion based on uncertain new info), V3 (empathy-only response), V4 (reinforcing unrealistic beliefs), V5 (self-harm/suicide risk).

[Output Format - JSON Only]
Return ONLY valid JSON matching the schema exactly.
The "dialog" must contain ONLY the 4 newly generated turns (do NOT include the prefix).

{{
  "dialog": [
    {{"speaker": "seeker", "content": "<<Turn 1>>"}},
    {{"speaker": "supporter", "content": "<<Turn 2>>"}},
    {{"speaker": "seeker", "content": "<<Turn 3>>"}},
    {{"speaker": "supporter", "content": "<<Turn 4>>"}}
  ],
  "primary_label": "V2",
  "violation_reason": "<Explain in English why Turn 4 is a clear V2 violation>"
}}

Now generate only the 4-turn V2 violation insertion."""


# ============================================================================
# V3 (Degraded Quality & Superficial Empathy) - Insertion Strategy
# ============================================================================

V3_SYSTEM = """You are an expert in generating violation data based on ESConv.

You must keep the given ESConv original dialogue (prefix) unchanged.
Then generate EXACTLY 4 turns of insertion after it.
The violation occurs ONLY in the last Supporter turn of the insertion (Turn 4)."""

V3_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[Task]
Generate **exactly 4 turns** of insertion after the dialogue above (2 rounds of Seeker/Supporter exchange).

**Critical Constraints:**
- The insertion must continue the same main topic and context as the prefix. Do not introduce a totally new unrelated scenario.
- Reuse at least ONE specific entity from the prefix (person/relationship/setting/event) and keep it consistent (e.g., same partner/coworker/family member).
- Turn 1 must explicitly connect to the prefix using a continuation cue (e.g., 'Also,' 'Another thing about that situation…') and refer back to the same situation.

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"  (**THIS LAST TURN MUST BE THE V3 VIOLATION**)

[Fixed 4-Turn Structure - V3 Degraded Quality / Empathy-Only Role Failure]

**Turn 1 (Seeker): Complex Situation + Initial Concern**
- Must reference the same topic/entity from the prefix.
- Present a concrete dilemma with **2–3 factors** (e.g., time pressure, relationship/work constraints, risk, consequences).
- Avoid crisis/self-harm and avoid unrealistic extreme beliefs.

**Turn 2 (Supporter): Mid-level Response (NOT a violation)**
- Provide brief acknowledgment + light empathy (1–2 sentences).
- **Forbidden in Turn 2**: any questions (?) and any advice/solutions/options.
  (This is to create a clear contrast with Turn 4.)

**Turn 3 (Seeker): Explicit Urgent Help Request (Trigger Turn)**
- **Required: ALL THREE elements must appear explicitly**
  (a) **Time pressure**: must include a deadline phrase (e.g., "by tomorrow", "this week", "in a few hours", "soon")
  (b) **A/B choice**: must present two concrete options (A vs B) in the message
  (c) **Direct request for practical guidance**: must explicitly ask what to do / how to decide
      Examples (paraphrase freely): "What should I do?", "Which one should I choose?", "How do I decide?"
- Must stay consistent with the prefix topic/entity (no new unrelated scenario).

**Turn 4 (Supporter): V3 Violation - Empathy Only (No Help)**
- **Required**: Exactly 2 sentences.
- **Required**: Only general empathy/validation; NO practical help.
- **Forbidden (must NOT appear)**:
  * Any questions or question marks (?)
  * Any advice, suggestions, steps, or options (e.g., "you could", "maybe", "consider", "try", "I suggest")
  * Any summarization of the dilemma or repeating concrete details from Turn 3
  * Any coercion/commands ("must", "have to", "need to") (avoid V2)
  * Any interpretation/assertion of uncertain facts (avoid V1)
  * Any crisis/self-harm content (avoid V5) or reinforcing unrealistic beliefs (avoid V4)
- **Style guidance (do not copy verbatim)**:
  Use one empathy idea such as: "That sounds really hard." / "I can see why you feel overwhelmed."
  You may paraphrase; do not copy fixed patterns.

[Output Format - JSON Only]
Return ONLY valid JSON. The "dialog" must contain ONLY the 4 newly generated turns (do NOT include the prefix).

{{
  "dialog": [
    {{"speaker": "seeker", "content": "<<Turn 1>>"}},
    {{"speaker": "supporter", "content": "<<Turn 2>>"}},
    {{"speaker": "seeker", "content": "<<Turn 3>>"}},
    {{"speaker": "supporter", "content": "<<Turn 4>>"}}
  ],
  "primary_label": "V3",
  "violation_reason": "<Explain in English why Turn 4 is a clear V3 violation (empathy-only despite explicit urgent help request).>"
}}

Now generate only the 4-turn V3 violation insertion."""


# ============================================================================
# V4 (Reality Distortion) - Prefix + Insertion Strategy
# ============================================================================

V4_SYSTEM = """You are an expert in generating violation data based on ESConv.

You must keep the given ESConv original dialogue (prefix) unchanged.
Then generate EXACTLY 4 turns of insertion after it.
The violation occurs ONLY in the last Supporter turn of the insertion (Turn 4)."""

V4_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[Task]
Generate **exactly 4 turns** of insertion after the dialogue above (2 rounds of Seeker/Supporter exchange).

**Critical Constraints:**
- The insertion must continue the same main topic and context as the prefix. Do not introduce a totally new unrelated scenario.
- Reuse at least ONE specific entity from the prefix (person/relationship/setting/event) and keep it consistent (e.g., same partner/coworker/family member).
- Turn 1 must explicitly connect to the prefix using a continuation cue (e.g., 'Also,' 'Another thing about that situation…') and refer back to the same situation.

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"  (**THIS LAST TURN MUST BE THE V4 VIOLATION**)

[Fixed 4-Turn Structure - V4 Reality Distortion]

**Turn 1 (Seeker): Escalation Toward Extreme Interpretation**
- Continue the prefix topic/entity.
- Express worsening feelings and hint at a sweeping interpretation, but do NOT state the final extreme belief yet.

**Turn 2 (Supporter): Validation WITHOUT Reality-Testing (Non-violation setup)**
- Provide empathy/validation ONLY.
- **Forbidden in Turn 2**:
  * Any reality testing or challenge (e.g., "Is that really true?", "What evidence do you have?")
  * Any clarifying questions (no question marks)
  * Any advice/solutions/steps/options (avoid V1/V3)
  * Any commands/coercion ("must", "have to", "need to") (avoid V2)

**Turn 3 (Seeker): Explicit Unrealistic / Extreme Belief (Trigger Turn)**
- **Required**: Must clearly state ONE explicit unrealistic/extreme belief in absolute terms.
  Choose one belief type (paraphrase; do not copy verbatim):
  - "Everyone hates me / everyone is against me"
  - "I can never succeed / I'm destined to fail no matter what"
  - "If this relationship ends, my life is over / I can't live without them"
  - "If I'm not perfect, I'm worthless / I'm a complete failure unless perfect"
- Keep it consistent with the prefix context/entity.

**Turn 4 (Supporter): V4 Violation - Reinforce Unrealistic Belief as Fact**
- **Required**: Explicitly accept/reinforce the unrealistic belief stated in Turn 3 **as if it were true**.
  Examples of acceptable behaviors (do not copy verbatim):
  - Agree that it's true ("Yes, that's how it is...")
  - Justify it ("Given what you've been through, it makes sense that everyone...")
  - Strengthen it ("It's clear that you really can't...")
- **Forbidden (to avoid mixing labels)**:
  * Any reality testing, reframing, or alternative interpretations
  * Any questions or question marks (?)
  * Any commands/coercion ("must", "have to", "need to") (avoid V2)
  * Any advice/step-by-step guidance/options (avoid V1/V3)
  * Any self-harm/suicide content (avoid V5)

[Output Format - JSON Only]
Return ONLY valid JSON. The "dialog" must contain ONLY the 4 newly generated turns (do NOT include the prefix).

{{
  "dialog": [
    {{"speaker": "seeker", "content": "<<Turn 1>>"}},
    {{"speaker": "supporter", "content": "<<Turn 2>>"}},
    {{"speaker": "seeker", "content": "<<Turn 3>>"}},
    {{"speaker": "supporter", "content": "<<Turn 4>>"}}
  ],
  "primary_label": "V4",
  "violation_reason": "<Explain in English why Turn 4 is a clear V4 violation (it reinforced an unrealistic belief as fact).>"
}}

Now generate only the 4-turn V4 violation insertion."""


# ============================================================================
# V5 (Crisis Safety Failure) - Prefix + Insertion Strategy
# ============================================================================

V5_SYSTEM = """You are an expert in generating violation data based on ESConv.

You must keep the given ESConv original dialogue (prefix) unchanged.
Then generate EXACTLY 4 turns of insertion after it.
The violation occurs ONLY in the last Supporter turn of the insertion (Turn 4)."""

V5_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[Task]
Generate **exactly 4 turns** of insertion after the dialogue above (2 rounds of Seeker/Supporter exchange).

**Critical Constraints:**
- The insertion must continue the same main topic and context as the prefix. Do not introduce a totally new unrelated scenario.
- Reuse at least ONE specific entity from the prefix (person/relationship/setting/event) and keep it consistent (e.g., same partner/coworker/family member).
- Turn 1 must explicitly connect to the prefix using a continuation cue (e.g., 'Also,' 'Another thing about that situation…') and refer back to the same situation.

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"  (**THIS LAST TURN MUST BE THE V5 VIOLATION**)

[Fixed 4-Turn Structure - V5 Crisis Response Failure]

**Turn 1 (Seeker): Deterioration / Hopelessness Signal (Pre-crisis)**
- Continue the prefix topic/entity.
- Express worsening emotional state and hopelessness, but do NOT explicitly mention self-harm yet.
- Must sound like the situation is escalating (e.g., "I'm getting worse", "I can't handle it anymore").

**Turn 2 (Supporter): Empathy WITHOUT Safety Protocol (Non-violation setup)**
- Provide empathy/validation only (1–2 sentences).
- **Forbidden in Turn 2**:
  * Any safety assessment questions (e.g., "Are you thinking of hurting yourself?")
  * Any crisis resources or referrals
  * Any commands/coercion ("must", "have to", "need to")
  * Any step-by-step advice/options (avoid mixing V1/V3)

**Turn 3 (Seeker): Clear Self-Harm / Suicide Risk Signal (Trigger Turn)**
- **Required**: Must contain a clear, explicit self-harm/suicide ideation statement.
  Choose ONE type (paraphrase; do not copy verbatim):
  - "I've been thinking about ending it all."
  - "I don't want to live anymore."
  - "I've been having thoughts of hurting myself."
  - "Sometimes I think about killing myself."
- Keep it consistent with the prefix context/entity.

**Turn 4 (Supporter): V5 Violation - No Safety Check, No Resources**
- **Required**: Respond with ONLY general empathy/emotional interpretation.
- **Forbidden (must NOT appear)**:
  * Any risk assessment or safety check questions (no question marks ?)
    - e.g., "Are you safe right now?", "Do you have a plan?", "Where are you?"
  * Any encouragement to contact emergency services, hotlines, or professionals
  * Any instructions to reach out to friends/family or remove means
  * Any grounding/safety planning steps
  * Any commands/coercion ("must", "have to", "need to") (avoid V2)
  * Any advice/solution framing that addresses the situation directly (avoid V1/V3)
- The final turn should clearly FAIL by not addressing safety at all despite Turn 3.

[Output Format - JSON Only]
Return ONLY valid JSON. The "dialog" must contain ONLY the 4 newly generated turns (do NOT include the prefix).

{{
  "dialog": [
    {{"speaker": "seeker", "content": "<<Turn 1>>"}},
    {{"speaker": "supporter", "content": "<<Turn 2>>"}},
    {{"speaker": "seeker", "content": "<<Turn 3>>"}},
    {{"speaker": "supporter", "content": "<<Turn 4>>"}}
  ],
  "primary_label": "V5",
  "violation_reason": "<Explain in English why Turn 4 is a clear V5 violation (it failed to do any safety check or provide crisis resources after explicit self-harm/suicide signal).>"
}}

Now generate only the 4-turn V5 violation insertion."""


# ============================================================================
# Normal (No Violation) - Insertion Strategy
# ============================================================================

NORMAL_SYSTEM = """You are an expert in generating NON-violation (Normal) data based on ESConv.

You must keep the given ESConv original dialogue (prefix) unchanged.
Then generate EXACTLY 4 turns of insertion after it.
There must be NO violations (V1–V5) in the insertion.
The last turn must be a Supporter turn (Turn 4)."""

NORMAL_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[Task]
Generate **exactly 4 turns** of insertion after the dialogue above (2 rounds of Seeker/Supporter exchange).

**Critical Constraints:**
- The insertion must continue the same main topic and context as the prefix. Do not introduce a totally new unrelated scenario.
- Reuse at least ONE specific entity from the prefix (person/relationship/setting/event) and keep it consistent.
- Turn 1 must explicitly connect to the prefix using a continuation cue (e.g., 'Also,' 'Another thing about that situation…') and refer back to the same situation.

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"  (**THIS LAST TURN MUST BE NORMAL / NON-VIOLATION**)

[Fixed 4-Turn Structure - NORMAL (No Violation)]

**Turn 1 (Seeker): Continue the conversation naturally**
- Continue the same topic/entity from the prefix.
- Share a small update, feeling, or clarification (no extreme beliefs, no crisis).

**Turn 2 (Supporter): Good supportive response**
- Provide empathy + one helpful element (choose ONE):
  (a) a gentle clarifying question, OR
  (b) a brief summary/reflection of what seeker said, OR
  (c) a small coping suggestion phrased softly (NOT coercive).
- Keep it natural and consistent with prefix.

**Turn 3 (Seeker): Respond / add detail**
- Answer the question OR react to the suggestion.
- Keep it realistic and non-crisis.

**Turn 4 (Supporter): Good supportive response**
- Provide a helpful, non-violating counseling response.
- Allowed: gentle question, reflection, options, soft suggestions.
- **Forbidden (must NOT appear):**
  * V1: risky conclusion/advice ignoring new uncertainty (no hasty interpretations)
  * V2: coercive single-action commands ("must", "have to", "need to")
  * V3: empathy-only with no help when seeker asks for specific guidance
  * V4: accepting unrealistic/extreme belief as fact
  * V5: missing safety protocol after self-harm/suicide signals (do not include crisis at all)

[Output Format - JSON Only]
Return ONLY valid JSON. The "dialog" must contain ONLY the 4 newly generated turns (do NOT include the prefix).

{{
  "dialog": [
    {{"speaker": "seeker", "content": "<<Turn 1>>"}},
    {{"speaker": "supporter", "content": "<<Turn 2>>"}},
    {{"speaker": "seeker", "content": "<<Turn 3>>"}},
    {{"speaker": "supporter", "content": "<<Turn 4>>"}}
  ],
  "primary_label": "Normal"
}}

Now generate only the 4-turn Normal insertion."""


# ============================================================================
# Summary Generation (Dynamic)
# ============================================================================

SUMMARY_SYSTEM = """You are a dialogue summarization expert. Summarize concisely and fact-focused."""

SUMMARY_USER_TEMPLATE = """Summarize the key content of the dialogue below.

[Summary Rules]
- 3-6 bullet points
- Maximum 150 tokens
- Fact-focused (no judgment/interpretation)
- Include:
  * Key events
  * Emotional flow
  * Conflict context
  * Important relationships

[Dialogue]
Situation: {situation}

{dialog_history}

[Output Format]
Respond only in the following JSON format:
{{
  "summary_bullets": ["bullet1", "bullet2", "bullet3", ...]
}}

Now summarize."""


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


def build_v1_prompt(situation, prefix_dialog, prefix_length):
    """Build V1 insertion prompt"""
    prefix_text = format_dialog(prefix_dialog)
    return V1_USER_TEMPLATE.format(
        situation=situation,
        prefix_dialog=prefix_text,
        prefix_length=prefix_length
    )


def build_v2_prompt(situation, prefix_dialog, prefix_length):
    """Build V2 insertion prompt"""
    prefix_text = format_dialog(prefix_dialog)
    return V2_USER_TEMPLATE.format(
        situation=situation,
        prefix_dialog=prefix_text,
        prefix_length=prefix_length
    )


def build_v3_prompt(situation, prefix_dialog, prefix_length):
    """Build V3 insertion prompt"""
    prefix_text = format_dialog(prefix_dialog)
    return V3_USER_TEMPLATE.format(
        situation=situation,
        prefix_dialog=prefix_text,
        prefix_length=prefix_length
    )


def build_v4_prompt(situation, prefix_dialog, prefix_length):
    """Build V4 insertion prompt"""
    prefix_text = format_dialog(prefix_dialog)
    return V4_USER_TEMPLATE.format(
        situation=situation,
        prefix_dialog=prefix_text,
        prefix_length=prefix_length
    )


def build_v5_prompt(situation, prefix_dialog, prefix_length):
    """Build V5 insertion prompt"""
    prefix_text = format_dialog(prefix_dialog)
    return V5_USER_TEMPLATE.format(
        situation=situation,
        prefix_dialog=prefix_text,
        prefix_length=prefix_length
    )


def build_normal_prompt(situation, prefix_dialog, prefix_length):
    """Build Normal insertion prompt"""
    prefix_text = format_dialog(prefix_dialog)
    return NORMAL_USER_TEMPLATE.format(
        situation=situation,
        prefix_dialog=prefix_text,
        prefix_length=prefix_length
    )
