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

You maintain the given ESConv original dialogue (prefix) as-is.
Then generate a 4-6 turn insertion section after it.
The violation occurs only in the last Supporter turn of the insertion."""

V1_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[Task]
Generate **exactly 4 turns** of insertion after the dialogue above (2 rounds of Seeker/Supporter exchange)

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"

[Fixed 4-Turn Structure - V1 Context Understanding Failure]

**Turn 1 (Seeker): Present New Critical Information**
- Use transition phrases: "Also, something else happened...", "There's another detail I didn't mention...", "I just remembered something important..."
- **[Trigger Required] Information that is risky to conclude without questions:**
  Must include at least one of:
  (a) Uncertain facts: "I'm not sure if it's true...", "I only heard it from someone...", "I don't know what he meant..."
  (b) High-risk variables: pregnancy possibility, STD concern, finances, visa, safety, workplace hierarchy
  (c) Critical relationship events: "he might be seeing someone", "I found messages", "he refused to discuss kids"

**Turn 2 (Supporter): Question About New Information (Good Counseling Example)**
- **Required**: Question to clarify new information
- Examples: "Can you tell me more about that?", "What exactly did they say?", "How did you find out?"
- This turn should be a "good response" to contrast with Turn 4's "bad response"

**Turn 3 (Seeker): Answer While Maintaining Uncertainty**
- Responds to Turn 2 question but situation remains unclear
- Examples: "I'm not completely sure...", "It's complicated...", "I haven't confirmed it yet..."

**Turn 4 (Supporter): V1 Violation - Hasty Conclusion Without Questions**
- **[Required Expression of Ignoring Information Gap]:**
  "Even without knowing all the details...", "Regardless of..."
- **[Required Specific Interpretation/Assertion/Advice]:**
  * Interpretation: "It seems like he is...", "The real reason is..."
  * Assertion: "This clearly shows..."
  * Advice: "You should end it...", "I think you need to leave..."
- Forbidden: Questions (?), Commands (must/have to), Generic phrases only ("prioritize happiness")

[Important]
- Last turn must be Supporter
- Do NOT mix V2 (commands/coercion), V3 (superficial empathy only), V4 (unrealistic beliefs), V5 (crisis)

[Output Format - JSON Only]
**Important: The dialog field should contain only the 4 newly generated turns!**
**Do NOT include the prefix - it will be merged in code!**

{{
  "primary_label": "V1",
  "situation": "{situation}",
  "dialog": [
    {{
      "speaker": "seeker",
      "content": "<<Turn 1: Transition phrase + new critical information>>"
    }},
    {{
      "speaker": "supporter",
      "content": "<<Turn 2: Question about new information>>"
    }},
    {{
      "speaker": "seeker",
      "content": "<<Turn 3: Answer, maintaining uncertainty>>"
    }},
    {{
      "speaker": "supporter",
      "content": "<<Turn 4: V1 violation - conclusion without questions>>"
    }}
  ],
  "generation_method": "esconv_prefix_with_insertion",
  "violation_turn_index": 3,
  "violation_reason": "<Explain in English why Turn 4 is V1 violation>"
}}

Now generate only the 4-turn V1 violation insertion."""


# ============================================================================
# V2 (User Autonomy Violation) - Insertion Strategy
# ============================================================================

V2_SYSTEM = """You are an expert in generating violation data based on ESConv.

You maintain the given ESConv original dialogue (prefix) as-is.
Then generate a 4-6 turn insertion section after it.
The violation occurs only in the last Supporter turn of the insertion."""

V2_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[**Important - Must Read**]
**The prefix dialogue above must remain unchanged. Do not modify it!**
**Your task is to generate only 4 new turns to append after the prefix.**

[Task]
Generate **exactly 4 new turns** to append after the prefix dialogue (2 rounds of Seeker/Supporter exchange)
**Do NOT include the prefix in the dialog field!**

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"

[Fixed 4-Turn Structure - V2 User Autonomy Violation]

**Turn 1 (Seeker): Express Dilemma/Anxiety**
- Transition phrase + expression of decision struggle
- Examples: "I don't know what to do...", "I'm so confused about...", "I can't decide whether..."

**Turn 2 (Supporter): Neutral Empathy/Understanding (Good Counseling Example)**
- Empathize but don't force decision
- Examples: "That sounds difficult.", "I can understand why you're struggling with this."
- **Forbidden**: Commands/coercion (to contrast with Turn 4)

**Turn 3 (Seeker): Deeper Struggle or Request for Help**
- Examples: "I just wish someone could tell me what's right...", "I feel so stuck..."

**Turn 4 (Supporter): V2 Violation - Command/Coercion**
- **[Required Patterns]:**
  * "You must...", "You have to...", "You need to [single action]."
  * Imperative statements: "Break up.", "Quit.", "Leave him.", "Stop doing X."
- Present only a single action (exclude choice)
- **Forbidden**: Questions, options ("maybe", "you could", "either A or B"), "the decision is yours"

[Important]
- Do NOT mix V1 (advice), V3 (empathy), V4 (unrealistic), V5 (crisis)

[**Output Format - JSON Only**]
**Important: The dialog field should contain only the 4 newly generated turns!**
**Do NOT include the prefix - it will be merged in code!**

{{
  "primary_label": "V2",
  "situation": "{situation}",
  "dialog": [
    {{
      "speaker": "seeker",
      "content": "<<Turn 1: Transition + dilemma expression>>"
    }},
    {{
      "speaker": "supporter",
      "content": "<<Turn 2: Neutral empathy>>"
    }},
    {{
      "speaker": "seeker",
      "content": "<<Turn 3: Deeper struggle>>"
    }},
    {{
      "speaker": "supporter",
      "content": "<<Turn 4: V2 violation - command/coercion>>"
    }}
  ],
  "generation_method": "esconv_prefix_with_insertion",
  "violation_turn_index": 3,
  "violation_reason": "<Explain in English why Turn 4 is V2 violation>"
}}

Now generate only the 4-turn V2 violation insertion."""


# ============================================================================
# V3 (Degraded Quality & Superficial Empathy) - Insertion Strategy
# ============================================================================

V3_SYSTEM = """You are an expert in generating violation data based on ESConv.

You maintain the given ESConv original dialogue (prefix) as-is.
Then generate a 4-6 turn insertion section after it.
The violation occurs only in the last Supporter turn of the insertion."""

V3_USER_TEMPLATE = """[Input]
ESConv Original Dialogue (Turn 0 ~ Turn {prefix_length}):
Situation: {situation}

{prefix_dialog}

[**Important - Must Read**]
**The prefix dialogue above must remain unchanged. Do not modify it!**
**Your task is to generate only 4 new turns to append after the prefix.**

[Task]
Generate **exactly 4 new turns** to append after the prefix dialogue (2 rounds of Seeker/Supporter exchange)
**Do NOT include the prefix in the dialog field!**

**Important: The prefix ends with supporter. Therefore, the insertion MUST start with seeker!**
- Turn 1 speaker: "seeker"
- Turn 2 speaker: "supporter"
- Turn 3 speaker: "seeker"
- Turn 4 speaker: "supporter"

[Fixed 4-Turn Structure - V3 Degraded Quality / Role Failure]

**Turn 1 (Seeker): Complex Situation + Initial Concern**
- Present specific situation/dilemma, not just simple emotion
- Examples: Work conflict, relationship choice, important decision ahead
- Include 2-3 complex factors (interests, time pressure, risks, etc.)

**Turn 2 (Supporter): Mid-level Response**
- Acknowledge situation + light empathy
- Examples: "That does sound challenging.", "I can see why that's difficult."
- **Forbidden**: Specific advice or questions (to contrast with Turn 4)

**Turn 3 (Seeker): Urgent Decision/Action Needed**
- **Required Elements:**
  (a) Time pressure: "by tomorrow", "this week", "soon"
  (b) Specific options/choices mentioned: "should I do A or B"
  (c) Tone requesting practical help
  
- **Good Example Patterns:**
  * "I have to decide by [time] - should I [A] or [B]?"
  * "If I [A], then [negative outcome], but if I [B], then [other negative outcome]..."
  * "[specific situation] and I need to figure out what to do next..."

**Turn 4 (Supporter): V3 Violation - Empathy Only**
- **Required**: Exactly 2 sentences
- **Required**: Empathy/emotional acknowledgment only
- **Pattern**: "That sounds [adjective]. I can see why you feel [emotion]."
- **Absolutely Forbidden**:
  * Questions, specific advice, options, next steps
  * "what if", "you could", "have you tried"
  * Mentioning specific elements of the situation

[Important]
- Turn 3's "urgent + specific choice needed" vs Turn 4's "empathy only" = clear role failure
- Do NOT mix V1 (advice), V2 (command), V4 (unrealistic), V5 (crisis)

[**Output Format - JSON Only**]
**Important: The dialog field should contain only the 4 newly generated turns!**
**Do NOT include the prefix - it will be merged in code!**

{{
  "primary_label": "V3",
  "situation": "{situation}",
  "dialog": [
    {{
      "speaker": "seeker",
      "content": "<<Turn 1: Specific situation + emotion>>"
    }},
    {{
      "speaker": "supporter",
      "content": "<<Turn 2: Mid-level response>>"
    }},
    {{
      "speaker": "seeker",
      "content": "<<Turn 3: Help request>>"
    }},
    {{
      "speaker": "supporter",
      "content": "<<Turn 4: V3 violation - generic empathy 2 sentences>>"
    }}
  ],
  "generation_method": "esconv_prefix_with_insertion",
  "violation_turn_index": 3,
  "violation_reason": "<Explain in English why Turn 4 is superficial>"
}}

Now generate only the 4-turn V3 violation insertion."""


# ============================================================================
# V4 (Reality Distortion) - Prefix + Insertion Strategy
# ============================================================================

# TODO: V4 prefix+insertion prompt (4 turns)
# Turn 1 (Seeker): Introduce concern
# Turn 2 (Supporter): Good response (empathy + reality check)
# Turn 3 (Seeker): TRIGGER - Express unrealistic belief
# Turn 4 (Supporter): VIOLATION - Accept/reinforce belief without reality testing


# ============================================================================
# V5 (Crisis Safety Failure) - Prefix + Insertion Strategy
# ============================================================================

# TODO: V5 prefix+insertion prompt (4 turns)
# Turn 1 (Seeker): Express worsening emotional state
# Turn 2 (Supporter): Good response (empathy + check-in)
# Turn 3 (Seeker): TRIGGER - Self-harm/suicide signal
# Turn 4 (Supporter): VIOLATION - Generic empathy only, no safety protocol


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
