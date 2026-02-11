"""
Prompts for Data Augmentation: Violation → Normal
===================================================
Generate Normal Turn 4 given prefix + Turn 1-3 context.
This is different from original generation - we're replacing existing violation Turn 4.
"""

# ============================================================================
# System Prompt (Common)
# ============================================================================

AUGMENTATION_SYSTEM = """You are an expert counselor generating normal (non-violating) responses for training data.

You will be given:
1. A conversation prefix (ESConv original dialogue)
2. Already generated Turn 1-3 of a 4-turn insertion

Your task is to generate ONLY Turn 4 (the last supporter response).
This Turn 4 must be a good counseling response with NO violations."""


# ============================================================================
# V1 → Normal (Avoid hasty conclusions)
# ============================================================================

V1_TO_NORMAL_TEMPLATE = """[Input - ESConv Prefix Dialogue]
Situation: {situation}

{prefix_dialog}

[Input - Already Generated Dialogue]
The following 3 turns have already been generated and must NOT be changed:

{generated_turns_123}

[Task]
Generate ONLY Turn 4 (the last supporter response).

IMPORTANT CONTEXT:
- Turn 3 contains UNCERTAINTY or missing information
- The seeker hasn't confirmed all the facts
- You must respond as a good counselor (NO V1 violation)

[Turn 4 Requirements - Avoid V1 Violation]

✅ DO (Good Counseling):
- Ask at least ONE clarifying question about the uncertainty
- Acknowledge what you don't know yet
- Use phrases like: "Can you tell me more about...", "What makes you think...", "Help me understand..."
- Offer gentle next steps WITHOUT assuming missing facts
- Keep it supportive and non-judgmental

❌ DON'T (V1 Violation Patterns to AVOID):
- "Even without knowing all the details..."
- "Regardless of the missing details..."
- "It seems clear that..."
- "You don't need every detail to see that..."
- Jump to conclusions or advice before understanding
- Ignore the uncertainty markers in Turn 3
- Make assumptions about unstated facts

[Output Format - JSON Only]
Return ONLY valid JSON:

{{
  "supporter_response": "<your Turn 4 here>"
}}

Now generate Turn 4 as a normal (non-violating) counseling response."""


# ============================================================================
# V2 → Normal (Avoid coercion, support autonomy)
# ============================================================================

V2_TO_NORMAL_TEMPLATE = """[Input - ESConv Prefix Dialogue]
Situation: {situation}

{prefix_dialog}

[Input - Already Generated Dialogue]
The following 3 turns have already been generated and must NOT be changed:

{generated_turns_123}

[Task]
Generate ONLY Turn 4 (the last supporter response).

IMPORTANT CONTEXT:
- Turn 3 may show the seeker asking for direction or expressing confusion
- You must respond WITHOUT coercion or commands (NO V2 violation)

[Turn 4 Requirements - Avoid V2 Violation]

✅ DO (Good Counseling):
- Offer TWO or more options (A/B) in autonomy-supportive language
- Use phrases like: "You could...", "Some people find it helpful to...", "What feels right to you?"
- Ask preference questions: "Which of these resonates with you?"
- Acknowledge their autonomy: "It's your choice", "Only you can decide"
- Keep it warm and non-judgmental

❌ DON'T (V2 Violation Patterns to AVOID):
- Commands: "You must...", "You have to...", "You need to..."
- Ultimatums: "If you don't... then..."
- Single directive: "Just do this"
- Pressure language: "You should really...", "The best thing is to..."
- Removing user autonomy

[Output Format - JSON Only]
Return ONLY valid JSON:

{{
  "supporter_response": "<your Turn 4 here>"
}}

Now generate Turn 4 as a normal (non-violating) counseling response."""


# ============================================================================
# V3 → Normal (Provide practical help, not just empathy)
# ============================================================================

V3_TO_NORMAL_TEMPLATE = """[Input - ESConv Prefix Dialogue]
Situation: {situation}

{prefix_dialog}

[Input - Already Generated Dialogue]
The following 3 turns have already been generated and must NOT be changed:

{generated_turns_123}

[Task]
Generate ONLY Turn 4 (the last supporter response).

IMPORTANT CONTEXT:
- Turn 3 may contain an explicit request for practical help or guidance
- You must provide both empathy AND concrete help (NO V3 violation)

[Turn 4 Requirements - Avoid V3 Violation]

✅ DO (Good Counseling):
- Start with ONE brief empathy/validation sentence
- Add exactly ONE concrete help element (choose one):
  (a) ONE clarifying question, OR
  (b) TWO options (A/B), OR
  (c) A short 2-3 step practical guide
- Balance emotional support with actionable guidance
- Keep it natural and person-centered

❌ DON'T (V3 Violation Patterns to AVOID):
- Empathy-only responses when practical help is requested
- Generic comfort with no actionable element
- Only saying: "I hear you", "That must be hard"
- Multiple questions (limit to ONE if using questions)
- Ignoring the request for practical guidance

[Output Format - JSON Only]
Return ONLY valid JSON:

{{
  "supporter_response": "<your Turn 4 here>"
}}

Now generate Turn 4 as a normal (non-violating) counseling response."""


# ============================================================================
# V4 → Normal (Question unrealistic beliefs, don't reinforce)
# ============================================================================

V4_TO_NORMAL_TEMPLATE = """[Input - ESConv Prefix Dialogue]
Situation: {situation}

{prefix_dialog}

[Input - Already Generated Dialogue]
The following 3 turns have already been generated and must NOT be changed:

{generated_turns_123}

[Task]
Generate ONLY Turn 4 (the last supporter response).

IMPORTANT CONTEXT:
- Turn 3 may contain an unrealistic or extreme belief
- You must validate FEELINGS without validating the BELIEF (NO V4 violation)

[Turn 4 Requirements - Avoid V4 Violation]

✅ DO (Good Counseling):
- Validate the feeling: "I can see why you'd feel..."
- Ask ONE gentle reality-check question
- Offer alternative interpretations or uncertainty framing
- Use phrases like: "Is it possible that...", "Could there be another explanation...", "What makes you certain..."
- Keep it empathetic while gently questioning

❌ DON'T (V4 Violation Patterns to AVOID):
- Confirming extreme beliefs as fact: "Yes, they definitely...", "You're absolutely right that..."
- Reinforcing unrealistic interpretations
- Agreeing with catastrophic thinking
- Providing step-by-step advice based on the false premise
- Ignoring the problematic belief

[Output Format - JSON Only]
Return ONLY valid JSON:

{{
  "supporter_response": "<your Turn 4 here>"
}}

Now generate Turn 4 as a normal (non-violating) counseling response."""


# ============================================================================
# V5 → Normal (Respond safely to crisis signals)
# ============================================================================

V5_TO_NORMAL_TEMPLATE = """[Input - ESConv Prefix Dialogue]
Situation: {situation}

{prefix_dialog}

[Input - Already Generated Dialogue]
The following 3 turns have already been generated and must NOT be changed:

{generated_turns_123}

[Task]
Generate ONLY Turn 4 (the last supporter response).

IMPORTANT CONTEXT:
- Turn 3 likely contains self-harm or suicide ideation signals
- You must respond with immediate safety protocol (NO V5 violation)

[Turn 4 Requirements - Avoid V5 Violation]

✅ DO (Good Counseling - SAFETY FIRST):
- Ask ONE direct safety question about immediate danger
- Encourage urgent help: emergency services (112/119) or trusted person
- Provide crisis resources: 1393 (suicide prevention), 1577-0199 (mental health)
- Suggest immediate safety step: "Please don't stay alone", "Can you reach out to someone right now?"
- Keep it calm, supportive, and concise

❌ DON'T (V5 Violation Patterns to AVOID):
- Generic empathy without safety check
- Ignoring or minimizing crisis signals
- Proceeding with normal counseling without addressing safety
- Delayed response to immediate danger
- No mention of emergency resources

[Output Format - JSON Only]
Return ONLY valid JSON:

{{
  "supporter_response": "<your Turn 4 here>"
}}

Now generate Turn 4 as a normal (non-violating) counseling response with SAFETY PROTOCOL."""


# ============================================================================
# Prompt Builder Functions
# ============================================================================

def format_dialog_turns(turns):
    """Format dialog turns for prompt display"""
    lines = []
    for i, turn in enumerate(turns):
        speaker = turn.get('speaker', 'unknown')
        text = turn.get('text') or turn.get('content', '')
        lines.append(f"Turn {i} ({speaker}): {text}")
    return "\n".join(lines)


def build_augmentation_prompt(violation_type, situation, prefix_dialog, generated_turns_123):
    """
    Build augmentation prompt for specific violation type
    
    Args:
        violation_type: V1, V2, V3, V4, or V5
        situation: Conversation situation
        prefix_dialog: ESConv prefix turns (list of dicts)
        generated_turns_123: Already generated Turn 0-2 (list of 3 dicts)
    
    Returns:
        Formatted user prompt string
    """
    template_map = {
        'V1': V1_TO_NORMAL_TEMPLATE,
        'V2': V2_TO_NORMAL_TEMPLATE,
        'V3': V3_TO_NORMAL_TEMPLATE,
        'V4': V4_TO_NORMAL_TEMPLATE,
        'V5': V5_TO_NORMAL_TEMPLATE,
    }
    
    template = template_map.get(violation_type)
    if not template:
        raise ValueError(f"Unknown violation type: {violation_type}")
    
    # Format prefix dialog
    prefix_str = format_dialog_turns(prefix_dialog)
    
    # Format generated turns 1-3
    turns_123_str = format_dialog_turns(generated_turns_123)
    
    return template.format(
        situation=situation,
        prefix_dialog=prefix_str,
        generated_turns_123=turns_123_str
    )
