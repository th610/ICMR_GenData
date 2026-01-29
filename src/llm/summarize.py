"""
Summarization functions for conversation history.
Supports both LLM-based and rule-based summarization.
"""
from typing import List, Dict, Any, Optional
from src.llm.openai_client import OpenAIClient
from src.llm.prompts import SUMMARY_SYSTEM, build_summary_prompt, RETRY_MESSAGE


def rule_based_summary(dialog_history: List[Dict[str, Any]], max_turns: int = 4) -> List[str]:
    """
    Create a simple rule-based summary without LLM.
    Takes last N seeker turns + last supporter turn.
    
    Args:
        dialog_history: List of turns (past only)
        max_turns: Maximum turns to include
    
    Returns:
        List of summary bullets (simple concatenation)
    """
    if not dialog_history:
        return ["No conversation history."]
    
    # Get last few turns
    recent = dialog_history[-max_turns:] if len(dialog_history) > max_turns else dialog_history
    
    bullets = []
    for turn in recent:
        speaker = turn.get('speaker', 'unknown')
        text = turn.get('text', turn.get('content', ''))[:100]  # Truncate long texts
        bullets.append(f"{speaker}: {text}")
    
    return bullets[:5]  # Max 5 bullets


def llm_based_summary(situation: str, 
                     dialog_history: List[Dict[str, Any]],
                     llm_client: OpenAIClient) -> Optional[List[str]]:
    """
    Create LLM-based summary (3-5 bullets).
    
    Args:
        situation: Session situation text
        dialog_history: List of turns (past only)
        llm_client: OpenAI client
    
    Returns:
        List of summary bullets, or None if failed
    """
    try:
        user_prompt = build_summary_prompt(situation, dialog_history)
        result = llm_client.call(
            system_prompt=SUMMARY_SYSTEM,
            user_prompt=user_prompt,
            retry_message=RETRY_MESSAGE
        )
        
        bullets = result.get('summary_bullets', [])
        return bullets if bullets else None
        
    except Exception as e:
        print(f"  Warning: LLM summary failed: {e}")
        return None


def generate_summary(situation: str,
                    dialog_history: List[Dict[str, Any]],
                    use_llm: bool = False,
                    llm_client: Optional[OpenAIClient] = None) -> List[str]:
    """
    Generate summary using LLM or rule-based approach.
    
    Args:
        situation: Session situation
        dialog_history: Past turns only
        use_llm: If True, use LLM; if False, use rules
        llm_client: OpenAI client (required if use_llm=True)
    
    Returns:
        List of summary bullets
    """
    if use_llm:
        if llm_client is None:
            raise ValueError("llm_client required when use_llm=True")
        
        summary = llm_based_summary(situation, dialog_history, llm_client)
        if summary:
            return summary
        else:
            # Fallback to rule-based
            return rule_based_summary(dialog_history)
    else:
        return rule_based_summary(dialog_history)
