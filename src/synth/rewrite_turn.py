"""
Rewrite a single supporter turn to inject a violation.
Method 2-A: Keep all other turns, rewrite exactly one supporter turn.
"""
import copy
from typing import Dict, Any, List, Optional
from src.llm.openai_client import OpenAIClient
from src.llm.prompts import (
    REWRITE_SYSTEM, 
    build_rewrite_prompt, 
    RETRY_MESSAGE
)


def get_supporter_turn_indices(dialog: List[Dict[str, Any]]) -> List[int]:
    """Get indices of all supporter turns in dialog."""
    return [i for i, turn in enumerate(dialog) if turn.get('speaker') == 'supporter']


def get_supporter_response_index(dialog: List[Dict[str, Any]], turn_idx: int) -> int:
    """
    Get the supporter response index (1-based count of supporter turns).
    
    Args:
        dialog: Full dialog list
        turn_idx: Absolute turn index in dialog
    
    Returns:
        1-based supporter response number (e.g., 1st supporter turn, 2nd supporter turn, etc.)
    """
    supporter_count = 0
    for i in range(turn_idx + 1):
        if dialog[i].get('speaker') == 'supporter':
            supporter_count += 1
    return supporter_count


def select_violation_turn(dialog: List[Dict[str, Any]], 
                         violation_type: str,
                         turn_ranges: Dict[str, List[int]]) -> Optional[int]:
    """
    Select a supporter turn index to inject violation based on allowed range.
    
    Args:
        dialog: Dialog list
        violation_type: V1, V2, V3, V4, or V5
        turn_ranges: Dict mapping violation type to [min, max] supporter response indices
    
    Returns:
        Absolute turn index in dialog, or None if no valid turn found
    """
    import random
    
    supporter_indices = get_supporter_turn_indices(dialog)
    
    if not supporter_indices:
        return None
    
    # Get allowed range for this violation type
    allowed_range = turn_ranges.get(violation_type, [1, 999])
    min_resp, max_resp = allowed_range
    
    # Find supporter turns within the allowed response index range
    valid_indices = []
    for turn_idx in supporter_indices:
        resp_idx = get_supporter_response_index(dialog, turn_idx)
        if min_resp <= resp_idx <= max_resp:
            valid_indices.append(turn_idx)
    
    if not valid_indices:
        # Fallback: pick any supporter turn if no valid range found
        return random.choice(supporter_indices)
    
    return random.choice(valid_indices)


def build_context_for_rewrite(dialog: List[Dict[str, Any]], target_turn_idx: int) -> List[Dict[str, Any]]:
    """
    Build context turns (all turns BEFORE target turn) for rewriting.
    
    Args:
        dialog: Full dialog
        target_turn_idx: Index of turn to rewrite
    
    Returns:
        List of context turns (before target)
    """
    context = []
    for i in range(target_turn_idx):
        turn = dialog[i]
        context.append({
            "speaker": turn.get("speaker"),
            "text": turn.get("content", "")
        })
    return context


def rewrite_session_with_violation(session: Dict[str, Any],
                                   violation_type: str,
                                   turn_ranges: Dict[str, List[int]],
                                   llm_client: OpenAIClient,
                                   verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Create a synthetic session by rewriting one supporter turn.
    
    Args:
        session: Original session dict
        violation_type: V1-V5
        turn_ranges: Violation type -> [min, max] supporter response index
        llm_client: OpenAI client
        verbose: Print debug info
    
    Returns:
        New session with injected violation, or None if failed
    """
    # Deep copy to avoid mutating original
    new_session = copy.deepcopy(session)
    dialog = new_session['dialog']
    
    # Select turn to rewrite
    target_turn_idx = select_violation_turn(dialog, violation_type, turn_ranges)
    
    if target_turn_idx is None:
        if verbose:
            print(f"  Warning: No valid supporter turn found for {violation_type}")
        return None
    
    # Get original response
    original_response = dialog[target_turn_idx].get('content', '')
    
    # Build context
    context_turns = build_context_for_rewrite(dialog, target_turn_idx)
    
    # Build prompt
    situation = new_session.get('situation', '')
    user_prompt = build_rewrite_prompt(
        situation=situation,
        context_turns=context_turns,
        original_response=original_response,
        target_violation=violation_type
    )
    
    # Call LLM
    try:
        result = llm_client.call(
            system_prompt=REWRITE_SYSTEM,
            user_prompt=user_prompt,
            retry_message=RETRY_MESSAGE
        )
        
        rewritten_response = result.get('rewritten_response', '')
        rationale = result.get('rationale_short', '')
        
        # Update dialog with rewritten response
        dialog[target_turn_idx]['content'] = rewritten_response
        
        # Store metadata about injection (DO NOT use in model training input)
        supporter_resp_idx = get_supporter_response_index(dialog, target_turn_idx)
        new_session['injected_violation'] = {
            'type': violation_type,
            'turn_id': target_turn_idx,
            'supporter_utterance_index': supporter_resp_idx,
            'original_text': original_response,
            'rewritten_text': rewritten_response,
            'rationale': rationale
        }
        
        # Update session_id to mark as synthetic
        orig_id = new_session.get('session_id', 'unknown')
        new_session['session_id'] = f"synth_{orig_id}"
        
        return new_session
        
    except Exception as e:
        if verbose:
            print(f"  Error rewriting session: {e}")
        return None
