"""
Normal session preparation from ESConv
- Sample sessions from ESConv
- Cut to 15~20 turns
- Last turn must be Supporter
"""
import random
from typing import Dict, List, Any, Optional


def cut_session_to_length(session: Dict[str, Any], 
                          min_turn: int = 15, 
                          max_turn: int = 20,
                          seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Cut ESConv session to target length with last turn as Supporter.
    
    Args:
        session: ESConv session dict
        min_turn: Minimum turn count
        max_turn: Maximum turn count
        seed: Random seed
    
    Returns:
        Session with cut dialog
    """
    if seed is not None:
        random.seed(seed)
    
    dialog = session['dialog']
    
    # If already within range, check last speaker
    if min_turn <= len(dialog) <= max_turn:
        # Check if last turn is supporter
        if dialog[-1]['speaker'] == 'supporter':
            return session
    
    # Cut to random point in range
    target_length = random.randint(min_turn, min(max_turn, len(dialog)))
    
    # Find closest supporter turn at or before target
    cut_point = None
    for i in range(target_length - 1, -1, -1):
        if dialog[i]['speaker'] == 'supporter':
            cut_point = i + 1  # Include this turn
            break
    
    # If no supporter found (very rare), find first supporter
    if cut_point is None:
        for i in range(len(dialog)):
            if dialog[i]['speaker'] == 'supporter':
                cut_point = i + 1
                break
    
    if cut_point is None:
        raise ValueError(f"No supporter turn found in session {session.get('conversation_id', 'unknown')}")
    
    # Create new session with cut dialog
    new_session = session.copy()
    new_session['dialog'] = dialog[:cut_point]
    new_session['original_length'] = len(dialog)
    new_session['cut_length'] = cut_point
    
    return new_session


def prepare_normal_sessions(esconv_sessions: List[Dict[str, Any]],
                           num_sessions: int,
                           target_length: tuple = (15, 20),
                           seed: int = 42) -> List[Dict[str, Any]]:
    """
    Prepare Normal sessions from ESConv.
    
    Args:
        esconv_sessions: List of ESConv sessions
        num_sessions: Number of sessions to prepare
        target_length: (min, max) turn range
        seed: Random seed
    
    Returns:
        List of Normal sessions with metadata
    """
    random.seed(seed)
    
    # Filter sessions that are long enough
    min_turn, max_turn = target_length
    eligible = [s for s in esconv_sessions if len(s.get('dialog', [])) >= min_turn]
    
    if len(eligible) < num_sessions:
        raise ValueError(f"Not enough eligible sessions: need {num_sessions}, have {len(eligible)}")
    
    # Sample sessions
    sampled = random.sample(eligible, num_sessions)
    
    # Cut to target length
    normal_sessions = []
    for i, session in enumerate(sampled):
        cut_session = cut_session_to_length(session, min_turn, max_turn, seed=seed + i)
        
        # Add metadata
        cut_session['primary_label'] = 'Normal'
        cut_session['session_id'] = f"normal_{i:04d}"
        cut_session['generation_method'] = 'esconv_cut'
        cut_session['violation_turn_index'] = None
        cut_session['violation_reason'] = None
        cut_session['source'] = 'ESConv'
        cut_session['processing'] = 'cut_to_length'
        
        normal_sessions.append(cut_session)
    
    return normal_sessions


def get_last_supporter_turn_index(dialog: List[Dict[str, Any]]) -> Optional[int]:
    """Get index of last supporter turn in dialog."""
    for i in range(len(dialog) - 1, -1, -1):
        if dialog[i]['speaker'] == 'supporter':
            return i
    return None


def validate_normal_session(session: Dict[str, Any]) -> bool:
    """
    Validate that a Normal session meets requirements.
    
    Requirements:
    - Has dialog
    - Last turn is supporter
    - Has primary_label = 'Normal'
    """
    if 'dialog' not in session:
        return False
    
    dialog = session['dialog']
    if len(dialog) == 0:
        return False
    
    if dialog[-1]['speaker'] != 'supporter':
        return False
    
    if session.get('primary_label') != 'Normal':
        return False
    
    return True
