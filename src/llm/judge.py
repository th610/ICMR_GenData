"""
LLM-judge labeling for turn samples.
Label each supporter response with V1~V5 violations.
"""
from typing import Dict, Any, Optional
from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt, RETRY_MESSAGE


def label_turn_sample(sample: Dict[str, Any], 
                     llm_client: OpenAIClient,
                     verbose: bool = False) -> Optional[Dict[str, Any]]:
    """
    Label one turn sample using LLM-judge.
    
    Args:
        sample: Turn sample dict with context_turns, summary, response
        llm_client: OpenAI client
        verbose: Print debug info
    
    Returns:
        Sample with added labels, top_violation, evidence_span
        None if failed
    """
    # Extract fields
    context_turns = sample.get('context_turns', [])
    response = sample.get('response', '')
    meta = sample.get('meta', {})
    
    # Build situation (use session metadata if available)
    # For simplified PoC, we use problem/emotion type as pseudo-situation
    situation = f"{meta.get('emotion_type', 'unknown')} related to {meta.get('problem_type', 'unknown')}"
    
    # Build judge prompt
    user_prompt = build_judge_prompt(
        situation=situation,
        context_turns=context_turns,
        response=response
    )
    
    # Call LLM judge
    try:
        result = llm_client.call(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=user_prompt,
            retry_message=RETRY_MESSAGE
        )
        
        # Extract labels
        labels = result.get('labels', {})
        top_violation = result.get('top_violation', 'None')
        evidence_span = result.get('evidence_span', '')
        
        # Validate labels structure
        if not isinstance(labels, dict):
            if verbose:
                print(f"  Warning: Invalid labels format: {labels}")
            return None
        
        # Add to sample (create new dict to avoid mutation)
        labeled_sample = sample.copy()
        labeled_sample['labels'] = labels
        labeled_sample['top_violation'] = top_violation
        labeled_sample['evidence_span'] = evidence_span
        
        return labeled_sample
        
    except Exception as e:
        if verbose:
            print(f"  Error labeling sample: {e}")
        return None
