"""
V4-V5 Session Generator
처음부터 전체 대화를 LLM으로 생성 (ESConv 사용 안 함)
"""
import os
import json
import random
from typing import List, Dict, Tuple
from openai import OpenAI
from src.llm.prompts import V4_SYSTEM, V4_USER_TEMPLATE, V5_SYSTEM, V5_USER_TEMPLATE


def generate_v4_v5_session(
    violation_type: str,
    session_idx: int,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> Dict:
    """
    V4 또는 V5 세션을 처음부터 생성
    
    Args:
        violation_type: V4 또는 V5
        session_idx: 세션 인덱스 (session_id 생성용)
        model: OpenAI model name
        temperature: LLM temperature
    
    Returns:
        Generated session
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    
    # Build prompt
    if violation_type == "V4":
        system_prompt = V4_SYSTEM
        user_prompt = V4_USER_TEMPLATE
    elif violation_type == "V5":
        system_prompt = V5_SYSTEM
        user_prompt = V5_USER_TEMPLATE
    else:
        raise ValueError(f"Unknown violation type: {violation_type}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call LLM
    print(f"    Calling LLM... (this may take 30-60 seconds)")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,  # 전체 대화 생성이므로 더 많은 토큰 필요
        timeout=120  # 2분 타임아웃
    )
    
    response_text = response.choices[0].message.content.strip()
    print(f"    LLM response received ({len(response_text)} chars)")
    
    # Parse JSON
    try:
        data = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"    ❌ JSON parsing failed!")
        print(f"    Error: {e}")
        print(f"    Response preview: {response_text[:500]}...")
        raise ValueError(f"Failed to parse JSON response: {e}")
    
    # Validate
    if 'dialog' not in data or len(data['dialog']) == 0:
        raise ValueError(f"No dialog in generated session")
    
    if data['dialog'][-1]['speaker'] != 'supporter':
        raise ValueError(f"Last turn is not supporter: {data['dialog'][-1]['speaker']}")
    
    # Add metadata
    session = {
        'situation': data.get('situation', ''),
        'dialog': data['dialog'],
        'primary_label': violation_type,
        'generation_method': 'full_multiturn',
        'session_id': f"{violation_type.lower()}_{session_idx:04d}",
        'source': 'llm_generated',
        'violation_turn_index': len(data['dialog']) - 1,  # 마지막 턴
        'violation_reason': data.get('violation_reason', f'{violation_type} violation in last supporter turn')
    }
    
    return session


def prepare_v4_v5_sessions(
    violation_type: str,
    num_sessions: int,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> List[Dict]:
    """
    V4 또는 V5 세션 여러 개 생성
    
    Args:
        violation_type: V4 또는 V5
        num_sessions: 생성할 세션 개수
        model: OpenAI model name
        temperature: LLM temperature
    
    Returns:
        List of generated sessions
    """
    sessions = []
    for i in range(num_sessions):
        print(f"  Generating {violation_type}_{i:04d}...")
        
        session = generate_v4_v5_session(
            violation_type,
            i,
            model,
            temperature
        )
        
        sessions.append(session)
    
    return sessions


def validate_v4_v5_session(session: Dict, violation_type: str) -> bool:
    """V4-V5 세션 검증"""
    if 'dialog' not in session or len(session['dialog']) == 0:
        return False
    
    if session['dialog'][-1]['speaker'] != 'supporter':
        return False
    
    if session.get('primary_label') != violation_type:
        return False
    
    return True
