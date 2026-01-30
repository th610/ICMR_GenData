"""
V1-V3 Session Rewriter
ESConv 세션의 마지막 supporter turn을 LLM으로 rewrite
"""
import os
import random
from typing import List, Dict, Tuple
from openai import OpenAI
from src.llm.prompts_v2 import build_v1_prompt, build_v2_prompt, build_v3_prompt, format_dialog
from src.generation.normal_cutter import cut_session_to_length


def rewrite_last_turn(
    session: Dict,
    violation_type: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> str:
    """
    마지막 supporter turn을 LLM으로 rewrite
    
    Args:
        session: ESConv 세션 (dialog 포함)
        violation_type: V1, V2, V3 중 하나
        model: OpenAI model name
        temperature: LLM temperature
    
    Returns:
        Rewritten turn text
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    client = OpenAI(api_key=api_key)
    
    # Extract data
    situation = session.get('situation', '')
    dialog = session['dialog']
    
    # Build prompt
    if violation_type == "V1":
        system_prompt = "당신은 감정지원 상담 대화 데이터를 생성하는 전문가입니다."
        user_prompt = build_v1_prompt(situation, dialog)
    elif violation_type == "V2":
        system_prompt = "당신은 감정지원 상담 대화 데이터를 생성하는 전문가입니다."
        user_prompt = build_v2_prompt(situation, dialog)
    elif violation_type == "V3":
        system_prompt = "당신은 감정지원 상담 대화 데이터를 생성하는 전문가입니다."
        user_prompt = build_v3_prompt(situation, dialog)
    else:
        raise ValueError(f"Unknown violation type: {violation_type}")
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # Call LLM
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=300
    )
    
    rewritten_text = response.choices[0].message.content.strip()
    
    # V3 JSON 출력 문제 해결: JSON 파싱 시도
    if rewritten_text.startswith('{'):
        try:
            import json
            data = json.loads(rewritten_text)
            # 마지막 dialog entry의 content 추출
            if 'dialog' in data and len(data['dialog']) > 0:
                rewritten_text = data['dialog'][-1].get('content', rewritten_text)
        except:
            pass  # JSON 파싱 실패시 원본 사용
    
    return rewritten_text


def generate_v1_v3_session(
    esconv_session: Dict,
    violation_type: str,
    length_range: Tuple[int, int],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> Dict:
    """
    V1-V3 세션 생성 (마지막 turn만 rewrite)
    
    Args:
        esconv_session: 원본 ESConv 세션
        violation_type: V1, V2, V3 중 하나
        length_range: (min, max) turns
        model: OpenAI model name
        temperature: LLM temperature
    
    Returns:
        Generated session with rewritten last turn
    """
    # 1. Cut to length
    session = cut_session_to_length(esconv_session, length_range[0], length_range[1])
    
    # 2. Rewrite last turn
    rewritten_text = rewrite_last_turn(session, violation_type, model, temperature)
    
    # 3. Update session
    session['dialog'][-1]['text'] = rewritten_text
    session['primary_label'] = violation_type
    session['generation_method'] = 'esconv_last_turn_rewrite'
    
    # Store original ESConv metadata
    if 'emotion_type' in session:
        session['source_emotion'] = session['emotion_type']
    if 'problem_type' in session:
        session['source_problem'] = session['problem_type']
    
    return session


def prepare_v1_v3_sessions(
    esconv_sessions: List[Dict],
    violation_type: str,
    num_sessions: int,
    length_range: Tuple[int, int],
    seed: int = 42,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7
) -> List[Dict]:
    """
    V1-V3 세션 여러 개 생성
    
    Args:
        esconv_sessions: ESConv 세션 리스트
        violation_type: V1, V2, V3 중 하나
        num_sessions: 생성할 세션 개수
        length_range: (min, max) turns
        seed: Random seed
        model: OpenAI model name
        temperature: LLM temperature
    
    Returns:
        List of generated sessions
    """
    random.seed(seed)
    
    # Violation reason templates
    violation_reasons = {
        'V1': 'Missing context gathering and clarification despite ambiguous situation',
        'V2': 'Directive advice without respecting user agency or offering choices',
        'V3': 'Generic empathy without addressing specific context or advancing dialog'
    }
    
    # Random sample
    sampled = random.sample(esconv_sessions, num_sessions)
    
    # Generate
    generated = []
    for idx, esconv_session in enumerate(sampled):
        print(f"  Generating {violation_type}_{idx:04d}...")
        
        session = generate_v1_v3_session(
            esconv_session,
            violation_type,
            length_range,
            model,
            temperature
        )
        
        # Add violation metadata
        session['session_id'] = f"{violation_type.lower()}_{idx:04d}"
        session['violation_turn_index'] = len(session['dialog']) - 1
        session['violation_reason'] = violation_reasons.get(violation_type, f'{violation_type} violation in last turn')
        session['generation_method'] = 'esconv_last_turn_rewrite'
        
        generated.append(session)
    
    return generated


def validate_v1_v3_session(session: Dict, violation_type: str) -> bool:
    """V1-V3 세션 검증"""
    if 'dialog' not in session or len(session['dialog']) == 0:
        return False
    
    if session['dialog'][-1]['speaker'] != 'supporter':
        return False
    
    if session.get('primary_label') != violation_type:
        return False
    
    return True
