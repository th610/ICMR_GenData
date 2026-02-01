"""
V1-V3 Insertion Generator (ESConv prefix + 4-6 turn insertion)
"""
import random
import json
from src.llm.openai_client import OpenAIClient
from src.llm.prompts import (
    V1_SYSTEM, V2_SYSTEM, V3_SYSTEM,
    build_v1_prompt, build_v2_prompt, build_v3_prompt,
    RETRY_MESSAGE
)


def extract_esconv_prefix(dialog, min_turns=12, max_turns=20):
    """
    ESConv 원본에서 prefix 추출
    - 마지막은 반드시 supporter로 끝나야 함 (위반 삽입 위치)
    """
    # 최대 가능 길이 (dialog 길이와 max_turns 중 작은 값)
    max_possible = min(max_turns, len(dialog))
    
    # min_turns보다 짧으면 사용 불가
    if max_possible < min_turns:
        return None, None
    
    # 가능한 모든 길이 생성
    possible_lengths = list(range(min_turns, max_possible + 1))
    
    # supporter로 끝나는 길이만 필터링
    valid_lengths = []
    for length in possible_lengths:
        if dialog[length - 1]['speaker'] == 'supporter':
            valid_lengths.append(length)
    
    if not valid_lengths:
        return None, None
    
    prefix_length = random.choice(valid_lengths)
    prefix_dialog = dialog[:prefix_length]
    
    return prefix_dialog, prefix_length


def parse_insertion_result(result_text):
    """JSON 파싱 (LLM 출력)"""
    try:
        # 코드 블록 제거
        if '```json' in result_text:
            result_text = result_text.split('```json')[1].split('```')[0]
        elif '```' in result_text:
            result_text = result_text.split('```')[1].split('```')[0]
        
        data = json.loads(result_text.strip())
        return data
    except Exception as e:
        raise ValueError(f"Failed to parse JSON: {e}\n{result_text[:500]}")


def generate_v1_v3_with_insertion(
    esconv_sessions,
    violation_type,
    num_sessions,
    seed,
    model,
    temperature
):
    """
    V1/V2/V3 생성 (Insertion 방식)
    
    Args:
        esconv_sessions: ESConv 원본 데이터
        violation_type: 'V1', 'V2', 'V3'
        num_sessions: 생성할 세션 수
        seed: 랜덤 시드
        model: LLM 모델
        temperature: LLM temperature
    
    Returns:
        list of generated sessions
    """
    random.seed(seed)
    
    # 시스템 프롬프트 선택
    system_prompts = {
        'V1': V1_SYSTEM,
        'V2': V2_SYSTEM,
        'V3': V3_SYSTEM
    }
    system_prompt = system_prompts[violation_type]
    
    # 프롬프트 빌더 선택
    prompt_builders = {
        'V1': build_v1_prompt,
        'V2': build_v2_prompt,
        'V3': build_v3_prompt
    }
    build_prompt = prompt_builders[violation_type]
    
    # ESConv에서 사용 가능한 세션만 필터링 (길이 12 이상)
    valid_sessions = [s for s in esconv_sessions if len(s['dialog']) >= 12]
    
    # 샘플링 (서로 다른 세션)
    if len(valid_sessions) < num_sessions:
        raise ValueError(f"Not enough valid sessions: {len(valid_sessions)} < {num_sessions}")
    
    sampled_sessions = random.sample(valid_sessions, num_sessions)
    
    # OpenAI Client 초기화
    client = OpenAIClient(
        model=model,
        temperature=temperature,
        max_tokens=2000  # insertion은 더 긴 응답 필요
    )
    
    generated = []
    
    for idx, session in enumerate(sampled_sessions):
        print(f"\n  Generating {violation_type} session {idx+1}/{num_sessions}...")
        
        # Prefix 추출
        prefix_dialog, prefix_length = extract_esconv_prefix(session['dialog'])
        
        if prefix_dialog is None:
            print(f"    ❌ Skipped: Cannot extract valid prefix")
            continue
        
        situation = session['situation']
        print(f"    Prefix: {prefix_length} turns (last speaker: {prefix_dialog[-1]['speaker']})")
        
        # Prompt 생성
        user_prompt = build_prompt(situation, prefix_dialog, prefix_length - 1)
        
        # LLM 호출
        try:
            data = client.call(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                retry_message=RETRY_MESSAGE
            )
        except Exception as e:
            print(f"    ❌ API error: {e}")
            continue
        
        # 검증
        if 'dialog' not in data or 'primary_label' not in data:
            print(f"    ❌ Invalid output: Missing required fields")
            continue
        
        if data['primary_label'] != violation_type:
            print(f"    ⚠️ Label mismatch: {data['primary_label']} != {violation_type}")
        
        # **중요**: LLM이 prefix를 포함하지 않았을 수 있으므로, 수동으로 병합
        insertion_dialog = data['dialog']
        
        # 4턴 검증
        if len(insertion_dialog) != 4:
            print(f"    ❌ Invalid insertion length: {len(insertion_dialog)} turns (expected 4)")
            continue
        
        # Speaker 순서 검증 (prefix 마지막=supporter이므로 삽입은 seeker로 시작)
        expected_speakers = ['seeker', 'supporter', 'seeker', 'supporter']
        actual_speakers = [t['speaker'] for t in insertion_dialog]
        if actual_speakers != expected_speakers:
            print(f"    ❌ Invalid speaker sequence: {actual_speakers} (expected {expected_speakers})")
            continue
        
        # Prefix + Insertion 병합
        full_dialog = list(prefix_dialog) + insertion_dialog
        
        # 연속 speaker 최종 검증 (prefix 마지막 + insertion 첫 번째)
        if prefix_dialog[-1]['speaker'] == insertion_dialog[0]['speaker']:
            print(f"    ❌ Consecutive speakers at merge point: {prefix_dialog[-1]['speaker']}")
            continue
        
        data['dialog'] = full_dialog
        
        # violation_turn_index 재계산 (마지막 supporter)
        data['violation_turn_index'] = len(full_dialog) - 1
        
        # 추가 메타 정보
        data['session_id'] = f"{violation_type.lower()}_{idx:02d}"
        data['source_esconv_idx'] = esconv_sessions.index(session)
        data['prefix_length'] = prefix_length
        data['insertion_length'] = len(insertion_dialog)
        
        num_turns = len(full_dialog)
        violation_idx = data['violation_turn_index']
        
        print(f"    ✅ Generated: {num_turns} turns (prefix={prefix_length}, insertion={len(insertion_dialog)}), violation at turn {violation_idx}")
        
        generated.append(data)
    
    if len(generated) < num_sessions:
        print(f"\n  ⚠️ Warning: Generated {len(generated)}/{num_sessions} sessions")
    
    return generated


def validate_insertion_session(session, violation_type):
    """삽입 세션 검증"""
    try:
        # 필수 필드
        required = ['primary_label', 'situation', 'dialog', 'generation_method', 'violation_turn_index']
        if not all(k in session for k in required):
            return False
        
        # 라벨 일치
        if session['primary_label'] != violation_type:
            return False
        
        # generation_method
        if session['generation_method'] != 'esconv_prefix_with_insertion':
            return False
        
        # dialog 구조
        dialog = session['dialog']
        if len(dialog) < 15:  # prefix(12+) + insertion(4+) = 16+ 정도
            return False
        
        # 마지막은 supporter
        if dialog[-1]['speaker'] != 'supporter':
            return False
        
        # violation_turn_index는 마지막 supporter
        violation_idx = session['violation_turn_index']
        if violation_idx != len(dialog) - 1:
            return False
        
        return True
        
    except Exception:
        return False
