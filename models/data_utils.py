"""
Data Preprocessing & Dataset Module for Violation Classifier
=============================================================
대화 위반 탐지를 위한 데이터 전처리 및 데이터셋 클래스

핵심 기능:
1. 512 토큰 역방향 truncation (TARGET 기준)
2. 역할 토큰 포맷팅 (<SEEKER>, <SUPPORTER>, <SUPPORTER_TARGET>)
3. 클래스 가중치 계산 (역제곱근 방식)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset


# ============================================================================
# Configuration
# ============================================================================

class DataConfig:
    """데이터 전처리 설정"""
    # Window
    WINDOW_SIZE = 8  # prefix에서 최근 몇 턴까지 포함할지
    
    # Class labels
    LABELS = ["Normal", "V1", "V2", "V3", "V4", "V5"]
    LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
    ID2LABEL = {idx: label for idx, label in enumerate(LABELS)}
    
    # Special tokens
    SEEKER_TOKEN = "<SEEKER>"
    SUPPORTER_TOKEN = "<SUPPORTER>"
    SUPPORTER_TARGET_TOKEN = "<SUPPORTER_TARGET>"


# ============================================================================
# Data Loading & Statistics
# ============================================================================

def load_json_data(filepath):
    """JSON 파일 로드"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def print_data_statistics(data, split_name):
    """데이터 통계 출력"""
    print(f"\n{'='*60}")
    print(f"[{split_name}] Data Statistics")
    print(f"{'='*60}")
    
    metadata = data.get('metadata', {})
    distribution = metadata.get('distribution', {})
    total = metadata.get('total_samples', len(data['samples']))
    
    print(f"Total samples: {total}")
    print(f"\nClass distribution:")
    for label in DataConfig.LABELS:
        count = distribution.get(label, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"  {label:8s}: {count:4d} ({pct:5.1f}%)")
    print(f"{'='*60}\n")


# ============================================================================
# Text Formatting with 512-Token Reverse Truncation
# ============================================================================

def format_conversation_with_truncation(prefix_dialog, generated_dialog, tokenizer, max_length=512):
    """
    대화를 포맷팅하고 512 토큰 역방향 truncation 적용
    
    우선순위: TARGET (Turn 4) > Trigger (Turn 3) > Turn 2 > Turn 1 > prefix (최근부터)
    - TARGET/Trigger는 부분 잘림 허용
    - 나머지 턴은 통째로 포함 or drop
    
    Args:
        prefix_dialog (list): 이전 대화 턴들
        generated_dialog (list): 4턴 삽입 대화 (Turn 1-4)
        tokenizer: HuggingFace tokenizer
        max_length (int): 최대 토큰 길이
    
    Returns:
        formatted_text (str): 최종 입력 텍스트
    """
    # Step 1: generated_dialog 4턴 구조 확인
    if len(generated_dialog) != 4:
        raise ValueError(f"generated_dialog must have exactly 4 turns, got {len(generated_dialog)}")
    
    turn1 = generated_dialog[0]  # seeker
    turn2 = generated_dialog[1]  # supporter
    turn3 = generated_dialog[2]  # seeker (Trigger)
    turn4 = generated_dialog[3]  # supporter (TARGET)
    
    # Step 2: 각 턴을 역할 토큰과 함께 포맷팅
    def format_turn(turn, is_target=False):
        speaker = turn['speaker'].lower()
        content = turn['content'].strip()
        
        if is_target:
            token = DataConfig.SUPPORTER_TARGET_TOKEN
        elif speaker == 'seeker':
            token = DataConfig.SEEKER_TOKEN
        elif speaker == 'supporter':
            token = DataConfig.SUPPORTER_TOKEN
        else:
            token = DataConfig.SUPPORTER_TOKEN
        
        return f"{token} {content}"
    
    target_text = format_turn(turn4, is_target=True)
    trigger_text = format_turn(turn3, is_target=False)
    turn2_text = format_turn(turn2, is_target=False)
    turn1_text = format_turn(turn1, is_target=False)
    
    # Step 3: prefix에서 최근 WINDOW_SIZE 턴만 가져오기
    prefix_turns = []
    if prefix_dialog:
        recent_prefix = prefix_dialog[-DataConfig.WINDOW_SIZE:] if len(prefix_dialog) > DataConfig.WINDOW_SIZE else prefix_dialog
        for turn in recent_prefix:
            prefix_turns.append(format_turn(turn, is_target=False))
    
    # Step 4: 역방향 truncation (뒤에서부터 채우기)
    # 우선 TARGET은 무조건 포함
    parts = [target_text]
    
    # Tokenize해서 길이 체크 (special tokens 고려)
    def get_token_count(text_parts):
        combined = "\n".join(text_parts)
        tokens = tokenizer.encode(combined, add_special_tokens=True)
        return len(tokens)
    
    current_length = get_token_count(parts)
    
    # Trigger 추가 시도
    temp_parts = [trigger_text] + parts
    if get_token_count(temp_parts) <= max_length:
        parts = temp_parts
        current_length = get_token_count(parts)
    else:
        # Trigger 부분 잘림 허용
        available_tokens = max_length - current_length - 10  # 여유분
        trigger_tokens = tokenizer.encode(trigger_text, add_special_tokens=False)
        if available_tokens > 0:
            truncated_trigger = tokenizer.decode(trigger_tokens[-available_tokens:], skip_special_tokens=False)
            parts = [truncated_trigger] + parts
        return "\n".join(parts)
    
    # Turn 2 추가 시도 (통째로만)
    temp_parts = [turn2_text] + parts
    if get_token_count(temp_parts) <= max_length:
        parts = temp_parts
        current_length = get_token_count(parts)
    else:
        # Turn 2는 부분 잘림 불가 → skip
        pass
    
    # Turn 1 추가 시도 (통째로만)
    temp_parts = [turn1_text] + parts
    if get_token_count(temp_parts) <= max_length:
        parts = temp_parts
        current_length = get_token_count(parts)
    else:
        # Turn 1은 부분 잘림 불가 → skip
        pass
    
    # Prefix 추가 (최근 턴부터, 통째로만)
    for prefix_turn in reversed(prefix_turns):
        temp_parts = [prefix_turn] + parts
        if get_token_count(temp_parts) <= max_length:
            parts = temp_parts
            current_length = get_token_count(parts)
        else:
            # 오래된 prefix는 drop
            break
    
    # Step 5: 최종 텍스트 생성
    final_text = "\n".join(parts)
    return final_text


# ============================================================================
# Dataset Class
# ============================================================================

class ViolationDataset(Dataset):
    """대화 위반 탐지 데이터셋"""
    
    def __init__(self, data, tokenizer, max_length=512, is_test=False):
        """
        Args:
            data (dict): JSON 데이터 (samples 키 포함)
            tokenizer: HuggingFace tokenizer
            max_length (int): 최대 시퀀스 길이
            is_test (bool): 테스트셋 여부
        """
        self.samples = data['samples']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 1. 텍스트 포맷팅 (512 토큰 역방향 truncation)
        prefix_dialog = sample.get('prefix_dialog', [])
        generated_dialog = sample['generated_dialog']
        
        text = format_conversation_with_truncation(
            prefix_dialog, 
            generated_dialog, 
            self.tokenizer, 
            self.max_length
        )
        
        # 2. 토크나이징
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 3. 레이블
        label = DataConfig.LABEL2ID[sample['primary_label']]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': sample.get('esconv_session_id', idx)
        }


# ============================================================================
# Class Weight Calculation
# ============================================================================

def calculate_class_weights(data, method="sqrt"):
    """
    클래스 가중치 계산 (클래스 불균형 대응)
    
    Args:
        data (dict): JSON 데이터 (metadata 포함)
        method (str): "sqrt" (역제곱근, 권장) or "inverse" (역비례)
    
    Returns:
        torch.Tensor: 클래스 가중치 (shape: [num_classes])
    
    역제곱근 방식:
        weight_i = sqrt(N_total / N_i)
        - V5처럼 적은 클래스에 완만한 boosting
        - 과적합 방지
    
    역비례 방식:
        weight_i = N_total / N_i
        - 더 강한 boosting (과적합 위험 높음)
    """
    distribution = data['metadata']['distribution']
    total = data['metadata']['total_samples']
    
    weights = []
    print(f"\n{'='*60}")
    print(f"Class Weights Calculation (method={method})")
    print(f"{'='*60}")
    
    for label in DataConfig.LABELS:
        count = distribution[label]
        if method == "sqrt":
            # 역제곱근: sqrt(N_total / N_i)
            weight = np.sqrt(total / count)
        elif method == "inverse":
            # 역비례: N_total / N_i
            weight = total / count
        else:
            weight = 1.0
        
        weights.append(weight)
        print(f"  {label:8s}: count={count:4d}, weight={weight:.2f}")
    
    # 정규화 (평균이 1.0이 되도록)
    weights = np.array(weights)
    weights = weights / weights.mean()
    
    print(f"\nNormalized weights (mean=1.0):")
    for label, weight in zip(DataConfig.LABELS, weights):
        print(f"  {label:8s}: {weight:.2f}")
    print(f"{'='*60}\n")
    
    return torch.FloatTensor(weights)


# ============================================================================
# Tokenizer Setup
# ============================================================================

def setup_tokenizer(tokenizer):
    """
    토크나이저에 특수 토큰 추가
    
    Args:
        tokenizer: HuggingFace tokenizer
    
    Returns:
        tokenizer: 특수 토큰이 추가된 tokenizer
        num_added: 추가된 토큰 수
    """
    special_tokens = {
        'additional_special_tokens': [
            DataConfig.SEEKER_TOKEN,
            DataConfig.SUPPORTER_TOKEN,
            DataConfig.SUPPORTER_TARGET_TOKEN
        ]
    }
    num_added = tokenizer.add_special_tokens(special_tokens)
    return tokenizer, num_added
