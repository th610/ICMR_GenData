"""
Normal 샘플에 "위기 상황 + 안전 응답" 추가

목적: V5 모델이 "응답 품질"을 평가하도록 학습 데이터 보강

생성 전략:
1. V5 샘플의 위기 상황 prefix 재사용
2. 안전 응답 생성 (LLM 또는 템플릿)
3. Normal 라벨 부여
"""
import json
import random
from pathlib import Path
from dotenv import load_dotenv
import sys

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient


def load_v5_contexts():
    """V5 샘플에서 위기 상황 prefix 추출"""
    with open('data/final/train_1000.json', 'r') as f:
        data = json.load(f)
    
    v5_samples = [s for s in data['samples'] if s.get('primary_label') == 'V5']
    
    contexts = []
    for sample in v5_samples:
        contexts.append({
            'session_id': sample['esconv_session_id'],
            'situation': sample['situation'],
            'prefix_dialog': sample['prefix_dialog']
        })
    
    return contexts


def generate_safety_response(context, llm):
    """위기 상황에 대한 적절한 안전 응답 생성"""
    
    # Prefix를 텍스트로 변환
    dialog_text = ""
    for turn in context['prefix_dialog']:
        speaker = turn['speaker'].upper()
        content = turn['content']
        dialog_text += f"{speaker}: {content}\n"
    
    prompt = f"""You are a crisis counselor. The user has mentioned suicidal thoughts or self-harm.

Conversation so far:
{dialog_text}

Generate a SAFE and APPROPRIATE supporter response that:
1. Directly asks if they are in immediate danger ("Are you safe right now?")
2. Provides crisis resources (hotline numbers: 1393, 1577-0199, or 112/119)
3. Emphasizes they don't have to go through this alone
4. Keep it brief, warm, and urgent

Output ONLY the supporter's response text (no labels, no JSON).
"""
    
    messages = [
        {"role": "system", "content": "You are an expert crisis counselor."},
        {"role": "user", "content": prompt}
    ]
    
    response = llm.client.chat.completions.create(
        model=llm.model,
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )
    
    return response.choices[0].message.content.strip()


def create_normal_crisis_samples(num_samples=50):
    """Normal 라벨의 위기 대응 샘플 생성"""
    
    print(f"Generating {num_samples} Normal samples with crisis context...")
    
    # Load V5 contexts
    v5_contexts = load_v5_contexts()
    print(f"Found {len(v5_contexts)} V5 contexts to reuse")
    
    # Initialize LLM
    llm = OpenAIClient(model="gpt-4o-mini")
    
    # Sample contexts
    selected_contexts = random.sample(v5_contexts, min(num_samples, len(v5_contexts)))
    
    normal_crisis_samples = []
    
    for i, context in enumerate(selected_contexts, 1):
        print(f"Generating {i}/{len(selected_contexts)}...")
        
        # Generate safety response
        safety_response = generate_safety_response(context, llm)
        
        # Create sample
        sample = {
            'esconv_session_id': f"{context['session_id']}_normal_crisis",
            'situation': context['situation'],
            'prefix_dialog': context['prefix_dialog'],
            'generated_dialog': [
                {
                    'speaker': 'supporter',
                    'content': safety_response
                }
            ],
            'primary_label': 'Normal',  # ← 안전 응답은 Normal!
            'generation_method': 'crisis_safety_response',
            'violation_turn_index': None
        }
        
        normal_crisis_samples.append(sample)
        
        # Print example
        if i <= 3:
            print(f"\n[Example {i}]")
            print(f"Context (last turn): {context['prefix_dialog'][-1]['content'][:80]}...")
            print(f"Safety Response: {safety_response[:150]}...")
            print()
    
    return normal_crisis_samples


def integrate_into_training_data(normal_crisis_samples):
    """기존 train_1000.json에 통합"""
    
    # Load existing data
    with open('data/final/train_1000.json', 'r') as f:
        data = json.load(f)
    
    # Add new samples
    original_count = len(data['samples'])
    data['samples'].extend(normal_crisis_samples)
    
    # Update metadata
    data['metadata']['total_samples'] = len(data['samples'])
    
    # Update distribution
    from collections import Counter
    labels = [s['primary_label'] for s in data['samples']]
    data['metadata']['distribution'] = dict(Counter(labels))
    
    # Save to new file
    output_path = 'data/final/train_augmented.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"AUGMENTED DATA SAVED")
    print(f"{'='*80}")
    print(f"Original samples: {original_count}")
    print(f"Added Normal+Crisis: {len(normal_crisis_samples)}")
    print(f"Total samples: {len(data['samples'])}")
    print(f"\nDistribution:")
    for label, count in sorted(data['metadata']['distribution'].items()):
        print(f"  {label:8}: {count:4}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*80}")


if __name__ == "__main__":
    # Generate samples
    normal_crisis_samples = create_normal_crisis_samples(num_samples=50)
    
    # Integrate into training data
    integrate_into_training_data(normal_crisis_samples)
    
    print("\n✅ Done! Now you can retrain the model with:")
    print("   python models/train.py --data data/final/train_augmented.json")
