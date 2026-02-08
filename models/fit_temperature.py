"""
Temperature Scaling for Calibration

Purpose: Valid/Test 데이터로 최적 Temperature를 찾아 모델 캘리브레이션

Process:
    1. Gold 데이터 로드 (test_gold_300.json)
    2. 각 샘플에 대해 logits 수집
    3. Temperature T를 최적화 (NLL minimize)
    4. 최적 T를 저장
"""
import json
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.optim import LBFGS
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.step1_context_builder import ContextBuilder
from src.agent.step3_violation_detector import ViolationDetector


def load_gold_data(data_path: str):
    """Gold 데이터 로드"""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['samples']


def prepare_sample(sample):
    """샘플에서 dialog history와 gold response 추출"""
    # prefix_dialog: 7턴 (Turn 1~3)
    # generated_dialog: 4턴 (Turn 4 전체 대화)
    # generated_dialog의 마지막 supporter turn이 gold response
    
    history = []
    for turn in sample['prefix_dialog']:
        history.append({
            'speaker': turn['speaker'],
            'text': turn['content']
        })
    
    # generated_dialog에서 turn 4까지 추가
    for i, turn in enumerate(sample['generated_dialog']):
        if i < 3:  # Turn 4의 seeker, supporter context, seeker target
            history.append({
                'speaker': turn['speaker'],
                'text': turn['content']
            })
        else:  # 마지막이 gold response
            gold_response = turn['content']
            break
    
    # Label 매핑: "Normal" -> "normal"
    label = sample['primary_label'].lower()
    
    return {
        'session_id': sample['esconv_session_id'],
        'situation': sample['situation'],
        'history': history,
        'gold_response': gold_response,
        'gold_label': label
    }


def collect_logits(detector, samples, context_builder):
    """
    각 샘플에 대한 logits 수집
    
    Returns:
        logits_list: List[torch.Tensor] (num_samples, num_classes)
        labels_list: List[int] (ground truth label ids)
    """
    logits_list = []
    labels_list = []
    
    label2id = {"normal": 0, "v1": 1, "v2": 2, "v3": 3, "v4": 4, "v5": 5}
    
    print(f"\n{'='*80}")
    print(f"Collecting logits from {len(samples)} samples...")
    print(f"{'='*80}\n")
    
    for i, sample_data in enumerate(samples):
        if i % 50 == 0:
            print(f"Processing {i}/{len(samples)}...")
        
        # Context 구성
        context = context_builder.build_context(
            history=sample_data['history'],
            situation=sample_data['situation']
        )
        
        # Candidate 생성 (gold response 사용)
        candidate = {
            'id': 0,
            'text': sample_data['gold_response']
        }
        
        # Logits 추출 (temperature 적용 전)
        # detector.detect()는 이미 temperature가 적용된 결과를 반환하므로
        # 직접 모델 호출해야 함
        with torch.no_grad():
            # Tokenization
            response_text = f"\n[RESPONSE]\n{candidate['text']}"
            response_tokens = detector.tokenizer.encode(response_text, add_special_tokens=False)
            response_length = len(response_tokens)
            remaining_tokens = 512 - response_length - 3
            
            # Context truncation
            context_parts = []
            if context.get("state_summary"):
                context_parts.append(f"[SUMMARY]\n{context['state_summary']}")
            context_parts.append("\n[CONTEXT]")
            
            turns = list(reversed(context["recent_turns"]))
            selected_turns = []
            current_tokens = 0
            
            for turn in turns:
                turn_text = f"\n{turn['speaker']}: {turn['text']}"
                turn_tokens = len(detector.tokenizer.encode(turn_text, add_special_tokens=False))
                
                if current_tokens + turn_tokens <= remaining_tokens:
                    selected_turns.insert(0, turn_text)
                    current_tokens += turn_tokens
                else:
                    break
            
            context_text = "".join(context_parts) + "".join(selected_turns)
            input_text = context_text + response_text
            
            # Tokenize
            inputs = detector.tokenizer(
                input_text,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            
            # Get logits (temperature 적용 전)
            outputs = detector.model(**inputs)
            logits = outputs.logits[0]  # (num_classes,)
        
        # Ground truth label
        gold_label_id = label2id[sample_data['gold_label']]
        
        logits_list.append(logits)
        labels_list.append(gold_label_id)
    
    print(f"Collected logits for {len(logits_list)} samples\n")
    
    return torch.stack(logits_list), torch.tensor(labels_list, dtype=torch.long)


def find_best_temperature(logits, labels, init_temp=1.5, max_iter=50):
    """
    Temperature T를 최적화 (NLL minimize)
    
    Args:
        logits: (num_samples, num_classes)
        labels: (num_samples,)
        init_temp: 초기 temperature
        max_iter: 최대 반복 횟수
    
    Returns:
        optimal_temperature: float
    """
    print(f"{'='*80}")
    print(f"Optimizing Temperature (Initial T={init_temp})")
    print(f"{'='*80}\n")
    
    # Temperature parameter (학습 가능)
    temperature = torch.nn.Parameter(torch.ones(1) * init_temp)
    
    # NLL loss
    nll_criterion = torch.nn.CrossEntropyLoss()
    
    # LBFGS optimizer (2nd order method)
    optimizer = LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        """Closure for LBFGS"""
        optimizer.zero_grad()
        
        # Temperature 적용
        scaled_logits = logits / temperature
        
        # NLL
        loss = nll_criterion(scaled_logits, labels)
        
        loss.backward()
        return loss
    
    # Optimize
    optimizer.step(eval_loss)
    
    optimal_temp = temperature.item()
    
    # Final NLL
    with torch.no_grad():
        scaled_logits = logits / optimal_temp
        final_nll = nll_criterion(scaled_logits, labels).item()
    
    print(f"Optimal Temperature: {optimal_temp:.4f}")
    print(f"Final NLL: {final_nll:.4f}\n")
    
    return optimal_temp


def evaluate_calibration(logits, labels, temperature):
    """
    Calibration 평가: ECE (Expected Calibration Error)
    
    Args:
        logits: (num_samples, num_classes)
        labels: (num_samples,)
        temperature: float
    
    Returns:
        ece_before: ECE before calibration
        ece_after: ECE after calibration
    """
    def compute_ece(probs, labels, n_bins=10):
        """ECE 계산"""
        confidences, predictions = torch.max(probs, dim=1)
        accuracies = predictions.eq(labels)
        
        ece = 0.0
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            lower = bin_boundaries[i]
            upper = bin_boundaries[i + 1]
            
            in_bin = (confidences > lower) & (confidences <= upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()
    
    # Before calibration (T=1.0)
    probs_before = F.softmax(logits, dim=-1)
    ece_before = compute_ece(probs_before, labels)
    
    # After calibration
    scaled_logits = logits / temperature
    probs_after = F.softmax(scaled_logits, dim=-1)
    ece_after = compute_ece(probs_after, labels)
    
    print(f"{'='*80}")
    print(f"Calibration Metrics")
    print(f"{'='*80}")
    print(f"ECE (Before): {ece_before:.4f}")
    print(f"ECE (After):  {ece_after:.4f}")
    print(f"Improvement:  {(ece_before - ece_after):.4f}\n")
    
    return ece_before, ece_after


def main():
    """Temperature Scaling 메인 실행"""
    print("\n" + "="*80)
    print("TEMPERATURE SCALING FOR CALIBRATION")
    print("="*80 + "\n")
    
    # 1. Load model
    model_path = Path(__file__).parent / "best_model"
    detector = ViolationDetector(
        mode="model",
        model_path=str(model_path),
        temperature=1.0  # 초기값 (fit 전)
    )
    
    # 2. Load gold data
    data_path = Path(__file__).parent / "data" / "test_gold_300.json"
    if not data_path.exists():
        data_path = Path(__file__).parent.parent / "data" / "final" / "test_gold_300.json"
    
    samples_raw = load_gold_data(str(data_path))
    samples = [prepare_sample(s) for s in samples_raw]
    
    print(f"Loaded {len(samples)} samples from {data_path.name}\n")
    
    # 3. Context Builder (summary 없이)
    from src.llm.openai_client import OpenAIClient
    client = OpenAIClient(model="gpt-4o-mini")
    context_builder = ContextBuilder(
        llm_client=client,
        window_size=6,
        use_summary=False
    )
    
    # 4. Collect logits
    logits, labels = collect_logits(detector, samples, context_builder)
    
    # 5. Find optimal temperature
    optimal_temp = find_best_temperature(logits, labels, init_temp=1.5, max_iter=50)
    
    # 6. Evaluate calibration
    ece_before, ece_after = evaluate_calibration(logits, labels, optimal_temp)
    
    # 7. Save result
    result = {
        "optimal_temperature": optimal_temp,
        "num_samples": len(samples),
        "ece_before": ece_before,
        "ece_after": ece_after,
        "improvement": ece_before - ece_after
    }
    
    output_path = Path(__file__).parent / "best_model" / "temperature.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    
    print(f"{'='*80}")
    print(f"Saved optimal temperature to: {output_path}")
    print(f"{'='*80}\n")
    
    print(f"Usage in code:")
    print(f"  detector = ViolationDetector(")
    print(f"      mode='model',")
    print(f"      model_path='models/best_model',")
    print(f"      temperature={optimal_temp:.4f}")
    print(f"  )\n")


if __name__ == "__main__":
    main()
