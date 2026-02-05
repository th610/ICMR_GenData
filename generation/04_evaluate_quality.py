"""
Step 4: Evaluate Generated Samples
===================================
GPT-4o-mini Judge를 사용하여 생성된 샘플의 품질을 평가합니다.
- 실제 레이블 vs 예측 레이블 비교
- 정확도 계산
- 잘못 분류된 샘플 필터링

Output: generation/outputs/evaluated/evaluation_results.json
"""
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from src.llm.prompts_judge import JUDGE_SYSTEM, build_judge_prompt

client = OpenAI()

def parse_generated_turn(generated_text):
    """V5의 generated_turn 텍스트를 dialog 형식으로 파싱"""
    turns = []
    lines = generated_text.strip().split('\n')
    
    current_speaker = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if line.startswith("Turn") and ":" in line:
            if current_speaker and current_content:
                turns.append({
                    "speaker": current_speaker,
                    "content": " ".join(current_content).strip()
                })
            
            parts = line.split(":", 1)
            speaker_part = parts[0].lower()
            if "seeker" in speaker_part:
                current_speaker = "seeker"
            elif "supporter" in speaker_part:
                current_speaker = "supporter"
            
            if len(parts) > 1 and parts[1].strip():
                current_content = [parts[1].strip()]
            else:
                current_content = []
        else:
            if current_speaker:
                current_content.append(line)
    
    if current_speaker and current_content:
        turns.append({
            "speaker": current_speaker,
            "content": " ".join(current_content).strip()
        })
    
    return turns

def evaluate_sample(sample, ground_truth_label):
    """단일 샘플 평가"""
    try:
        # Prefix와 generated dialog 합치기
        prefix = sample.get("prefix_dialog", sample.get("prefix_conversation", []))
        generated_dialog = sample.get("generated_dialog", [])
        
        # V5는 generated_turn (텍스트)
        if not generated_dialog:
            generated_text = sample.get("generated_turn", "")
            if isinstance(generated_text, str):
                try:
                    generated_dialog = json.loads(generated_text)
                except:
                    generated_dialog = parse_generated_turn(generated_text)
        
        # Full conversation
        full_conversation = prefix + generated_dialog
        
        # Build judge prompt
        user_prompt = build_judge_prompt(
            situation=sample.get("situation", ""),
            conversation=full_conversation
        )
        
        # Call Judge
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            max_tokens=100
        )
        
        prediction = response.choices[0].message.content.strip()
        
        # Parse prediction
        if "V1" in prediction:
            predicted_label = "V1"
        elif "V2" in prediction:
            predicted_label = "V2"
        elif "V3" in prediction:
            predicted_label = "V3"
        elif "V4" in prediction:
            predicted_label = "V4"
        elif "V5" in prediction:
            predicted_label = "V5"
        elif "Normal" in prediction:
            predicted_label = "Normal"
        else:
            predicted_label = "Unknown"
        
        is_correct = (predicted_label == ground_truth_label)
        
        return {
            "predicted_label": predicted_label,
            "is_correct": is_correct,
            "judge_response": prediction
        }
    
    except Exception as e:
        print(f"  Error evaluating sample: {str(e)[:100]}")
        return {
            "predicted_label": "Error",
            "is_correct": False,
            "judge_response": str(e)
        }

def evaluate_file(label, input_file):
    """하나의 파일 평가"""
    print(f"\n{'='*70}")
    print(f"Evaluating {label}")
    print(f"{'='*70}")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    results = []
    correct = 0
    
    for i, sample in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Session {sample.get('esconv_session_id', 'unknown')}")
        
        eval_result = evaluate_sample(sample, label)
        
        sample_result = {
            **sample,
            "ground_truth_label": label,
            "predicted_label": eval_result["predicted_label"],
            "is_correct": eval_result["is_correct"],
            "judge_response": eval_result["judge_response"]
        }
        
        results.append(sample_result)
        
        if eval_result["is_correct"]:
            correct += 1
            print(f"  ✓ Correct: {eval_result['predicted_label']}")
        else:
            print(f"  ✗ Wrong: GT={label}, Pred={eval_result['predicted_label']}")
    
    accuracy = correct / len(samples) * 100 if samples else 0
    print(f"\nAccuracy: {correct}/{len(samples)} ({accuracy:.1f}%)")
    
    return results, correct, len(samples)

def main():
    labels = ["V1", "V2", "V3", "V4", "V5", "Normal"]
    
    all_results = []
    total_correct = 0
    total_samples = 0
    
    for label in labels:
        input_file = Path(f"generation/outputs/generated/generated_{label}.json")
        if not input_file.exists():
            print(f"Warning: {input_file} not found, skipping...")
            continue
        
        results, correct, count = evaluate_file(label, input_file)
        all_results.extend(results)
        total_correct += correct
        total_samples += count
    
    # Save all results
    output_file = Path("generation/outputs/evaluated/evaluation_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Samples: {total_samples}")
    print(f"Correct: {total_correct}")
    print(f"Accuracy: {total_correct/total_samples*100:.1f}%")
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()
