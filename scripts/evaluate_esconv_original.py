"""
Evaluate ESConv ORIGINAL responses with our Judge

This tests our judge on the actual ESConv data we didn't modify.
Should give us a baseline violation rate for "real counseling data".
"""
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt, RETRY_MESSAGE


def evaluate_esconv_original(max_samples: int = 100,
                             output_path: str = "data/external/esconv_original_judged.json"):
    """Evaluate original ESConv responses"""
    
    print("Loading ESConv original data...")
    with open("ESConv.json", 'r', encoding='utf-8') as f:
        esconv_data = json.load(f)
    
    # Sample sessions
    sessions = list(esconv_data.values())[:max_samples]
    
    print(f"Evaluating {len(sessions)} sessions...")
    
    llm = OpenAIClient()
    results = []
    violation_counts = {
        "normal": 0,
        "v1": 0, "v2": 0, "v3": 0, "v4": 0, "v5": 0
    }
    
    for session in tqdm(sessions, desc="Judging"):
        # Get dialog from session
        dialog = session.get('dialog', [])
        
        if len(dialog) < 3:
            continue
        
        # Build dialog text (up to last 6 turns)
        recent_dialog = dialog[-6:] if len(dialog) > 6 else dialog
        dialog_text = "\n".join([
            f"{turn['speaker']}: {turn['content']}"
            for turn in recent_dialog
        ])
        
        prompt = build_judge_prompt(dialog_text)
        
        try:
            result = llm.call(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=prompt,
                retry_message=RETRY_MESSAGE
            )
            
            label = result.get('label', 'Normal')
            if label and label != 'Normal':
                label = label.lower()
            else:
                label = 'normal'
            
            violation_counts[label] += 1
            
            results.append({
                "session_id": list(esconv_data.keys())[sessions.index(session)],
                "num_turns": len(dialog),
                "last_response": dialog[-1]['content'],
                "last_speaker": dialog[-1]['speaker'],
                "judge_label": label,
                "judge_reason": result.get('reason', '')
            })
            
        except Exception as e:
            print(f"\nError: {e}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "esconv_original",
            "total_samples": len(results),
            "violation_counts": violation_counts,
            "violation_rate": (len(results) - violation_counts['normal']) / len(results) * 100 if results else 0,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("ESConv Original Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"\nViolation Distribution:")
    for label, count in violation_counts.items():
        pct = count / len(results) * 100 if results else 0
        print(f"  {label.upper()}: {count} ({pct:.1f}%)")
    
    violation_rate = (len(results) - violation_counts['normal']) / len(results) * 100 if results else 0
    print(f"\nTotal Violations: {violation_rate:.1f}%")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    
    evaluate_esconv_original(args.max_samples)
