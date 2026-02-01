"""
Evaluate Empathetic Dialogues with our Judge

Multi-turn empathy conversations.
"""
import sys
import os
import json
from pathlib import Path
from tqdm import tqdm
import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt, RETRY_MESSAGE


def download_empathetic_data():
    """Download parquet data directly from HuggingFace"""
    url = "https://huggingface.co/datasets/facebook/empathetic_dialogues/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
    
    print("Downloading EmpatheticDialogues train data...")
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download: {response.status_code}")
    
    # Save to temp file
    temp_path = "data/external/empathetic_train.parquet"
    os.makedirs(os.path.dirname(temp_path), exist_ok=True)
    
    with open(temp_path, 'wb') as f:
        f.write(response.content)
    
    return pd.read_parquet(temp_path)


def evaluate_empathetic(max_samples: int = 100,
                        output_path: str = "data/external/empathetic_judged.json"):
    """Evaluate EmpatheticDialogues dataset"""
    
    # Download data
    df = download_empathetic_data()
    
    # Group by conversation
    print("Grouping conversations...")
    conversations = {}
    for _, row in df.iterrows():
        conv_id = row['conv_id']
        if conv_id not in conversations:
            conversations[conv_id] = {
                'conv_id': conv_id,
                'context': row['context'],
                'utterances': []
            }
        conversations[conv_id]['utterances'].append({
            'utterance_idx': row['utterance_idx'],
            'speaker_idx': row['speaker_idx'],
            'utterance': row['utterance']
        })
    
    # Sample conversations
    samples = []
    for conv_id, conv_data in list(conversations.items())[:max_samples]:
        # Sort utterances by index
        utterances = sorted(conv_data['utterances'], key=lambda x: x['utterance_idx'])
        
        if len(utterances) < 3:
            continue
        
        # Build dialog
        dialog = []
        for utt in utterances:
            speaker = "seeker" if utt['speaker_idx'] == 0 else "supporter"
            dialog.append({
                "speaker": speaker,
                "text": utt['utterance']
            })
        
        samples.append({
            "conv_id": conv_id,
            "context": conv_data['context'],
            "num_turns": len(dialog),
            "dialog": dialog
        })
    
    print(f"Evaluating {len(samples)} conversations...")
    
    llm = OpenAIClient()
    results = []
    violation_counts = {
        "normal": 0,
        "v1": 0, "v2": 0, "v3": 0, "v4": 0, "v5": 0
    }
    
    for sample in tqdm(samples, desc="Judging"):
        # Build dialog text
        dialog_text = "\n".join([
            f"{turn['speaker'].capitalize()}: {turn['text']}"
            for turn in sample['dialog']
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
                "conv_id": sample['conv_id'],
                "context": sample['context'],
                "num_turns": sample['num_turns'],
                "last_response": sample['dialog'][-1]['text'],
                "judge_label": label,
                "judge_reason": result.get('reason', '')
            })
            
        except Exception as e:
            print(f"\nError on {sample['conv_id']}: {e}")
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "dataset": "empathetic_dialogues",
            "total_samples": len(results),
            "violation_counts": violation_counts,
            "violation_rate": (len(results) - violation_counts['normal']) / len(results) * 100,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("EmpatheticDialogues Evaluation Complete")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"\nViolation Distribution:")
    for label, count in violation_counts.items():
        pct = count / len(results) * 100
        print(f"  {label.upper()}: {count} ({pct:.1f}%)")
    
    violation_rate = (len(results) - violation_counts['normal']) / len(results) * 100
    print(f"\nTotal Violations: {violation_rate:.1f}%")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()
    
    evaluate_empathetic(args.max_samples)
