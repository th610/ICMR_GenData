"""
Analyze counsel-chat dataset for violations

Purpose:
    - Download counsel-chat from Hugging Face
    - Evaluate responses with our Judge
    - Check violation rates in professional counselor responses
"""
import sys
import os
import json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm.judge import build_judge_prompt


def download_counsel_chat(output_path: str = "data/external/counsel_chat.json"):
    """Download counsel-chat dataset from Hugging Face"""
    print("Downloading counsel-chat dataset...")
    
    try:
        dataset = load_dataset("nbertagnolli/counsel-chat")
        
        # Convert to list
        data = []
        for item in dataset['train']:
            data.append({
                "question": item['questionTitle'],
                "question_text": item.get('questionText', ''),
                "answer": item['answerText'],
                "topics": item.get('topics', [])
            })
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Downloaded {len(data)} samples")
        print(f"Saved to: {output_path}")
        
        return data
    
    except Exception as e:
        print(f"Error downloading: {e}")
        print("Make sure you have 'datasets' package: pip install datasets")
        return None


def convert_to_dialog_format(sample: dict) -> dict:
    """Convert counsel-chat format to our dialog format"""
    # Build dialog with question as seeker, answer as supporter
    dialog = []
    
    # Question as seeker turn
    question_full = sample['question']
    if sample.get('question_text'):
        question_full += " " + sample['question_text']
    
    dialog.append({
        "speaker": "seeker",
        "text": question_full
    })
    
    # Answer as supporter turn
    dialog.append({
        "speaker": "supporter",
        "text": sample['answer']
    })
    
    return {
        "conversation_id": f"counsel_{hash(sample['question']) % 100000}",
        "dialog": dialog,
        "topics": sample.get('topics', []),
        "original_question": sample['question']
    }


def judge_counsel_chat(data_path: str = "data/external/counsel_chat.json",
                       output_path: str = "data/external/counsel_chat_judged.json",
                       max_samples: int = None):
    """Evaluate counsel-chat with our Judge"""
    
    # Load data
    if not os.path.exists(data_path):
        print(f"Data not found: {data_path}")
        print("Downloading first...")
        data = download_counsel_chat(data_path)
        if not data:
            return
    else:
        with open(data_path, encoding='utf-8') as f:
            data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
    
    print(f"\nEvaluating {len(data)} samples with Judge...")
    
    # Initialize Judge
    llm = OpenAIClient()
    
    results = []
    violation_counts = {
        "normal": 0,
        "v1": 0,
        "v2": 0,
        "v3": 0,
        "v4": 0,
        "v5": 0
    }
    
    for i, sample in enumerate(tqdm(data, desc="Judging")):
        # Convert to dialog format
        session = convert_to_dialog_format(sample)
        
        # Build judge prompt - format full dialog text
        dialog_text = "\n".join([
            f"{turn['speaker'].capitalize()}: {turn['text']}"
            for turn in session['dialog']
        ])
        prompt = build_judge_prompt(dialog_text)
        
        # Get judgment
        try:
            # Use Judge system prompts
            from src.llm.prompts import JUDGE_SYSTEM, RETRY_MESSAGE
            result = llm.call(
                system_prompt=JUDGE_SYSTEM,
                user_prompt=prompt,
                retry_message=RETRY_MESSAGE
            )
            
            # Extract label from JSON result
            label = result.get('label', 'Normal')
            if label and label != 'Normal':
                label = label.lower()
            else:
                label = 'normal'
            
            response = result.get('reason', '')
            violation_counts[label] += 1
            
            # Save result
            results.append({
                "sample_id": i,
                "question": sample['question'],
                "answer": sample['answer'],
                "topics": sample.get('topics', []),
                "judge_label": label,
                "judge_reasoning": response
            })
            
            # Progress report every 50
            if (i + 1) % 50 == 0:
                total = i + 1
                violation_rate = (total - violation_counts['normal']) / total * 100
                print(f"\nProgress: {total}/{len(data)}")
                print(f"Violation rate so far: {violation_rate:.1f}%")
                print(f"Distribution: {violation_counts}")
        
        except Exception as e:
            print(f"\nError on sample {i}: {e}")
            results.append({
                "sample_id": i,
                "question": sample['question'],
                "answer": sample['answer'],
                "judge_label": "error",
                "error": str(e)
            })
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_samples": len(data),
            "violation_counts": violation_counts,
            "violation_rate": (len(data) - violation_counts['normal']) / len(data) * 100,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("Counsel-Chat Violation Analysis Complete")
    print(f"{'='*60}")
    print(f"Total samples: {len(data)}")
    print(f"\nViolation Distribution:")
    for label, count in violation_counts.items():
        pct = count / len(data) * 100
        print(f"  {label.upper()}: {count} ({pct:.1f}%)")
    
    violation_total = len(data) - violation_counts['normal']
    violation_rate = violation_total / len(data) * 100
    print(f"\nTotal violations: {violation_total} ({violation_rate:.1f}%)")
    print(f"Safe (Normal): {violation_counts['normal']} ({100-violation_rate:.1f}%)")
    print(f"\nResults saved to: {output_path}")
    
    return results


def parse_judge_output(response: str) -> str:
    """Parse judge output to extract label"""
    response_lower = response.lower()
    
    # Check in severity order
    if "v5" in response_lower or ("safety" in response_lower and "single" in response_lower):
        return "v5"
    if "v4" in response_lower or ("safety" in response_lower and "mixed" in response_lower):
        return "v4"
    if "v3" in response_lower or "advice violation" in response_lower:
        return "v3"
    if "v2" in response_lower or "fact" in response_lower:
        return "v2"
    if "v1" in response_lower or "empathy" in response_lower:
        return "v1"
    
    return "normal"


def analyze_violations_by_topic(results_path: str = "data/external/counsel_chat_judged.json"):
    """Analyze which topics have more violations"""
    with open(results_path, encoding='utf-8') as f:
        data = json.load(f)
    
    results = data['results']
    
    # Topic-wise violation counts
    topic_stats = {}
    
    for r in results:
        topics = r.get('topics', ['unknown'])
        label = r['judge_label']
        
        for topic in topics:
            if topic not in topic_stats:
                topic_stats[topic] = {
                    "total": 0,
                    "normal": 0,
                    "violations": 0,
                    "v1": 0, "v2": 0, "v3": 0, "v4": 0, "v5": 0
                }
            
            topic_stats[topic]["total"] += 1
            if label == "normal":
                topic_stats[topic]["normal"] += 1
            else:
                topic_stats[topic]["violations"] += 1
                topic_stats[topic][label] += 1
    
    # Print analysis
    print(f"\n{'='*60}")
    print("Violations by Topic")
    print(f"{'='*60}\n")
    
    # Sort by violation rate
    sorted_topics = sorted(
        topic_stats.items(),
        key=lambda x: x[1]["violations"] / x[1]["total"] if x[1]["total"] > 0 else 0,
        reverse=True
    )
    
    for topic, stats in sorted_topics:
        if stats["total"] < 5:  # Skip rare topics
            continue
        
        violation_rate = stats["violations"] / stats["total"] * 100
        print(f"{topic} (n={stats['total']})")
        print(f"  Violation rate: {violation_rate:.1f}%")
        if stats["violations"] > 0:
            print(f"  Breakdown: V1={stats['v1']}, V2={stats['v2']}, V3={stats['v3']}, V4={stats['v4']}, V5={stats['v5']}")
        print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze counsel-chat for violations")
    parser.add_argument("--download-only", action="store_true", help="Only download, don't judge")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples to judge")
    parser.add_argument("--analyze-topics", action="store_true", help="Analyze violations by topic")
    
    args = parser.parse_args()
    
    if args.download_only:
        download_counsel_chat()
    elif args.analyze_topics:
        analyze_violations_by_topic()
    else:
        judge_counsel_chat(max_samples=args.max_samples)
