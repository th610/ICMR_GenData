"""
Step 6: Analyze Dataset Statistics
===================================
Train/Valid/Test 데이터셋의 통계를 분석합니다:
- 클래스별 분포
- 텍스트 길이 (단어 수)
- 대화 턴 수
- 토큰 수 추정

Output: Console output with statistics
"""
import json
from pathlib import Path
from collections import Counter

def count_words(text):
    """Count words in text"""
    return len(text.split())

def analyze_dataset(filepath, name):
    """Analyze a single dataset file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*70}")
    print(f"{name.upper()} Dataset")
    print(f"{'='*70}")
    print(f"Total samples: {len(data)}")
    
    # Class distribution
    labels = [sample.get("primary_label", sample.get("ground_truth_label", "Unknown")) for sample in data]
    label_counts = Counter(labels)
    
    print("\nClass Distribution:")
    for label in ["Normal", "V1", "V2", "V3", "V4", "V5"]:
        count = label_counts.get(label, 0)
        percentage = count / len(data) * 100 if data else 0
        print(f"  {label:8s}: {count:4d} ({percentage:5.1f}%)")
    
    # Check for class imbalance
    if data:
        avg_per_class = len(data) / len(label_counts)
        imbalanced = [label for label, count in label_counts.items() 
                     if count < avg_per_class * 0.5]
        if imbalanced:
            print(f"\n⚠️  Underrepresented classes: {', '.join(imbalanced)}")
    
    # Text statistics
    word_counts = []
    turn_counts = []
    
    for sample in data:
        # Combine prefix and generated
        prefix = sample.get("prefix_dialog", sample.get("prefix_conversation", []))
        generated = sample.get("generated_dialog", [])
        
        # Handle V5 format
        if not generated:
            generated_text = sample.get("generated_turn", "")
            if isinstance(generated_text, str) and generated_text:
                # Estimate turns from text
                turn_counts.append(len(prefix) + 4)  # V5 typically adds 4 turns
                word_counts.append(sum(count_words(t.get("content", "")) for t in prefix) + count_words(generated_text))
                continue
        
        full_conversation = prefix + generated
        
        # Count turns
        turn_counts.append(len(full_conversation))
        
        # Count words
        total_words = sum(count_words(turn.get("content", "")) for turn in full_conversation)
        word_counts.append(total_words)
    
    if word_counts:
        print(f"\nText Statistics:")
        print(f"  Avg words per sample: {sum(word_counts)/len(word_counts):.1f}")
        print(f"  Min words: {min(word_counts)}")
        print(f"  Max words: {max(word_counts)}")
        print(f"  Avg tokens (estimate): {sum(word_counts)/len(word_counts)*1.3:.0f}")
        print(f"  Max tokens (estimate): {max(word_counts)*1.3:.0f}")
    
    if turn_counts:
        print(f"\nDialog Statistics:")
        print(f"  Avg turns per sample: {sum(turn_counts)/len(turn_counts):.1f}")
        print(f"  Min turns: {min(turn_counts)}")
        print(f"  Max turns: {max(turn_counts)}")
    
    return {
        "total": len(data),
        "labels": dict(label_counts),
        "avg_words": sum(word_counts)/len(word_counts) if word_counts else 0,
        "avg_turns": sum(turn_counts)/len(turn_counts) if turn_counts else 0
    }

def main():
    data_dir = Path("data/final")
    
    datasets = [
        ("train.json", "Train"),
        ("valid.json", "Valid"),
        ("test.json", "Test"),
    ]
    
    stats = {}
    for filename, name in datasets:
        filepath = data_dir / filename
        if filepath.exists():
            stats[name] = analyze_dataset(filepath, name)
        else:
            print(f"\n⚠️  {filepath} not found")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_samples = sum(s["total"] for s in stats.values())
    print(f"Total samples: {total_samples}")
    
    for name, s in stats.items():
        print(f"\n{name}:")
        print(f"  Samples: {s['total']}")
        print(f"  Avg words: {s['avg_words']:.1f}")
        print(f"  Avg turns: {s['avg_turns']:.1f}")

if __name__ == "__main__":
    main()
