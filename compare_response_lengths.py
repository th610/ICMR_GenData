"""
Compare response lengths between our data and counsel-chat
"""
import json
import numpy as np

# Our data (ESConv-based)
print("="*60)
print("OUR DATA (ESConv-based)")
print("="*60)

our_data_files = [
    ('Normal', 'data/generated/normal_400.json'),
    ('V1', 'data/generated/v1_240.json'),
    ('V2', 'data/generated/v2_160.json'),
    ('V3', 'data/generated/v3_200.json'),
]

for label, filepath in our_data_files:
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        
        # Get last supporter response lengths
        lengths = []
        for session in data:
            dialog = session.get('dialog', [])
            supporter_turns = [t.get('content', t.get('text', '')) for t in dialog if t.get('speaker') == 'supporter']
            if supporter_turns:
                lengths.append(len(supporter_turns[-1]))
        
        print(f"\n{label}:")
        print(f"  Samples: {len(lengths)}")
        print(f"  Avg length: {np.mean(lengths):.1f} chars")
        print(f"  Min: {np.min(lengths)}, Max: {np.max(lengths)}")
        print(f"  Median: {np.median(lengths):.1f}")
        
        # Show example
        if lengths:
            sample_idx = len(lengths) // 2
            sample = data[sample_idx]
            supporter_turns = [t.get('content', t.get('text', '')) for t in sample['dialog'] if t.get('speaker') == 'supporter']
            if supporter_turns:
                example = supporter_turns[-1]
                print(f"  Example ({len(example)} chars): {example[:100]}...")
    except Exception as e:
        print(f"{label}: Error - {e}")

# Counsel-chat
print("\n" + "="*60)
print("COUNSEL-CHAT (Professional Counselors)")
print("="*60)

try:
    with open('data/external/counsel_chat.json', encoding='utf-8') as f:
        counsel = json.load(f)
    
    lengths = [len(s['answer']) for s in counsel[:50]]  # First 50
    
    print(f"\nCounsel-chat:")
    print(f"  Samples: {len(lengths)}")
    print(f"  Avg length: {np.mean(lengths):.1f} chars")
    print(f"  Min: {np.min(lengths)}, Max: {np.max(lengths)}")
    print(f"  Median: {np.median(lengths):.1f}")
    
    # Show example
    example = counsel[0]['answer']
    print(f"  Example ({len(example)} chars): {example[:100]}...")
    
    # Distribution
    print(f"\n  Distribution:")
    print(f"    < 200 chars: {sum(1 for l in lengths if l < 200)}")
    print(f"    200-500: {sum(1 for l in lengths if 200 <= l < 500)}")
    print(f"    500-1000: {sum(1 for l in lengths if 500 <= l < 1000)}")
    print(f"    > 1000: {sum(1 for l in lengths if l >= 1000)}")
    
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("COMPARISON")
print("="*60)

# Calculate ratio
our_avg = []
for label, filepath in our_data_files:
    try:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        lengths = []
        for session in data:
            dialog = session.get('dialog', [])
            supporter_turns = [t.get('content', t.get('text', '')) for t in dialog if t.get('speaker') == 'supporter']
            if supporter_turns:
                lengths.append(len(supporter_turns[-1]))
        if lengths:
            our_avg.append(np.mean(lengths))
    except:
        pass

if our_avg:
    our_mean = np.mean(our_avg)
    counsel_mean = np.mean(lengths) if 'lengths' in locals() else 0
    
    print(f"\nOur data avg: {our_mean:.1f} chars")
    print(f"Counsel-chat avg: {counsel_mean:.1f} chars")
    print(f"Ratio: Counsel-chat is {counsel_mean/our_mean:.1f}x longer")
