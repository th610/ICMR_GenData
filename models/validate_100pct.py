"""
Valid 100% 검증 스크립트
- Data leakage 체크
- 라벨 노출 체크
- Confusion matrix 검증
"""
import json
import hashlib
from collections import defaultdict, Counter

print("="*80)
print("1. DATA LEAKAGE 체크 (Train/Valid 중복)")
print("="*80)

# Train/Valid 데이터 로드
with open('models/data/train_split.json', 'r') as f:
    train_data = json.load(f)
with open('models/data/valid_split.json', 'r') as f:
    valid_data = json.load(f)

train_samples = train_data['samples']
valid_samples = valid_data['samples']

print(f"\nTrain: {len(train_samples)} samples")
print(f"Valid: {len(valid_samples)} samples")

# 1-1. Session ID 중복 체크
print("\n[Check 1-1] Session ID 중복")
train_sessions = set(s['esconv_session_id'] for s in train_samples)
valid_sessions = set(s['esconv_session_id'] for s in valid_samples)
overlap_sessions = train_sessions & valid_sessions

if overlap_sessions:
    print(f"❌ CRITICAL: {len(overlap_sessions)} sessions overlap!")
    print(f"Overlapping sessions: {list(overlap_sessions)[:10]}")
else:
    print(f"✅ No session ID overlap")

# 1-2. Prefix 중복 체크 (prefix_dialog 해시)
print("\n[Check 1-2] Prefix 중복 (해시 기준)")

def hash_prefix(prefix_dialog):
    """prefix_dialog를 해시로 변환"""
    text = json.dumps(prefix_dialog, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()

train_prefix_hashes = defaultdict(list)
for idx, s in enumerate(train_samples):
    h = hash_prefix(s.get('prefix_dialog', []))
    train_prefix_hashes[h].append(idx)

valid_prefix_hashes = defaultdict(list)
for idx, s in enumerate(valid_samples):
    h = hash_prefix(s.get('prefix_dialog', []))
    valid_prefix_hashes[h].append(idx)

overlap_hashes = set(train_prefix_hashes.keys()) & set(valid_prefix_hashes.keys())

if overlap_hashes:
    print(f"❌ CRITICAL: {len(overlap_hashes)} prefix overlap!")
    print(f"Example overlapping prefix hash: {list(overlap_hashes)[0]}")
else:
    print(f"✅ No prefix overlap")

# 1-3. 전체 텍스트 중복 체크
print("\n[Check 1-3] 전체 샘플 중복 (generated_dialog 포함)")

def hash_sample(sample):
    """전체 샘플을 해시로 변환"""
    key_data = {
        'prefix': sample.get('prefix_dialog', []),
        'generated': sample.get('generated_dialog', [])
    }
    text = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(text.encode()).hexdigest()

train_full_hashes = set(hash_sample(s) for s in train_samples)
valid_full_hashes = set(hash_sample(s) for s in valid_samples)
full_overlap = train_full_hashes & valid_full_hashes

if full_overlap:
    print(f"❌ CRITICAL: {len(full_overlap)} full samples overlap!")
else:
    print(f"✅ No full sample overlap")

print("\n" + "="*80)
print("2. 라벨 노출 체크 (입력 텍스트에 라벨 정보)")
print("="*80)

# 샘플 5개 뽑아서 실제 입력 텍스트 확인
import sys
sys.path.insert(0, '/home/mindrium-admin3/ICMR_GenData')
from models.data_utils import format_conversation_with_truncation, DataConfig
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
tokenizer.add_tokens(['<SEEKER>', '<SUPPORTER>', '<SUPPORTER_TARGET>'])

print("\n[Check 2-1] 입력 텍스트에 라벨 문자열 포함 여부")

label_keywords = ['V1', 'V2', 'V3', 'V4', 'V5', 'Normal', 'violation', 'primary_label']
suspicious_samples = []

for idx, sample in enumerate(train_samples[:100]):  # 처음 100개만 체크
    text = format_conversation_with_truncation(
        sample.get('prefix_dialog', []),
        sample['generated_dialog'],
        tokenizer,
        512
    )
    
    for keyword in label_keywords:
        if keyword in text:
            suspicious_samples.append((idx, sample['primary_label'], keyword, text[:200]))
            break

if suspicious_samples:
    print(f"❌ CRITICAL: {len(suspicious_samples)} samples contain label keywords!")
    print("Examples:")
    for idx, label, keyword, text_snippet in suspicious_samples[:3]:
        print(f"  Sample {idx} (Label: {label})")
        print(f"    Keyword: '{keyword}'")
        print(f"    Text: {text_snippet}...")
else:
    print(f"✅ No label keywords in input text (checked 100 samples)")

# 2-2. Special token 체크
print("\n[Check 2-2] 스페셜 토큰 사용 패턴")
token_counts = Counter()
for sample in train_samples[:50]:
    text = format_conversation_with_truncation(
        sample.get('prefix_dialog', []),
        sample['generated_dialog'],
        tokenizer,
        512
    )
    if '<SEEKER>' in text:
        token_counts['<SEEKER>'] += 1
    if '<SUPPORTER>' in text:
        token_counts['<SUPPORTER>'] += 1
    if '<SUPPORTER_TARGET>' in text:
        token_counts['<SUPPORTER_TARGET>'] += 1

print(f"Token usage in 50 samples:")
for token, count in token_counts.items():
    print(f"  {token}: {count}/50 samples")

print("\n" + "="*80)
print("3. CONFUSION MATRIX & 샘플 프린트 (평가 코드 검증)")
print("="*80)

# 모델 로드 및 예측
import torch
from models.model import ViolationClassifier
from models.data_utils import ViolationDataset
from torch.utils.data import DataLoader

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# 모델 로드
model = ViolationClassifier(
    model_name='roberta-base',
    num_labels=6,
    pooling='cls',
    dropout=0.1
)
# ViolationClassifier 내부의 roberta model에 토큰 크기 조정
model.roberta.resize_token_embeddings(len(tokenizer))

checkpoint = torch.load('models/outputs/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Valid 데이터셋
valid_dataset = ViolationDataset(valid_data, tokenizer, 512, is_test=False)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# 예측
all_preds = []
all_labels = []
all_sample_ids = []

with torch.no_grad():
    for batch in valid_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']
        sample_ids = batch['sample_id']
        
        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1).cpu()
        
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_sample_ids.extend(sample_ids.tolist())

print(f"\n[Check 3-1] Confusion Matrix")
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_labels, all_preds)

labels = ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']
print(f"\n{'True↓/Pred→':<12}", end='')
for label in labels:
    print(f"{label:<8}", end='')
print()
print("-"*80)

for i, row in enumerate(cm):
    print(f"{labels[i]:<12}", end='')
    for val in row:
        print(f"{val:<8}", end='')
    print()

# 대각선 체크
is_perfect = all(cm[i][i] == sum(cm[i]) for i in range(len(cm)))
if is_perfect:
    print(f"\n✅ Perfect diagonal (might be too good!)")
else:
    print(f"\n✅ Has some errors (more realistic)")

print(f"\n[Check 3-2] 샘플 20개 수동 검증")
print("-"*80)

ID2LABEL = {v: k for k, v in DataConfig.LABEL2ID.items()}

for i in range(min(20, len(all_labels))):
    true_label = ID2LABEL[all_labels[i]]
    pred_label = ID2LABEL[all_preds[i]]
    match = "✅" if all_labels[i] == all_preds[i] else "❌"
    
    print(f"Sample {i:2d}: True={true_label:<8} Pred={pred_label:<8} {match}")

# 틀린 샘플만 출력
mismatches = [(i, ID2LABEL[all_labels[i]], ID2LABEL[all_preds[i]]) 
              for i in range(len(all_labels)) 
              if all_labels[i] != all_preds[i]]

if mismatches:
    print(f"\n❌ {len(mismatches)} mismatches found:")
    for i, true_l, pred_l in mismatches[:10]:
        print(f"  Sample {i}: True={true_l} → Pred={pred_l}")
else:
    print(f"\n⚠️  0 mismatches (perfect 100%)")

print("\n" + "="*80)
print("4. Valid 분포 체크")
print("="*80)

valid_label_dist = Counter(ID2LABEL[l] for l in all_labels)
print(f"\nValid distribution:")
for label in ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']:
    count = valid_label_dist[label]
    pct = count / len(all_labels) * 100
    print(f"  {label:<8}: {count:3d} ({pct:5.1f}%)")

# V4/V5 적은지 체크
if valid_label_dist['V4'] < 20 or valid_label_dist['V5'] < 20:
    print(f"\n⚠️  V4 or V5 has very few samples (<20)")
    print(f"   Small errors can cause large metric swings")

print("\n" + "="*80)
print("최종 평가")
print("="*80)
print("\n체크 결과 요약:")
print(f"  1. Session ID overlap: {'❌ YES' if overlap_sessions else '✅ NO'}")
print(f"  2. Prefix overlap: {'❌ YES' if overlap_hashes else '✅ NO'}")
print(f"  3. Full sample overlap: {'❌ YES' if full_overlap else '✅ NO'}")
print(f"  4. Label in text: {'❌ YES' if suspicious_samples else '✅ NO'}")
print(f"  5. Perfect predictions: {'⚠️  YES (100%)' if not mismatches else '✅ NO'}")

if overlap_sessions or overlap_hashes or suspicious_samples:
    print(f"\n❌ CRITICAL ISSUES FOUND - 재검토 필요!")
elif not mismatches:
    print(f"\n⚠️  No critical issues, but 100% is suspicious")
    print(f"   → TARGET-only ablation 추천")
else:
    print(f"\n✅ Looks legitimate")

print("\n" + "="*80)
