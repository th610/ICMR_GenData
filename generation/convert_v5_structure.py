"""
V5 데이터 구조 변환 스크립트
==========================
기존 generated_V5.json의 구조를:
  - prefix_conversation → prefix_dialog
  - generated_turn (JSON 문자열) → generated_dialog (list)
  - label → primary_label

로 변환합니다.
"""
import json
from pathlib import Path

def convert_v5():
    """Convert V5 structure to match other labels"""
    input_file = Path("generation/outputs/generated/generated_V5.json")
    output_file = Path("generation/outputs/generated/generated_V5_converted.json")
    
    # Load original V5
    with open(input_file, 'r', encoding='utf-8') as f:
        v5_data = json.load(f)
    
    print(f"Converting {len(v5_data['samples'])} V5 samples...")
    
    converted_samples = []
    for i, sample in enumerate(v5_data['samples']):
        # Parse generated_turn JSON string
        try:
            generated_turn_data = json.loads(sample['generated_turn'])
            
            # Build new structure
            converted = {
                "esconv_session_id": sample["esconv_session_id"],
                "situation": sample.get("situation", ""),  # 없으면 빈 문자열
                "prefix_dialog": sample["prefix_conversation"],  # 키 변경
                "generated_dialog": generated_turn_data["dialog"],  # JSON 파싱
                "primary_label": "V5",  # label → primary_label
                "generation_method": "esconv_prefix_with_insertion",
                "violation_turn_index": 3,
                "violation_reason": generated_turn_data.get("violation_reason", "")
            }
            
            converted_samples.append(converted)
            
        except Exception as e:
            print(f"  ✗ Error at index {i}: {e}")
    
    # Save converted data
    output_data = {
        "metadata": {
            "label": "V5",
            "total_samples": len(converted_samples),
            "distribution": {"V5": len(converted_samples)}
        },
        "samples": converted_samples
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Converted {len(converted_samples)} samples")
    print(f"Saved: {output_file}")
    
    # 구조 확인
    print(f"\n{'='*70}")
    print("Sample structure check:")
    print(f"{'='*70}")
    if converted_samples:
        sample = converted_samples[0]
        print(f"Keys: {list(sample.keys())}")
        print(f"prefix_dialog: {len(sample['prefix_dialog'])} turns")
        print(f"generated_dialog: {len(sample['generated_dialog'])} turns")
        print(f"primary_label: {sample['primary_label']}")
        print("✅ Structure matches V1/V2/V3/V4!")

if __name__ == "__main__":
    convert_v5()
