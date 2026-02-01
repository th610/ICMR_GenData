"""
Test Judge on a single counsel-chat sample to verify it's working correctly
"""
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.llm.prompts import JUDGE_SYSTEM, build_judge_prompt, RETRY_MESSAGE


def test_judge_single_sample():
    """Test Judge on one counsel-chat sample"""
    
    # Load one sample
    with open('data/external/counsel_chat.json', encoding='utf-8') as f:
        data = json.load(f)
    
    sample = data[0]  # First sample
    
    print("="*60)
    print("Testing Judge on Counsel-Chat Sample")
    print("="*60)
    
    print("\n[QUESTION]")
    print(sample['question'])
    if sample.get('question_text'):
        print(sample['question_text'])
    
    print("\n[ANSWER]")
    print(sample['answer'][:300] + "..." if len(sample['answer']) > 300 else sample['answer'])
    
    # Build dialog
    dialog_text = f"Seeker: {sample['question']}"
    if sample.get('question_text'):
        dialog_text += " " + sample['question_text']
    dialog_text += f"\nSupporter: {sample['answer']}"
    
    # Build prompt
    prompt = build_judge_prompt(dialog_text)
    
    print("\n" + "="*60)
    print("Judge Prompt (first 500 chars):")
    print("="*60)
    print(prompt[:500] + "...")
    
    # Call Judge
    print("\n" + "="*60)
    print("Calling Judge API...")
    print("="*60)
    
    llm = OpenAIClient()
    
    try:
        result = llm.call(
            system_prompt=JUDGE_SYSTEM,
            user_prompt=prompt,
            retry_message=RETRY_MESSAGE
        )
        
        print("\n[RAW RESULT]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        print("\n" + "="*60)
        print("Parsed Results:")
        print("="*60)
        
        # Parse like in analyze_counsel_chat.py
        label = result.get('top_violation', 'Normal')
        if label and label != 'Normal':
            label = label.lower()
        else:
            label = 'normal'
        
        reasoning = result.get('reasoning', '')
        
        print(f"Label: {label}")
        print(f"Reasoning length: {len(reasoning)} chars")
        print(f"Reasoning: {reasoning[:200] if reasoning else '(EMPTY)'}")
        
        # Check all fields
        print("\n[ALL RESULT KEYS]")
        for key in result.keys():
            value = result[key]
            if isinstance(value, str):
                print(f"  {key}: {value[:100]}..." if len(value) > 100 else f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_judge_single_sample()
