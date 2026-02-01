"""
Preprocess all sessions for training
- Summarize previous turns with LLM
- Keep last 6 turns as context window
- Extract last response
"""
import json
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.openai_client import OpenAIClient
from src.utils import load_json, save_json


SUMMARY_SYSTEM = """You are a helpful assistant that summarizes emotional support conversations.
Summarize the given conversation into 3-5 concise bullet points.
Focus on: emotional state, main problems, key progress, important context.
Output format: JSON with key "bullets" containing an array of strings.
Example: {"bullets": ["First point", "Second point", "Third point"]}"""


def summarize_turns(client, turns):
    """Summarize turns into bullet points"""
    
    # Build dialog text
    dialog_lines = []
    for turn in turns:
        speaker = turn.get('speaker', 'unknown')
        content = turn.get('content', turn.get('text', ''))
        dialog_lines.append(f"{speaker}: {content}")
    
    dialog_text = '\n'.join(dialog_lines)
    
    user_prompt = f"""Summarize this conversation into bullet points:

{dialog_text}

Provide 3-5 bullet points in JSON format."""
    
    try:
        response = client.call(
            system_prompt=SUMMARY_SYSTEM,
            user_prompt=user_prompt
        )
        
        # Extract bullets from JSON response
        bullets = response.get('bullets', [])
        
        # Format as bullet points
        return [f"- {b}" if not b.startswith('-') else b for b in bullets]
        
    except Exception as e:
        print(f"  ⚠️  Summary failed: {e}")
        return [f"- Previous conversation with {len(turns)} turns"]


def process_session(client, session, window_size=6):
    """Process a single session with summary + window"""
    
    situation = session.get('situation', '')
    dialog = session.get('dialog', [])
    label = session.get('label', 'unknown')
    session_id = session.get('session_id', 'unknown')
    
    # Extract emotion_type and problem_type from situation if possible
    # For now, use full situation as is
    meta = {
        'emotion_type': 'emotional support',
        'problem_type': situation[:50] + '...' if len(situation) > 50 else situation
    }
    
    # If dialog is short, no summary needed
    if len(dialog) <= window_size:
        summary = []
        context_turns = dialog[:-1] if len(dialog) > 1 else []
        response = dialog[-1].get('content', dialog[-1].get('text', '')) if dialog else ''
    else:
        # Need summary
        previous_turns = dialog[:-window_size]
        recent_turns = dialog[-window_size:-1]
        last_turn = dialog[-1]
        
        # Summarize previous turns
        summary = summarize_turns(client, previous_turns)
        
        context_turns = recent_turns
        response = last_turn.get('content', last_turn.get('text', ''))
    
    # Build processed session
    processed = {
        'session_id': session_id,
        'label': label,
        'meta': meta,
        'summary': summary,
        'context_turns': [
            {
                'speaker': t.get('speaker', 'unknown'),
                'text': t.get('content', t.get('text', ''))
            } for t in context_turns
        ],
        'response': response
    }
    
    return processed


def process_file(client, input_path, output_path, window_size=6):
    """Process all sessions in a file"""
    
    print(f"\n📂 Processing: {input_path.name}")
    data = load_json(str(input_path))
    print(f"   Total: {len(data)} sessions")
    
    processed = []
    
    for i, session in enumerate(data):
        session_id = session.get('session_id', f'session_{i:04d}')
        
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(data)}")
        
        try:
            proc_session = process_session(client, session, window_size)
            processed.append(proc_session)
        except Exception as e:
            print(f"   ❌ Error processing {session_id}: {e}")
            continue
    
    # Save
    save_json(processed, str(output_path))
    print(f"   ✅ Saved: {output_path.name} ({len(processed)} sessions)")
    
    return processed


def main():
    print("\n" + "="*80)
    print("Preprocessing All Sessions")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print("\nWindow size: 6 turns")
    print("Summary: LLM (gpt-4o-mini)\n")
    
    # Setup
    client = OpenAIClient(model="gpt-4o-mini", temperature=0.3, max_tokens=200)
    
    input_files = [
        ('data/generated/normal_400.json', 'normal'),
        ('data/generated/v1_240.json', 'v1'),
        ('data/generated/v2_160.json', 'v2'),
        ('data/generated/v3_200.json', 'v3'),
        ('data/pilot/v4_full_150.json', 'v4'),
        ('data/pilot/v5_full_150.json', 'v5')
    ]
    
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Process each file
    for input_path_str, label in input_files:
        input_path = Path(input_path_str)
        output_path = output_dir / f"{label}_processed.json"
        
        process_file(client, input_path, output_path)
    
    print("\n" + "="*80)
    print("Preprocessing Complete!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print(f"\nProcessed files saved in: {output_dir}/")
    print("\nNext steps:")
    print("1. Run: python scripts/compile_and_split.py (with processed files)")
    print("2. Run: python scripts/train_classifier.py")
    print()


if __name__ == "__main__":
    main()
