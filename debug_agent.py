"""
간단한 에이전트 디버그 - 단계별 실행
"""
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.openai_client import OpenAIClient
from src.agent.step1_context_builder import ContextBuilder
from src.agent.step2_candidate_generator import CandidateGenerator
from src.agent.step3_violation_detector import ViolationDetector
from src.agent.step4_controller import ControllerV2

print("="*80)
print("DEBUG: Agent Pipeline - Step by Step")
print("="*80)

# Load sample
print("\n[1/6] Loading sample...")
gold_path = Path("src/agent/test_gold_300_prefix.json")
with open(gold_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

sample = data['samples'][0]
print(f"  Session ID: {sample['esconv_session_id']}")
print(f"  Gold Label: {sample.get('gold_label', 'Unknown')}")

# Prepare history
history = []
for turn in sample['prefix_dialog']:
    history.append({'speaker': turn['speaker'], 'text': turn['content']})
print(f"  History: {len(history)} turns")

# Initialize components
print("\n[2/6] Initializing components...")
client = OpenAIClient(model="gpt-4o-mini", max_tokens=200)
print("  ✓ OpenAI client")

context_builder = ContextBuilder(llm_client=client, window_size=6, use_summary=False)
print("  ✓ Context builder")

generator = CandidateGenerator(llm_client=client, num_candidates=1, temperature=0.7)
print("  ✓ Candidate generator")

model_path = Path("models/best_model")
detector = ViolationDetector(mode="model", model_path=str(model_path))
print("  ✓ Violation detector")

controller = ControllerV2(llm_client=client)
print("  ✓ Controller")

# Build context
print("\n[3/6] Building context...")
context = context_builder.build_context(history, sample['situation'])
print(f"  Context: {len(context['recent_turns'])} recent turns")

# Generate candidate
print("\n[4/6] Generating candidate...")
print("  (Calling LLM...)")
candidates = generator.generate_candidates(context)
print(f"  Generated: {candidates[0]['text'][:100]}...")

# Detect violations
print("\n[5/6] Detecting violations...")
detections = detector.detect_batch(context, candidates)
print(f"  Label: {detections[0]['top_violation'].upper()}")
print(f"  Confidence: {detections[0]['confidence']:.2%}")
print(f"  Severity: {detections[0]['severity']}")

# Controller decision
print("\n[6/6] Controller decision...")
decision = controller.decide({'label': detections[0]['top_violation'], 'confidence': detections[0]['confidence']})
print(f"  Action: {decision['type'].upper()}")
if decision['type'] == 'rewrite':
    print(f"  Violation: {decision['violation'].upper()}")

print("\n" + "="*80)
print("DEBUG COMPLETE!")
print("="*80)
