"""
Fix missing tokenizer in best_model directory
"""
from transformers import AutoTokenizer

print("Downloading roberta-base tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

print("Saving tokenizer to best_model directory...")
tokenizer.save_pretrained("models/violation_classifier/best_model")

print("✅ Done! Tokenizer files saved.")

# Verify
import os
files = os.listdir("models/violation_classifier/best_model")
print(f"\nFiles in best_model directory:")
for f in sorted(files):
    print(f"  - {f}")
