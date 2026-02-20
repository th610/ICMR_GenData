"""
Run all paper_targets tables — 전체 한번에 실행
================================================
실행: python run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
scripts = [
    'table1_safety.py',
    'table2_gate.py',
    'table3_cost.py',
    'table4_detector.py',
    'table5_natural_rate.py',
]

print("=" * 70)
print("  PAPER TARGETS — Run All Tables")
print("=" * 70)

for s in scripts:
    path = HERE / s
    print(f"\n{'─'*70}")
    print(f"  Running: {s}")
    print(f"{'─'*70}\n")
    result = subprocess.run([sys.executable, str(path)], capture_output=False)
    if result.returncode != 0:
        print(f"  [FAIL] {s} exited with code {result.returncode}")

print(f"\n{'═'*70}")
print(f"  All done. Figures: {HERE / 'figures'}/")
print(f"{'═'*70}")
