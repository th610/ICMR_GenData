"""
Table 3. Safety/Trigger Statistics & Expected Cost — VACCINE vs LLM-Judge
=========================================================================
Section 5.4 — R3 비용/지연 트레이드오프

수치 수정: DATA dict만 바꾸면 전체 반영 (파생 지표 자동 계산).
실행: python table3_cost.py
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).parent / 'figures'
OUT.mkdir(exist_ok=True)

# ════════════════════════════════════════
# DATA — 여기만 수정하면 전체 반영
# ════════════════════════════════════════
DATA = {
    # Gold sample composition
    "n_total": 300,
    "n_normal": 80,
    "n_violation": 220,
    "pv": 0.7333,

    # VACCINE (Detector gate)
    "vaccine": {
        "fnr": 7 / 220,           # 0.0318
        "fpr": 10 / 80,           # 0.1250
        "fn_count": 7,
        "fp_count": 10,
        "tp_count": 213,
        "cg": 1493.2,             # Generation (ms)
        "cd": 29.1,               # Detector gate (ms)
        "cr": 6496.3,             # Rewrite single call (ms)
        "avg_rw_attempts": 1.353,
    },

    # LLM-Judge (monitoring-only baseline)
    "judge": {
        "fnr": 18 / 220,          # 0.0818
        "fpr": 6 / 80,            # 0.0750
        "fn_count": 18,
        "fp_count": 6,
        "tp_count": 202,
        "cg": 1493.2,
        "cj": 878.8,              # Judge gate (ms)
        "cr": 0.0,                # Judge does NOT rewrite
        "avg_rw_attempts": 0.0,
    },
}


# ════════════════════════════════════════
# 파생 지표 자동 계산
# ════════════════════════════════════════
def compute(d):
    n = d['n_total']
    v = d['vaccine']
    j = d['judge']

    # P(flagged as violation) = (TP + FP) / n
    v_flagged = (v['tp_count'] + v['fp_count']) / n   # 223/300
    j_flagged = (j['tp_count'] + j['fp_count']) / n   # 208/300

    # Effective rewrite latency
    v_cr_eff = v['avg_rw_attempts'] * v['cr']
    j_cr_eff = j['avg_rw_attempts'] * j['cr']

    # Expected cost
    v_ecost = v['cg'] + v['cd'] + v_flagged * v_cr_eff
    j_ecost = j['cg'] + j['cj'] + j_flagged * j_cr_eff   # j_cr_eff=0

    # Breakeven
    p_breakeven = (j['cj'] - v['cd']) / v_cr_eff if v_cr_eff else float('inf')

    return {
        "v_flagged": v_flagged,
        "j_flagged": j_flagged,
        "v_cr_eff": v_cr_eff,
        "j_cr_eff": j_cr_eff,
        "v_ecost": v_ecost,
        "j_ecost": j_ecost,
        "p_breakeven": p_breakeven,
    }


# ════════════════════════════════════════
# PRINT — 콘솔 테이블
# ════════════════════════════════════════
def print_table():
    d = DATA
    v = d['vaccine']
    j = d['judge']
    c = compute(d)

    W = 75

    print("=" * W)
    print("  Table 3(a). Safety / Trigger Statistics (Gold n=300)")
    print("=" * W)
    print(f"  {'Metric':<40s} {'VACCINE':>16s} {'LLM-Judge':>16s}")
    print(f"  {'-'*40} {'-'*16} {'-'*16}")
    print(f"  {'Violation rate (p_v)':<40s} {d['pv']:>16.4f} {d['pv']:>16.4f}")
    print(f"  {'FNR (Violation → Normal)':<40s} {v['fnr']:>16.4f} {j['fnr']:>16.4f}")
    print(f"  {'FPR (Normal → Violation)':<40s} {v['fpr']:>16.4f} {j['fpr']:>16.4f}")
    print(f"  {'P(flagged as violation)':<40s} {c['v_flagged']:>16.4f} {c['j_flagged']:>16.4f}")
    print(f"  {'Residual unsafe exposure (≈FNR)':<40s} {v['fnr']:>16.4f} {j['fnr']:>16.4f}")
    print()
    print()

    print("=" * W)
    print("  Table 3(b). Latency Components & Expected Cost (ms)")
    print("=" * W)
    print(f"  {'Metric':<40s} {'VACCINE':>16s} {'LLM-Judge':>16s}")
    print(f"  {'-'*40} {'-'*16} {'-'*16}")
    print(f"  {'Generation (C_g)':<40s} {v['cg']:>16.1f} {j['cg']:>16.1f}")
    print(f"  {'Gate latency (C_d / C_j)':<40s} {v['cd']:>16.1f} {j['cj']:>16.1f}")
    print(f"  {'Rewrite latency (C_r)':<40s} {v['cr']:>16.1f} {j['cr']:>16.1f}")
    print(f"  {'Avg rewrite attempts':<40s} {v['avg_rw_attempts']:>16.3f} {j['avg_rw_attempts']:>16.3f}")
    print(f"  {'Effective rewrite latency (C_r,eff)':<40s} {c['v_cr_eff']:>16.1f} {c['j_cr_eff']:>16.1f}")
    print(f"  {'Expected latency E[C]':<40s} {c['v_ecost']:>16.1f} {c['j_ecost']:>16.1f}")
    print()
    print(f"  Breakeven:  P* = (C_j − C_d) / C_r,eff = ({j['cj']:.1f} − {v['cd']:.1f}) / {c['v_cr_eff']:.1f} = {c['p_breakeven']:.4f} ({c['p_breakeven']:.1%})")
    print()


# ════════════════════════════════════════
# FIGURE — Cost Breakdown (stacked bar, side by side)
# ════════════════════════════════════════
def generate_figure():
    d = DATA
    v = d['vaccine']
    j = d['judge']
    c = compute(d)

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── VACCINE ──
    rw_cost_v = c['v_flagged'] * c['v_cr_eff']
    ax.bar(0, v['cg'], 0.5, color='#42A5F5', label='Response Generation (C_g)')
    ax.bar(0, v['cd'], 0.5, bottom=v['cg'], color='#66BB6A', label='Detector Gate (C_d)')
    ax.bar(0, rw_cost_v, 0.5, bottom=v['cg'] + v['cd'],
           color='#EF5350', label=f"Rewrite (P={c['v_flagged']:.2f} × C_r,eff={c['v_cr_eff']:.0f})")

    # ── LLM-Judge ──
    ax.bar(1, j['cg'], 0.5, color='#42A5F5')
    ax.bar(1, j['cj'], 0.5, bottom=j['cg'], color='#FFA726', label='Judge Gate (C_j)')

    # Totals
    ax.text(0, c['v_ecost'] + 150, f"{c['v_ecost']:.0f} ms", ha='center', fontsize=12, fontweight='bold')
    ax.text(1, c['j_ecost'] + 150, f"{c['j_ecost']:.0f} ms", ha='center', fontsize=12, fontweight='bold')

    # Labels inside bars
    ax.text(0, v['cg'] / 2, f"{v['cg']:.0f}", ha='center', va='center', fontsize=9, color='white')
    ax.text(0, v['cg'] + v['cd'] / 2, f"{v['cd']:.0f}", ha='center', va='center', fontsize=8, color='white')
    if rw_cost_v > 200:
        ax.text(0, v['cg'] + v['cd'] + rw_cost_v / 2, f"{rw_cost_v:.0f}",
                ha='center', va='center', fontsize=9, color='white')
    ax.text(1, j['cg'] / 2, f"{j['cg']:.0f}", ha='center', va='center', fontsize=9, color='white')
    ax.text(1, j['cg'] + j['cj'] / 2, f"{j['cj']:.0f}", ha='center', va='center', fontsize=9, color='white')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['VACCINE\n(Detect + Rewrite)', 'LLM-Judge\n(Monitoring Only)'])
    ax.set_ylabel('Expected Latency per Sample (ms)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(-0.5, 1.5)

    fig.tight_layout()
    fig.savefig(OUT / 'fig_table3_cost_breakdown.pdf', dpi=300)
    fig.savefig(OUT / 'fig_table3_cost_breakdown.png', dpi=300)
    plt.close(fig)
    print(f"[OK] Figure → {OUT}/fig_table3_cost_breakdown.pdf/png")


# ════════════════════════════════════════
# SAVE JSON
# ════════════════════════════════════════
def save_json():
    out = Path(__file__).parent / 'table3_cost.json'
    derived = compute(DATA)
    payload = {
        "input": {k: (v if not isinstance(v, dict) else v) for k, v in DATA.items()},
        "derived": {k: float(v) for k, v in derived.items()},
    }
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] JSON → {out}")


if __name__ == '__main__':
    print_table()
    generate_figure()
    save_json()
