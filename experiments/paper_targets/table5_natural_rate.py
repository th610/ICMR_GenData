"""
Table 5. Natural Violation Rate on ESConv (Session-level)
=========================================================
Random audit: N=1,000 sessions — GPT-4o-mini Judge로 ESConv 원본 세션의
자연 발생 violation 비율 측정.

수치 수정: DATA dict만 바꾸면 전체 반영.
실행: python table5_natural_rate.py
"""
import json
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
    "n_sessions": 1000,
    "type_counts": {
        "V1": 32,
        "V2": 18,
        "V3": 48,
        "V4": 10,
        "V5":  5,
    },
}


# ════════════════════════════════════════
# 파생 지표 자동 계산
# ════════════════════════════════════════
def compute(d):
    n = d['n_sessions']
    tc = d['type_counts']
    total_viol = sum(tc.values())
    rates = {k: v / n for k, v in tc.items()}
    return {
        "n_violation_sessions": total_viol,
        "overall_rate": total_viol / n,
        "type_rates": rates,
    }


# ════════════════════════════════════════
# PRINT — 콘솔 테이블
# ════════════════════════════════════════
def print_table():
    d = DATA
    c = compute(d)
    tc = d['type_counts']
    n = d['n_sessions']
    W = 62

    print("=" * W)
    print("  Table 5. Natural Violation Rate on ESConv (Session-level)")
    print("  (Random audit: N={:,} sessions)".format(n))
    print("=" * W)

    print(f"  {'Metric':<50s} {'Value':>8s}")
    print(f"  {'-'*50} {'-'*8}")
    print(f"  {'# evaluated sessions (N)':<50s} {n:>8,d}")
    print(f"  {'# sessions with ≥1 violation (any V1–V5)':<50s} {c['n_violation_sessions']:>8d}")
    print(f"  {'Overall violation rate p_v^S':<50s} {c['overall_rate']:>7.1%}")
    print()

    print(f"  --- Type-wise prevalence (session-level) ---")
    print(f"  {'Type':<10s} {'# sessions':>10s} {'Rate':>10s}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    for vtype in ['V1', 'V2', 'V3', 'V4', 'V5']:
        cnt = tc[vtype]
        rate = c['type_rates'][vtype]
        print(f"  {vtype:<10s} {cnt:>10d} {rate:>9.1%}")
    print(f"  {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Sum':<10s} {c['n_violation_sessions']:>10d} {c['overall_rate']:>9.1%}")
    print("=" * W)
    print()
    print("  Note: Type-wise counts reflect the primary violation type per session.")
    print()


# ════════════════════════════════════════
# FIGURE — Horizontal bar chart
# ════════════════════════════════════════
def generate_figure():
    d = DATA
    c = compute(d)
    tc = d['type_counts']

    types = ['V1', 'V2', 'V3', 'V4', 'V5']
    counts = [tc[t] for t in types]
    rates  = [c['type_rates'][t] for t in types]

    colors = ['#EF5350', '#FFA726', '#FFEE58', '#66BB6A', '#42A5F5']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: Horizontal bar (counts) ──
    y_pos = range(len(types))
    bars1 = ax1.barh(y_pos, counts, color=colors, edgecolor='#333', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(types)
    ax1.invert_yaxis()
    ax1.set_xlabel('# Sessions')
    ax1.set_title('Violation Count by Type')
    for bar, cnt in zip(bars1, counts):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 str(cnt), va='center', fontsize=10, fontweight='bold')

    # ── Right: Horizontal bar (rate %) ──
    pcts = [r * 100 for r in rates]
    bars2 = ax2.barh(y_pos, pcts, color=colors, edgecolor='#333', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(types)
    ax2.invert_yaxis()
    ax2.set_xlabel('Prevalence Rate (%)')
    ax2.set_title('Session-level Prevalence')
    for bar, p in zip(bars2, pcts):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{p:.1f}%', va='center', fontsize=10, fontweight='bold')

    # Overall annotation
    fig.suptitle(
        f"Natural Violation Rate on ESConv  —  "
        f"Overall: {c['n_violation_sessions']}/{d['n_sessions']:,} = {c['overall_rate']:.1%}",
        fontsize=12, fontweight='bold', y=1.02,
    )

    fig.tight_layout()
    fig.savefig(OUT / 'fig_table5_natural_rate.pdf', dpi=300, bbox_inches='tight')
    fig.savefig(OUT / 'fig_table5_natural_rate.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] Figure → {OUT}/fig_table5_natural_rate.pdf/png")


# ════════════════════════════════════════
# SAVE JSON
# ════════════════════════════════════════
def save_json():
    out = Path(__file__).parent / 'table5_natural_rate.json'
    c = compute(DATA)
    payload = {
        "input": DATA,
        "derived": {
            "n_violation_sessions": c['n_violation_sessions'],
            "overall_rate": c['overall_rate'],
            "type_rates": {k: round(v, 4) for k, v in c['type_rates'].items()},
        },
    }
    with open(out, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] JSON → {out}")


if __name__ == '__main__':
    print_table()
    generate_figure()
    save_json()
