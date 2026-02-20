"""
Generate all 6 figures for the paper.
Output: experiments/figures/ (PDF + PNG)
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from collections import Counter

# ── Style ──
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

BASE = Path(__file__).parent.parent
OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)

COLORS = {
    'normal': '#4CAF50',
    'v1': '#2196F3',
    'v2': '#FF9800',
    'v3': '#9C27B0',
    'v4': '#F44336',
    'v5': '#795548',
}
CLASS_NAMES = ['Normal', 'V1', 'V2', 'V3', 'V4', 'V5']
CLASS_KEYS = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']


# ════════════════════════════════════════════════════════════
# Figure 1: Per-Class Precision / Recall / F1 (Table 1)
# ════════════════════════════════════════════════════════════
def figure1():
    with open(BASE / 'table1' / 'table1_metrics.json') as f:
        m = json.load(f)

    classes = CLASS_KEYS
    precision = [m['per_class'][c]['precision'] for c in classes]
    recall    = [m['per_class'][c]['recall'] for c in classes]
    f1        = [m['per_class'][c]['f1'] for c in classes]

    x = np.arange(len(classes))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w, precision, w, label='Precision', color='#1976D2', alpha=0.85)
    bars2 = ax.bar(x,     recall,    w, label='Recall',    color='#388E3C', alpha=0.85)
    bars3 = ax.bar(x + w, f1,        w, label='F1-Score',  color='#F57C00', alpha=0.85)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, h),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Violation Type')
    ax.set_ylabel('Score')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0.8, 1.02)
    ax.axhline(y=m['macro_f1'], color='red', linestyle='--', alpha=0.5, label=f"Macro F1={m['macro_f1']:.3f}")
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT / 'fig1_per_class_performance.pdf')
    fig.savefig(OUT / 'fig1_per_class_performance.png')
    plt.close(fig)
    print("[OK] Figure 1: Per-Class Performance")


# ════════════════════════════════════════════════════════════
# Figure 2: Confusion Matrix Heatmap (Table 1)
# ════════════════════════════════════════════════════════════
def figure2():
    with open(BASE / 'table1' / 'table1_metrics.json') as f:
        m = json.load(f)

    cm = np.array(m['confusion_matrix'])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')

    # Text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                   color=color, fontsize=12, fontweight='bold')

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Gold Label')

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_confusion_matrix.pdf')
    fig.savefig(OUT / 'fig2_confusion_matrix.png')
    plt.close(fig)
    print("[OK] Figure 2: Confusion Matrix")


# ════════════════════════════════════════════════════════════
# Figure 3: Model vs Judge Per-Class F1 (Table 2)
# ════════════════════════════════════════════════════════════
def figure3():
    with open(BASE / 'table2' / 'table2_metrics.json') as f:
        m = json.load(f)

    classes = CLASS_KEYS
    model_f1 = [m['model_per_class_f1'][c] for c in classes]
    judge_f1 = [m['judge_per_class_f1'][c] for c in classes]

    x = np.arange(len(classes))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars1 = ax.bar(x - w/2, model_f1, w, label=f"RoBERTa Model (Macro F1={m['model_macro_f1']:.3f})",
                   color='#1976D2', alpha=0.85)
    bars2 = ax.bar(x + w/2, judge_f1, w, label=f"GPT-4o-mini Judge (Macro F1={m['judge_macro_f1']:.3f})",
                   color='#E65100', alpha=0.85)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width()/2, max(h, 0.02)),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Violation Type')
    ax.set_ylabel('F1-Score')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')

    fig.tight_layout()
    fig.savefig(OUT / 'fig3_model_vs_judge.pdf')
    fig.savefig(OUT / 'fig3_model_vs_judge.png')
    plt.close(fig)
    print("[OK] Figure 3: Model vs Judge")


# ════════════════════════════════════════════════════════════
# Figure 4: Cost Breakdown Comparison (Table 3)
# ════════════════════════════════════════════════════════════
def figure4():
    with open(BASE / 'table3' / 'table3_metrics.json') as f:
        m = json.load(f)

    lat = m['latency']
    cg = lat['cg']['mean']
    cd = lat['cd']['mean']
    cr = lat['cr']['mean']
    cj = lat['cj']['mean']

    obs = m['observed']
    p_rw = obs['pv'] * obs['pd'] + obs['pfp']  # P(rewrite)

    # Our system breakdown
    our_gen = cg
    our_det = cd
    our_rw = p_rw * cr

    # Judge breakdown
    judge_gen = cg
    judge_eval = cj

    fig, ax = plt.subplots(figsize=(7, 5))

    # Stacked bars
    systems = ['Our System\n(Detect + Rewrite)', 'LLM Judge\n(GPT-4o-mini)']
    x = [0, 1]

    # Our system
    b1 = ax.bar(0, our_gen, 0.5, color='#42A5F5', label='Response Generation')
    b2 = ax.bar(0, our_det, 0.5, bottom=our_gen, color='#66BB6A', label='RoBERTa Detection')
    b3 = ax.bar(0, our_rw, 0.5, bottom=our_gen+our_det, color='#EF5350', label=f'Rewrite (P={p_rw:.2f})')

    # Judge
    b4 = ax.bar(1, judge_gen, 0.5, color='#42A5F5')
    b5 = ax.bar(1, judge_eval, 0.5, bottom=judge_gen, color='#FFA726', label='Judge Evaluation')

    our_total = our_gen + our_det + our_rw
    judge_total = judge_gen + judge_eval

    ax.text(0, our_total + 80, f'{our_total:.0f}ms', ha='center', fontsize=12, fontweight='bold')
    ax.text(1, judge_total + 80, f'{judge_total:.0f}ms', ha='center', fontsize=12, fontweight='bold')

    # Component labels inside bars
    ax.text(0, our_gen/2, f'{our_gen:.0f}ms', ha='center', va='center', fontsize=9, color='white')
    ax.text(0, our_gen + our_det/2, f'{our_det:.0f}ms', ha='center', va='center', fontsize=9, color='white')
    ax.text(0, our_gen + our_det + our_rw/2, f'{our_rw:.0f}ms', ha='center', va='center', fontsize=9, color='white')
    ax.text(1, judge_gen/2, f'{judge_gen:.0f}ms', ha='center', va='center', fontsize=9, color='white')
    ax.text(1, judge_gen + judge_eval/2, f'{judge_eval:.0f}ms', ha='center', va='center', fontsize=9, color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_ylabel('Expected Latency (ms)')
    ax.legend(loc='upper right')
    ax.set_xlim(-0.5, 1.5)

    fig.tight_layout()
    fig.savefig(OUT / 'fig4_cost_breakdown.pdf')
    fig.savefig(OUT / 'fig4_cost_breakdown.png')
    plt.close(fig)
    print("[OK] Figure 4: Cost Breakdown")


# ════════════════════════════════════════════════════════════
# Figure 5: Safety Pipeline Flow (Table 4 - Sankey-style)
# ════════════════════════════════════════════════════════════
def figure5():
    with open(BASE / 'table4' / 'table4_metrics.json') as f:
        m = json.load(f)

    total = m['total_violations']       # 220
    detected = m['detected']             # 212
    missed = m['missed']                 # 8
    corrected = m['corrected']           # 46
    failsafed = m['failsafed']           # 166
    leaked = m['still_violated']         # 0

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Boxes
    box_style = dict(boxstyle='round,pad=0.4', edgecolor='#333', linewidth=1.5)

    # Stage 1: Input
    ax.text(0.5, 5, f'Gold\nViolations\n{total}', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(**box_style, facecolor='#FFCDD2'))

    # Stage 2: Detection
    ax.text(3, 7.5, f'Detected\n{detected} ({detected/total*100:.1f}%)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(**box_style, facecolor='#BBDEFB'))
    ax.text(3, 2.5, f'Missed\n{missed} ({missed/total*100:.1f}%)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(**box_style, facecolor='#FFECB3'))

    # Stage 3: Resolution
    ax.text(6, 8.5, f'Corrected\n{corrected} ({corrected/detected*100:.1f}%)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(**box_style, facecolor='#C8E6C9'))
    ax.text(6, 6.2, f'Failsafe\n{failsafed} ({failsafed/detected*100:.1f}%)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(**box_style, facecolor='#D1C4E9'))

    # Stage 4: Final
    ax.text(9, 7.5, f'Safe Output\n{detected} (100%)', ha='center', va='center',
            fontsize=12, fontweight='bold',
            bbox=dict(**box_style, facecolor='#A5D6A7'))
    ax.text(9, 2.5, f'Leakage\n{leaked} (0%)', ha='center', va='center',
            fontsize=11, fontweight='bold',
            bbox=dict(**box_style, facecolor='#E8F5E9'))

    # Arrows
    arrow_kw = dict(arrowstyle='->', lw=2, color='#555')
    thin_arrow = dict(arrowstyle='->', lw=1.2, color='#AAA')

    ax.annotate('', xy=(1.7, 7.2), xytext=(1.2, 5.5), arrowprops=dict(**arrow_kw))
    ax.annotate('', xy=(1.7, 3.0), xytext=(1.2, 4.5), arrowprops=dict(**thin_arrow))
    ax.annotate('', xy=(4.7, 8.3), xytext=(4.2, 7.7), arrowprops=dict(**arrow_kw))
    ax.annotate('', xy=(4.7, 6.5), xytext=(4.2, 7.3), arrowprops=dict(**arrow_kw))
    ax.annotate('', xy=(7.7, 7.8), xytext=(7.2, 8.3), arrowprops=dict(**arrow_kw))
    ax.annotate('', xy=(7.7, 7.2), xytext=(7.2, 6.5), arrowprops=dict(**arrow_kw))

    # Title
    # Labels
    ax.text(1.5, 8.0, 'RoBERTa\nDetection', ha='center', fontsize=9, style='italic', color='#666')
    ax.text(4.5, 9.2, 'Rewrite\nLoop', ha='center', fontsize=9, style='italic', color='#666')
    ax.text(7.5, 9.2, 'Final\nOutput', ha='center', fontsize=9, style='italic', color='#666')

    fig.tight_layout()
    fig.savefig(OUT / 'fig5_safety_pipeline.pdf')
    fig.savefig(OUT / 'fig5_safety_pipeline.png')
    plt.close(fig)
    print("[OK] Figure 5: Safety Pipeline Flow")


# ════════════════════════════════════════════════════════════
# Figure 6: Retry Distribution + Per-Type Analysis (Table 4)
# ════════════════════════════════════════════════════════════
def figure6():
    with open(BASE / 'table4' / 'table4_metrics.json') as f:
        m = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # ── Left: Retry Distribution ──
    dist = m['retry_stats']['distribution']
    retries = [0, 1, 2, 3]
    # Missed (8) = 0 retries (not detected)
    counts = [m['missed'], int(dist.get('1', 0)), int(dist.get('2', 0)), int(dist.get('3', 0))]
    colors = ['#FFCDD2', '#81D4FA', '#A5D6A7', '#EF9A9A']
    labels_r = ['Missed\n(not detected)', '1 retry\n(corrected)', '2 retries\n(corrected)', '3 retries\n(failsafe)']

    bars = ax1.bar(range(4), counts, color=colors, edgecolor='#333', linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(cnt), ha='center', fontsize=11, fontweight='bold')

    ax1.set_xticks(range(4))
    ax1.set_xticklabels(labels_r, fontsize=9)
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('(a) Retry Distribution (N=220)')

    # ── Right: Per-Type Detection & Correction ──
    types = ['V1', 'V2', 'V3', 'V4', 'V5']
    type_keys = ['v1', 'v2', 'v3', 'v4', 'v5']

    det_rates = [m['per_violation'][k]['detection_rate'] for k in type_keys]
    corr_rates = [m['per_violation'][k]['corrected'] / m['per_violation'][k]['detected']
                  if m['per_violation'][k]['detected'] > 0 else 0 for k in type_keys]

    x = np.arange(len(types))
    w = 0.35

    b1 = ax2.bar(x - w/2, det_rates, w, label='Detection Rate', color='#1976D2', alpha=0.85)
    b2 = ax2.bar(x + w/2, corr_rates, w, label='Correction Rate', color='#388E3C', alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax2.annotate(f'{h:.0%}', xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=9)

    ax2.set_xlabel('Violation Type')
    ax2.set_ylabel('Rate')
    ax2.set_title('(b) Detection & Correction Rate by Type')
    ax2.set_xticks(x)
    ax2.set_xticklabels(types)
    ax2.set_ylim(0, 1.15)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUT / 'fig6_retry_analysis.pdf')
    fig.savefig(OUT / 'fig6_retry_analysis.png')
    plt.close(fig)
    print("[OK] Figure 6: Retry Analysis")


# ══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating all figures...\n")
    figure1()
    figure2()
    figure3()
    figure4()
    figure5()
    figure6()
    print(f"\n✅ All figures saved to: {OUT}/")
    print("  fig1_per_class_performance.pdf/png")
    print("  fig2_confusion_matrix.pdf/png")
    print("  fig3_model_vs_judge.pdf/png")
    print("  fig4_cost_breakdown.pdf/png")
    print("  fig5_safety_pipeline.pdf/png")
    print("  fig6_retry_analysis.pdf/png")
