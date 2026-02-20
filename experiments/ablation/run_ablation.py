"""
Ablation Study — Table 4 (Gold Injection) 데이터 기반 시뮬레이션

두 가지 ablation:
  (a) w/o Rewrite Loop: 탐지 즉시 failsafe (rewrite 없이)
  (b) w/o Failsafe: rewrite만 수행, failsafe 비활성화

기존 table4_log.jsonl 로그를 재분석하여 시뮬레이션.
API 호출 불필요.

Output:
    experiments/ablation/
        ablation_report.txt
        ablation_metrics.json
        ablation_figure.pdf / .png
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

BASE = Path(__file__).parent.parent
TABLE4_LOG = BASE / "table4" / "table4_log.jsonl"
TABLE4_METRICS = BASE / "table4" / "table4_metrics.json"
OUTPUT_DIR = Path(__file__).parent
OUTPUT_DIR.mkdir(exist_ok=True)

FULL_EXP_LOG = BASE / "experiment_log.jsonl"


def load_table4():
    with open(TABLE4_LOG) as f:
        return [json.loads(l) for l in f]


def load_table4_metrics():
    with open(TABLE4_METRICS) as f:
        return json.load(f)


def load_full_exp():
    with open(FULL_EXP_LOG) as f:
        return [json.loads(l) for l in f]


# ═══════════════════════════════════════════════════════════
# FULL SYSTEM (baseline from Table 4)
# ═══════════════════════════════════════════════════════════
def analyze_full_system(samples):
    """기존 시스템 (Table 4 결과 그대로)"""
    total = len(samples)
    detected = sum(1 for s in samples if s['initial_detection']['predicted_label'] != 'normal')
    missed = total - detected
    corrected = sum(1 for s in samples if not s['final_result']['used_failsafe'] and s['rewrite_iterations'])
    failsafed = sum(1 for s in samples if s['final_result']['used_failsafe'])

    # Table 4 정의: leakage = 탐지되었으나 해결 실패 (still_violated)
    # detected된 것 중 corrected도 failsafed도 아닌 것
    still_violated = detected - corrected - failsafed
    resolved = corrected + failsafed
    resolved_rate = resolved / detected if detected > 0 else 0

    # Latency
    latencies = [s['timing']['total_ms'] for s in samples]
    avg_latency = np.mean(latencies)

    # Response quality: corrected = natural, failsafe = templated
    natural_responses = corrected
    template_responses = failsafed

    return {
        "name": "Full System",
        "total": total,
        "detected": detected,
        "detection_rate": detected / total,
        "missed": missed,
        "corrected": corrected,
        "failsafed": failsafed,
        "resolved": resolved,
        "resolved_rate": resolved_rate,
        "still_violated": still_violated,
        "leakage_rate": still_violated / detected if detected > 0 else 0,
        "natural_responses": natural_responses,
        "template_responses": template_responses,
        "natural_rate": natural_responses / max(detected, 1),
        "avg_latency_ms": avg_latency,
    }


# ═══════════════════════════════════════════════════════════
# ABLATION (a): w/o Rewrite Loop
# ═══════════════════════════════════════════════════════════
def analyze_wo_rewrite(samples):
    """
    Rewrite Loop 제거: 탐지 즉시 failsafe.
    - 탐지됨(≠Normal) → 모두 failsafe (rewrite 없이)
    - 탐지 안됨(=Normal) → 그대로 통과 (leaked)
    """
    total = len(samples)
    detected = sum(1 for s in samples if s['initial_detection']['predicted_label'] != 'normal')
    missed = total - detected
    corrected = 0  # No rewrite → no correction
    failsafed = detected  # All detected → direct failsafe

    # Table 4 정의: leakage = 탐지되었으나 해결 실패
    still_violated = 0  # 탐지 즉시 failsafe → 해결 실패 없음
    resolved = failsafed
    resolved_rate = resolved / detected if detected > 0 else 0

    # Latency: only context + detection (no rewrite overhead)
    latencies = []
    for s in samples:
        ctx_ms = s['timing']['context_build_ms']
        det_ms = s['timing']['detection_ms']
        latencies.append(ctx_ms + det_ms)
    avg_latency = np.mean(latencies)

    return {
        "name": "w/o Rewrite",
        "total": total,
        "detected": detected,
        "detection_rate": detected / total,
        "missed": missed,
        "corrected": corrected,
        "failsafed": failsafed,
        "resolved": resolved,
        "resolved_rate": resolved_rate,
        "still_violated": still_violated,
        "leakage_rate": still_violated / detected if detected > 0 else 0,
        "natural_responses": 0,
        "template_responses": failsafed,
        "natural_rate": 0.0,
        "avg_latency_ms": avg_latency,
    }


# ═══════════════════════════════════════════════════════════
# ABLATION (b): w/o Failsafe
# ═══════════════════════════════════════════════════════════
def analyze_wo_failsafe(samples):
    """
    Failsafe 비활성화: rewrite만 수행, 3회 후에도 violation이면 그대로 반환.
    - 탐지 → rewrite loop (최대 3회) → 성공하면 corrected
    - 3회 실패해도 그냥 반환 → leaked (violation 통과)
    - 탐지 안됨 → leaked
    """
    total = len(samples)
    detected = sum(1 for s in samples if s['initial_detection']['predicted_label'] != 'normal')
    missed = total - detected

    # Corrected: same as full system (rewrite가 normal로 만든 것)
    corrected = sum(1 for s in samples if not s['final_result']['used_failsafe'] and s['rewrite_iterations'])

    # Failed rewrite: would have been failsafed in full system, now leaked
    failed_rewrite = sum(1 for s in samples if s['final_result']['used_failsafe'])
    failsafed = 0  # No failsafe

    # Table 4 정의: leakage = 탐지되었으나 해결 실패
    # failsafe 없으므로 failed_rewrite = still_violated
    still_violated = failed_rewrite
    resolved = corrected
    resolved_rate = resolved / detected if detected > 0 else 0

    # Latency: context + detection + rewrite attempts (no failsafe overhead)
    latencies = [s['timing']['total_ms'] for s in samples]
    avg_latency = np.mean(latencies)

    return {
        "name": "w/o Failsafe",
        "total": total,
        "detected": detected,
        "detection_rate": detected / total,
        "missed": missed,
        "corrected": corrected,
        "failsafed": failsafed,
        "resolved": resolved,
        "resolved_rate": resolved_rate,
        "still_violated": still_violated,
        "leakage_rate": still_violated / detected if detected > 0 else 0,
        "natural_responses": corrected,
        "template_responses": 0,
        "natural_rate": corrected / max(detected, 1) if detected > 0 else 0,
        "avg_latency_ms": avg_latency,
    }


# ═══════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════
def generate_report(full, wo_rw, wo_fs):
    lines = []
    w = lines.append

    w("=" * 80)
    w("  ABLATION STUDY: Component Contribution Analysis")
    w("=" * 80)
    w("")
    w("  Source: Table 4 Gold Injection data (N=220 violation samples)")
    w("  Method: Simulation on existing table4_log.jsonl")
    w("")

    # ── Main comparison table ──
    w("─" * 80)
    w("  [A] Safety Metrics")
    w("─" * 80)
    w(f"  {'Metric':<30s} {'Full System':>14s} {'w/o Rewrite':>14s} {'w/o Failsafe':>14s}")
    w(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*14}")

    metrics = [
        ("Detected", "detected"),
        ("Missed (not detected)", "missed"),
        ("Corrected (→Normal)", "corrected"),
        ("Failsafed", "failsafed"),
        ("Still Violated (leaked)", "still_violated"),
    ]
    for label, key in metrics:
        w(f"  {label:<30s} {full[key]:>14d} {wo_rw[key]:>14d} {wo_fs[key]:>14d}")

    w("")
    w(f"  {'Detection Rate':.<30s} {full['detection_rate']:>13.1%} {wo_rw['detection_rate']:>13.1%} {wo_fs['detection_rate']:>13.1%}")
    w(f"  {'Resolved Rate (of detected)':.<30s} {full['resolved_rate']:>13.1%} {wo_rw['resolved_rate']:>13.1%} {wo_fs['resolved_rate']:>13.1%}")
    w(f"  {'Leakage Rate (of detected)':.<30s} {full['leakage_rate']:>13.1%} {wo_rw['leakage_rate']:>13.1%} {wo_fs['leakage_rate']:>13.1%}")

    w("")
    w("─" * 80)
    w("  [B] Response Quality")
    w("─" * 80)
    w(f"  {'Metric':<30s} {'Full System':>14s} {'w/o Rewrite':>14s} {'w/o Failsafe':>14s}")
    w(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*14}")
    w(f"  {'Natural Responses':.<30s} {full['natural_responses']:>14d} {wo_rw['natural_responses']:>14d} {wo_fs['natural_responses']:>14d}")
    w(f"  {'Template Responses':.<30s} {full['template_responses']:>14d} {wo_rw['template_responses']:>14d} {wo_fs['template_responses']:>14d}")
    w(f"  {'Natural Rate (of resolved)':.<30s} {full['natural_rate']:>13.1%} {wo_rw['natural_rate']:>13.1%} {wo_fs['natural_rate']:>13.1%}")

    w("")
    w("─" * 80)
    w("  [C] Efficiency")
    w("─" * 80)
    w(f"  {'Metric':<30s} {'Full System':>14s} {'w/o Rewrite':>14s} {'w/o Failsafe':>14s}")
    w(f"  {'─'*30} {'─'*14} {'─'*14} {'─'*14}")
    w(f"  {'Avg Latency (ms)':.<30s} {full['avg_latency_ms']:>13.1f} {wo_rw['avg_latency_ms']:>13.1f} {wo_fs['avg_latency_ms']:>13.1f}")

    w("")
    w("─" * 80)
    w("  [D] Analysis")
    w("─" * 80)
    w("")
    w("  (a) w/o Rewrite Loop:")
    w(f"      - Resolved Rate: {wo_rw['resolved_rate']:.1%} (same as Full: {full['resolved_rate']:.1%})")
    w(f"        All detected violations go directly to failsafe")
    w(f"      - Leakage: {wo_rw['leakage_rate']:.1%} (same as Full: {full['leakage_rate']:.1%})")
    w(f"      - Quality: 0% natural responses (all template-based)")
    w(f"        Δ Natural = -{full['natural_responses']} responses ({full['corrected']} corrections lost)")
    w(f"      - Latency: {wo_rw['avg_latency_ms']:.0f}ms vs {full['avg_latency_ms']:.0f}ms")
    w(f"        Speedup: {full['avg_latency_ms'] / wo_rw['avg_latency_ms']:.1f}x faster")
    w(f"      → Rewrite loop contributes RESPONSE QUALITY, not safety.")
    w(f"        {full['corrected']} natural corrections would be lost (all become template).")
    w("")
    w("  (b) w/o Failsafe:")
    w(f"      - Resolved Rate: {wo_fs['resolved_rate']:.1%} (vs Full: {full['resolved_rate']:.1%})")
    w(f"      - Leakage: {wo_fs['leakage_rate']:.1%} ({wo_fs['still_violated']} of {wo_fs['detected']} detected violations leak)")
    w(f"      - Quality: Only {wo_fs['corrected']} natural responses remain")
    w(f"      - Latency: Same as Full System ({wo_fs['avg_latency_ms']:.0f}ms)")
    w(f"      → Failsafe is CRITICAL for safety guarantee.")
    w(f"        Without it, {wo_fs['still_violated']} detected violations still leak through.")
    w("")
    w("  KEY FINDINGS:")
    w(f"    1. Rewrite Loop: Quality component (+{full['corrected']} natural corrections)")
    w(f"    2. Failsafe: Safety component (prevents {wo_fs['still_violated']} leaks among detected)")
    w(f"    3. Both are necessary: Rewrite for quality, Failsafe for safety guarantee")
    w("")
    w("=" * 80)

    return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════
# FIGURE: Ablation comparison
# ═══════════════════════════════════════════════════════════
def generate_figure(full, wo_rw, wo_fs):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    systems = ['Full System', 'w/o Rewrite', 'w/o Failsafe']
    colors_sys = ['#1976D2', '#FF9800', '#F44336']

    # ── (a) Safety: Resolved Rate ──
    ax = axes[0]
    resolved_rates = [full['resolved_rate'], wo_rw['resolved_rate'], wo_fs['resolved_rate']]
    leak_rates = [full['leakage_rate'], wo_rw['leakage_rate'], wo_fs['leakage_rate']]

    x = np.arange(3)
    bars_safe = ax.bar(x, resolved_rates, 0.6, color=colors_sys, alpha=0.85, edgecolor='#333', linewidth=0.8)

    for i, (bar, rr, lr) in enumerate(zip(bars_safe, resolved_rates, leak_rates)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{rr:.1%}', ha='center', fontsize=11, fontweight='bold')
        if lr > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.06,
                    f'Leak: {lr:.1%}', ha='center', fontsize=8, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel('Resolved Rate (of detected)')
    ax.set_title('(a) Safety Guarantee')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)

    # ── (b) Response Quality ──
    ax = axes[1]
    natural = [full['natural_responses'], wo_rw['natural_responses'], wo_fs['natural_responses']]
    template = [full['template_responses'], wo_rw['template_responses'], wo_fs['template_responses']]
    violated_cnt = [full['still_violated'], wo_rw['still_violated'], wo_fs['still_violated']]

    w = 0.6
    b1 = ax.bar(x, natural, w, label='Natural (corrected)', color='#66BB6A', edgecolor='#333', linewidth=0.5)
    b2 = ax.bar(x, template, w, bottom=natural, label='Template (failsafe)', color='#FFCA28', edgecolor='#333', linewidth=0.5)
    b3 = ax.bar(x, violated_cnt, w, bottom=[n+t for n, t in zip(natural, template)],
                label='Still Violated (leaked)', color='#EF5350', edgecolor='#333', linewidth=0.5)

    for i in range(3):
        if natural[i] > 5:
            ax.text(x[i], natural[i]/2, str(natural[i]), ha='center', va='center', fontsize=10, fontweight='bold')
        if template[i] > 5:
            ax.text(x[i], natural[i] + template[i]/2, str(template[i]), ha='center', va='center', fontsize=10, fontweight='bold')
        if violated_cnt[i] > 5:
            ax.text(x[i], natural[i] + template[i] + violated_cnt[i]/2, str(violated_cnt[i]),
                    ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel('Number of Responses')
    ax.set_title('(b) Response Quality')
    ax.legend(fontsize=8, loc='upper right')

    # ── (c) Latency ──
    ax = axes[2]
    latencies = [full['avg_latency_ms'], wo_rw['avg_latency_ms'], wo_fs['avg_latency_ms']]
    bars = ax.bar(x, latencies, 0.6, color=colors_sys, alpha=0.85, edgecolor='#333', linewidth=0.8)

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{lat:.0f}ms', ha='center', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.set_ylabel('Avg Latency (ms)')
    ax.set_title('(c) Efficiency')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'ablation_figure.pdf', dpi=300)
    fig.savefig(OUTPUT_DIR / 'ablation_figure.png', dpi=300)
    plt.close(fig)
    print("[OK] Ablation figure saved")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Loading Table 4 log data...")
    samples = load_table4()
    print(f"  {len(samples)} samples loaded\n")

    # Run analyses
    full = analyze_full_system(samples)
    wo_rw = analyze_wo_rewrite(samples)
    wo_fs = analyze_wo_failsafe(samples)

    # Generate report
    report = generate_report(full, wo_rw, wo_fs)
    print(report)

    # Save report
    with open(OUTPUT_DIR / 'ablation_report.txt', 'w') as f:
        f.write(report)

    # Save metrics
    metrics = {
        "full_system": full,
        "wo_rewrite": wo_rw,
        "wo_failsafe": wo_fs,
    }
    with open(OUTPUT_DIR / 'ablation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Generate figure
    generate_figure(full, wo_rw, wo_fs)

    print(f"\n[SAVED] {OUTPUT_DIR}/")
    print("  → ablation_report.txt")
    print("  → ablation_metrics.json")
    print("  → ablation_figure.pdf/png")
