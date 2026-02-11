"""
Table 3: Expected Cost & Latency Comparison

두 시스템의 기대 비용(지연시간 기준) 비교:
  - Our System: Generate → Detect(RoBERTa) → [Rewrite if flagged]
  - LLM-Judge:  Generate → Judge(GPT-4o-mini)

Expected Cost Model:
  E[C_ours]  = Cg + Cd + (pv·pd + pfp) · Cr
  E[C_judge] = Cg + Cj

  Cg  = Response generation latency (GPT)
  Cd  = Detection latency (RoBERTa model)
  Cr  = Rewrite latency per triggered rewrite (avg cycles)
  Cj  = Judge evaluation latency (GPT-4o-mini)
  pv  = Violation probability in dataset
  pd  = Detection rate (1 - FNR) = P(detect | violation)
  pfp = False Positive Rate = P(flag | normal)

Sections:
  [A] Observed Statistics (pv, pd, pfp)
  [B] Component Latencies (Cg, Cd, Cr, Cj)
  [C] Expected Cost per Session
  [D] Savings Summary

Input:
    experiment_log.jsonl       (timing: generation, detection, rewrite)
    table2/judge_log.jsonl     (timing: judge latency)

Output:
    table3/
        table3_report.txt
        table3_metrics.json
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent
EXPERIMENT_LOG = RESULTS_DIR / "experiment_log.jsonl"
JUDGE_LOG = RESULTS_DIR / "table2" / "judge_log.jsonl"
OUTPUT_DIR = RESULTS_DIR / "table3"

LABELS = ['normal', 'v1', 'v2', 'v3', 'v4', 'v5']


def load_jsonl(path):
    results = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def deduplicate_by_index(entries):
    """sample_index 기준 중복 제거 (마지막 run 사용)"""
    seen = {}
    for e in entries:
        seen[e['sample_index']] = e
    return list(seen.values())


def stats(values):
    if not values:
        return {'mean': 0, 'median': 0, 'p95': 0, 'min': 0, 'max': 0, 'n': 0}
    values = sorted(values)
    n = len(values)
    return {
        'mean': sum(values) / n,
        'median': values[n // 2],
        'p95': values[int(n * 0.95)] if n >= 20 else values[-1],
        'min': values[0],
        'max': values[-1],
        'n': n
    }


def compute_all(exp_data, judge_data):
    """모든 메트릭 계산"""

    # ── [A] Observed Statistics ──
    normals = [r for r in exp_data if r['gold_label'].lower() == 'normal']
    violations = [r for r in exp_data if r['gold_label'].lower() != 'normal']

    n_total = len(exp_data)
    n_normal = len(normals)
    n_violation = len(violations)

    # pv: 위반 비율
    pv = n_violation / n_total if n_total > 0 else 0.0

    # Detection results from experiment_log
    fp = sum(1 for r in normals if r['initial_detection']['predicted_label'].lower() != 'normal')
    fn = sum(1 for r in violations if r['initial_detection']['predicted_label'].lower() == 'normal')

    pfp = fp / n_normal if n_normal > 0 else 0.0   # FPR
    fnr = fn / n_violation if n_violation > 0 else 0.0
    pd = 1.0 - fnr  # detection rate

    # Per-violation detection
    per_v_detect = {}
    for label in ['v1', 'v2', 'v3', 'v4', 'v5']:
        vs = [r for r in violations if r['gold_label'].lower() == label]
        detected = [r for r in vs if r['initial_detection']['predicted_label'].lower() != 'normal']
        per_v_detect[label] = {
            'total': len(vs),
            'detected': len(detected),
            'rate': len(detected) / len(vs) if vs else 0.0
        }

    # ── [B] Component Latencies ──
    # Cg: generation latency
    gen_times = [r['timing']['generation_ms'] for r in exp_data]
    cg = stats(gen_times)

    # Cd: detection latency (RoBERTa)
    det_times = [r['timing']['detection_ms'] for r in exp_data]
    cd = stats(det_times)

    # Cr: rewrite latency (triggered rewrites only)
    rewrite_entries = [r for r in exp_data
                       if r['final_result']['total_retries'] > 0]
    rewrite_times = [r['timing']['rewrite_total_ms'] for r in rewrite_entries]
    cr = stats(rewrite_times) if rewrite_times else stats([0])

    # Average retries when triggered
    retry_counts = [r['final_result']['total_retries'] for r in rewrite_entries]
    avg_retries = sum(retry_counts) / len(retry_counts) if retry_counts else 0

    # Total pipeline latency (ours)
    total_times = [r['timing']['total_ms'] for r in exp_data]
    total_ours = stats(total_times)

    # Cj: judge latency (GPT-4o-mini)
    judge_latencies = [r['judge']['latency_ms'] for r in judge_data
                       if r['judge'].get('error') is None]
    cj = stats(judge_latencies)

    # Model detection latency from judge_log (for cross-validation)
    model_lat_judge = [r['model_latency_ms'] for r in judge_data]
    cd_cross = stats(model_lat_judge)

    # ── [C] Expected Cost per Session ──
    # E[C_ours]  = Cg + Cd + (pv·pd + pfp) · Cr
    # E[C_judge] = Cg + Cj
    p_rewrite = pv * pd + pfp  # probability of triggering rewrite

    ec_ours = cg['mean'] + cd['mean'] + p_rewrite * cr['mean']
    ec_judge = cg['mean'] + cj['mean']

    savings_ms = ec_judge - ec_ours
    savings_pct = savings_ms / ec_judge * 100 if ec_judge > 0 else 0

    # Breakdown: when NO violation detected (fast path)
    ec_ours_fast = cg['mean'] + cd['mean']
    # When violation detected (slow path)
    ec_ours_slow = cg['mean'] + cd['mean'] + cr['mean']

    # ── Gold-corrected analysis ──
    # experiment_log uses GPT-generated responses, so pd reflects GPT's violation rate, not model accuracy
    # From Gold test (Table 1): pd=1.0 (100% detection), pfp=0.0
    pd_gold = 1.0
    pfp_gold = 0.0
    p_rewrite_gold = pv * pd_gold + pfp_gold  # = pv

    ec_ours_gold = cg['mean'] + cd['mean'] + p_rewrite_gold * cr['mean']
    savings_gold = ec_judge - ec_ours_gold
    savings_gold_pct = savings_gold / ec_judge * 100 if ec_judge > 0 else 0

    # Breakeven: at what pv does our system cost == judge?
    # Cg + Cd + pv*Cr = Cg + Cj → pv = (Cj - Cd) / Cr
    breakeven_pv = (cj['mean'] - cd['mean']) / cr['mean'] if cr['mean'] > 0 else float('inf')

    return {
        'observed': {
            'n_total': n_total,
            'n_normal': n_normal,
            'n_violation': n_violation,
            'pv': pv,
            'pd': pd,
            'pfp': pfp,
            'fnr': fnr,
            'fp_count': fp,
            'fn_count': fn,
            'per_violation': per_v_detect
        },
        'latency': {
            'cg': cg,
            'cd': cd,
            'cr': cr,
            'cj': cj,
            'cd_cross': cd_cross,
            'total_ours': total_ours,
            'avg_retries': avg_retries,
            'n_rewrite_triggered': len(rewrite_entries)
        },
        'expected_cost': {
            'p_rewrite': p_rewrite,
            'ec_ours': ec_ours,
            'ec_judge': ec_judge,
            'savings_ms': savings_ms,
            'savings_pct': savings_pct,
            'ec_ours_fast': ec_ours_fast,
            'ec_ours_slow': ec_ours_slow
        },
        'gold_corrected': {
            'pd_gold': pd_gold,
            'pfp_gold': pfp_gold,
            'p_rewrite_gold': p_rewrite_gold,
            'ec_ours_gold': ec_ours_gold,
            'savings_gold_ms': savings_gold,
            'savings_gold_pct': savings_gold_pct,
            'breakeven_pv': breakeven_pv
        }
    }


def format_report(m):
    obs = m['observed']
    lat = m['latency']
    ec = m['expected_cost']

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("  TABLE 3: Expected Cost & Latency Comparison")
    lines.append("=" * 80)

    # ── A. Observed Statistics ──
    lines.append("")
    lines.append("─" * 80)
    lines.append("  [A] Observed Violation Statistics")
    lines.append("─" * 80)
    lines.append(f"  Total sessions:       {obs['n_total']}")
    lines.append(f"  Normal:               {obs['n_normal']}")
    lines.append(f"  Violation:            {obs['n_violation']}")
    lines.append(f"  pv (violation rate):  {obs['pv']:.4f}")
    lines.append(f"  pd (detection rate):  {obs['pd']:.4f}  (1 - FNR)")
    lines.append(f"  pfp (FPR):            {obs['pfp']:.4f}  ({obs['fp_count']}/{obs['n_normal']})")
    lines.append(f"  FNR:                  {obs['fnr']:.4f}  ({obs['fn_count']}/{obs['n_violation']})")
    lines.append("")
    lines.append(f"  Per-Violation Detection:")
    lines.append(f"  {'Type':<8} {'Total':>6} {'Detected':>8} {'Rate':>8}")
    lines.append(f"  {'─'*8} {'─'*6} {'─'*8} {'─'*8}")
    for label in ['v1', 'v2', 'v3', 'v4', 'v5']:
        v = obs['per_violation'][label]
        lines.append(f"  {label.upper():<8} {v['total']:>6} {v['detected']:>8} {v['rate']:>8.2%}")

    # ── B. Component Latencies ──
    lines.append("")
    lines.append("─" * 80)
    lines.append("  [B] Component Latencies (ms)")
    lines.append("─" * 80)
    lines.append(f"  {'Component':<30} {'Mean':>8} {'Median':>8} {'P95':>8} {'N':>6}")
    lines.append(f"  {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")

    for name, st in [('Cg: Generation (GPT)', lat['cg']),
                      ('Cd: Detection (RoBERTa)', lat['cd']),
                      ('Cr: Rewrite (when triggered)', lat['cr']),
                      ('Cj: Judge (GPT-4o-mini)', lat['cj'])]:
        lines.append(f"  {name:<30} {st['mean']:>8.1f} {st['median']:>8.1f} "
                      f"{st['p95']:>8.1f} {st['n']:>6d}")

    lines.append("")
    lines.append(f"  Rewrite triggered:   {lat['n_rewrite_triggered']} sessions")
    lines.append(f"  Avg retries/trigger: {lat['avg_retries']:.2f}")

    # ── C. Expected Cost ──
    lines.append("")
    lines.append("─" * 80)
    lines.append("  [C] Expected Cost per Session (latency, ms)")
    lines.append("─" * 80)
    lines.append("")
    lines.append("  Formula:")
    lines.append("    E[C_ours]  = Cg + Cd + (pv·pd + pfp) · Cr")
    lines.append("    E[C_judge] = Cg + Cj")
    lines.append("")
    lines.append(f"  Parameters:")
    lines.append(f"    Cg  = {lat['cg']['mean']:>8.1f} ms   (response generation)")
    lines.append(f"    Cd  = {lat['cd']['mean']:>8.1f} ms   (RoBERTa detection)")
    lines.append(f"    Cr  = {lat['cr']['mean']:>8.1f} ms   (rewrite when flagged)")
    lines.append(f"    Cj  = {lat['cj']['mean']:>8.1f} ms   (LLM judge evaluation)")
    lines.append(f"    pv  = {obs['pv']:>8.4f}       (violation probability)")
    lines.append(f"    pd  = {obs['pd']:>8.4f}       (detection rate)")
    lines.append(f"    pfp = {obs['pfp']:>8.4f}       (false positive rate)")
    lines.append(f"    P(rewrite) = pv·pd + pfp = {ec['p_rewrite']:.4f}")
    lines.append("")
    lines.append(f"  ┌─────────────────────────────────────────────────────────┐")
    lines.append(f"  │  E[C_ours]  = {lat['cg']['mean']:.1f} + {lat['cd']['mean']:.1f} + "
                  f"{ec['p_rewrite']:.4f} × {lat['cr']['mean']:.1f}")
    lines.append(f"  │             = {ec['ec_ours']:.1f} ms")
    lines.append(f"  │")
    lines.append(f"  │  E[C_judge] = {lat['cg']['mean']:.1f} + {lat['cj']['mean']:.1f}")
    lines.append(f"  │             = {ec['ec_judge']:.1f} ms")
    lines.append(f"  └─────────────────────────────────────────────────────────┘")
    lines.append("")

    # ── D. Summary ──
    lines.append("─" * 80)
    lines.append("  [D] Savings Summary")
    lines.append("─" * 80)
    lines.append(f"  Our System (expected):   {ec['ec_ours']:>10.1f} ms")
    lines.append(f"  LLM-Judge (expected):    {ec['ec_judge']:>10.1f} ms")
    lines.append(f"  Savings:                 {ec['savings_ms']:>10.1f} ms  ({ec['savings_pct']:.1f}%)")
    lines.append("")
    lines.append(f"  Fast path (no violation): {ec['ec_ours_fast']:.1f} ms  "
                  f"(Cg + Cd only)")
    lines.append(f"  Slow path (rewrite):      {ec['ec_ours_slow']:.1f} ms  "
                  f"(Cg + Cd + Cr)")
    lines.append(f"  Judge always:             {ec['ec_judge']:.1f} ms  "
                  f"(Cg + Cj always)")
    lines.append("")

    # ── E. Gold-corrected Analysis ──
    gc = m['gold_corrected']
    lines.append("─" * 80)
    lines.append("  [E] Gold-Corrected Analysis (pd=1.0, pfp=0.0)")
    lines.append("─" * 80)
    lines.append("")
    lines.append("  NOTE: Section [A]의 pd/pfp는 experiment_log.jsonl 기반 (GPT 생성 응답).")
    lines.append("        Gold 테스트에서 모델은 pd=1.0 (100% 탐지), pfp=0.0 (0% 오탐).")
    lines.append("        여기서는 Gold 기준 pd/pfp로 보정된 기대 비용을 계산합니다.")
    lines.append("")
    lines.append(f"  Gold-corrected parameters:")
    lines.append(f"    pd  = {gc['pd_gold']:.4f}  (Gold test: 100% detection)")
    lines.append(f"    pfp = {gc['pfp_gold']:.4f}  (Gold test: 0% false positive)")
    lines.append(f"    P(rewrite) = pv × pd + pfp = {gc['p_rewrite_gold']:.4f}")
    lines.append("")
    lines.append(f"  ┌─────────────────────────────────────────────────────────┐")
    lines.append(f"  │  E[C_ours]  = {lat['cg']['mean']:.1f} + {lat['cd']['mean']:.1f} + "
                  f"{gc['p_rewrite_gold']:.4f} × {lat['cr']['mean']:.1f}")
    lines.append(f"  │             = {gc['ec_ours_gold']:.1f} ms")
    lines.append(f"  │  E[C_judge] = {ec['ec_judge']:.1f} ms")
    lines.append(f"  │  Savings    = {gc['savings_gold_ms']:.1f} ms  ({gc['savings_gold_pct']:.1f}%)")
    lines.append(f"  └─────────────────────────────────────────────────────────┘")
    lines.append("")
    lines.append(f"  Breakeven: Our system = Judge when pv = {gc['breakeven_pv']:.4f}  ({gc['breakeven_pv']:.1%})")
    lines.append(f"  → 위반율이 {gc['breakeven_pv']:.1%} 미만이면 Our System이 더 저렴")
    lines.append("")

    lines.append("=" * 80)
    lines.append("")

    return "\n".join(lines)


def main():
    for path, name in [(EXPERIMENT_LOG, 'experiment_log.jsonl'),
                        (JUDGE_LOG, 'judge_log.jsonl')]:
        if not path.exists():
            print(f"ERROR: {path} not found.")
            sys.exit(1)

    # Load data
    exp_data = load_jsonl(EXPERIMENT_LOG)
    judge_raw = load_jsonl(JUDGE_LOG)
    judge_data = deduplicate_by_index(judge_raw)
    print(f"[DATA] experiment_log: {len(exp_data)} entries")
    print(f"[DATA] judge_log:      {len(judge_data)} unique entries "
          f"(from {len(judge_raw)} total)\n")

    # Compute
    metrics = compute_all(exp_data, judge_data)

    # Report
    report = format_report(metrics)
    print(report)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_DIR / "table3_report.txt", 'w', encoding='utf-8') as f:
        f.write(report)

    with open(OUTPUT_DIR / "table3_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    print(f"[SAVED] {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
