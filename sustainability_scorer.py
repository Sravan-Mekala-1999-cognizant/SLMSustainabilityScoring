"""
Ollama SLM vs LLM sustainability comparison.

Inputs:
  sentiment_results.json   from sentiment_analysis.py
  vm_metrics.csv           from vm_monitor.py  (optional but needed for energy)

Output:
  comparison_report.json   full data
  printed report in terminal

Usage:
  python sustainability_scorer.py
  python sustainability_scorer.py --results sentiment_results.json --vm vm_metrics.csv
"""

import argparse
import csv
import json
import os
from datetime import datetime, timezone

# Local electricity cost and carbon intensity — adjust to your region
ELECTRICITY_COST_PER_KWH = 0.12   # USD
GRID_CARBON_INTENSITY     = 400    # gCO2eq per kWh


# ── VM METRICS ────────────────────────────────────────────────────────────────

def load_vm_metrics(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def parse_ts(s: str):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T"))
    except ValueError:
        return None


def resource_window(vm_rows: list[dict], wall_start: str, wall_end: str) -> dict:
    """Slice vm_metrics rows within the model's run window and aggregate stats."""
    t0 = parse_ts(wall_start)
    t1 = parse_ts(wall_end)
    if not t0 or not t1 or not vm_rows:
        return {}

    window = []
    for row in vm_rows:
        ts = parse_ts(row.get("timestamp", ""))
        if ts is None:
            continue
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if t0.tzinfo is None:
            t0 = t0.replace(tzinfo=timezone.utc)
        if t1.tzinfo is None:
            t1 = t1.replace(tzinfo=timezone.utc)
        if t0 <= ts <= t1:
            window.append(row)

    if not window:
        return {"samples": 0}

    def floats(key):
        vals = []
        for r in window:
            try:
                vals.append(float(r[key]))
            except (KeyError, ValueError, TypeError):
                pass
        return vals

    def stats(vals):
        if not vals:
            return None, None, None
        return round(sum(vals)/len(vals), 1), round(max(vals), 1), round(min(vals), 1)

    cpu_avg, cpu_peak, _   = stats(floats("cpu_pct"))
    ram_avg, ram_peak, _   = stats(floats("ram_used_gb"))
    gpu_avg, gpu_peak, _   = stats(floats("gpu0_util_pct"))
    vram_avg, vram_peak, _ = stats([v/1024 for v in floats("gpu0_mem_used_mb")])
    pwr_avg, pwr_peak, _   = stats(floats("gpu0_power_w"))
    temp_avg, temp_peak, _ = stats(floats("gpu0_temp_c"))

    return {
        "samples":        len(window),
        "cpu_avg_pct":    cpu_avg,    "cpu_peak_pct":    cpu_peak,
        "ram_avg_gb":     ram_avg,    "ram_peak_gb":     ram_peak,
        "gpu_avg_pct":    gpu_avg,    "gpu_peak_pct":    gpu_peak,
        "vram_avg_gb":    vram_avg,   "vram_peak_gb":    vram_peak,
        "gpu_avg_power_w":pwr_avg,    "gpu_peak_power_w":pwr_peak,
        "gpu_avg_temp_c": temp_avg,   "gpu_peak_temp_c": temp_peak,
    }


# ── METRICS EXTRACTION ────────────────────────────────────────────────────────

def extract_model_metrics(model: str, data: dict, vm_rows: list[dict]) -> dict:
    results   = data.get("results", [])
    elapsed   = data.get("elapsed_sec", 0)
    total_tok = data.get("total_tokens", 0)
    avg_tps   = data.get("avg_tokens_per_sec", 0)

    parse_ok  = sum(1 for r in results if r.get("score", -1) >= 0)
    quality   = round(parse_ok / len(results) * 100, 1) if results else 0

    res = resource_window(vm_rows, data.get("wall_start", ""), data.get("wall_end", ""))

    avg_watts = res.get("gpu_avg_power_w") or 0
    energy_wh = round(avg_watts * (elapsed / 3600), 4)
    energy_kwh_per_1k = round((energy_wh / 1000) / (total_tok / 1000), 6) if total_tok > 0 else 0
    co2_g_per_1k      = round(energy_kwh_per_1k * GRID_CARBON_INTENSITY, 4)
    cost_usd_per_1k   = round((energy_wh / 1000) * ELECTRICITY_COST_PER_KWH / (total_tok / 1000), 6) if total_tok > 0 else 0

    return {
        "model":           model,
        "elapsed_sec":     elapsed,
        "total_tokens":    total_tok,
        "avg_tokens_per_sec": avg_tps,
        "parse_success_pct":  quality,
        "resource":        res,
        "energy_wh":       energy_wh,
        "energy_source":   "measured" if avg_watts > 0 else "unavailable (run vm_monitor.py)",
        "energy_kwh_per_1k_tokens": energy_kwh_per_1k,
        "co2_g_per_1k_tokens":      co2_g_per_1k,
        "cost_usd_per_1k_tokens":   cost_usd_per_1k,
    }


# ── SENTIMENT AGREEMENT ───────────────────────────────────────────────────────

def agreement_analysis(model_a: str, data_a: dict, model_b: str, data_b: dict) -> dict:
    results_a = data_a.get("results", [])
    results_b = data_b.get("results", [])

    agree, disagree, details = 0, 0, []
    scores_a, scores_b = [], []
    conf_dist = {model_a: {}, model_b: {}}

    for i, (ra, rb) in enumerate(zip(results_a, results_b)):
        sa = ra.get("sentiment")
        sb = rb.get("sentiment")
        matched = sa == sb and sa not in (None, "error")

        if matched:
            agree += 1
        elif sa not in (None, "error") and sb not in (None, "error"):
            disagree += 1
            details.append({
                "text":    ra.get("text", "")[:60],
                model_a:   sa,
                model_b:   sb,
                "score_" + model_a: ra.get("score", -1),
                "score_" + model_b: rb.get("score", -1),
            })

        if ra.get("score", -1) >= 0:
            scores_a.append(ra["score"])
        if rb.get("score", -1) >= 0:
            scores_b.append(rb["score"])

        for model, result, dist in [(model_a, ra, conf_dist[model_a]),
                                    (model_b, rb, conf_dist[model_b])]:
            c = result.get("confidence", "unknown")
            dist[c] = dist.get(c, 0) + 1

    total = agree + disagree
    return {
        "agreement_pct":   round(agree / total * 100, 1) if total > 0 else 0,
        "agree_count":     agree,
        "disagree_count":  disagree,
        "avg_score":       {
            model_a: round(sum(scores_a)/len(scores_a), 3) if scores_a else None,
            model_b: round(sum(scores_b)/len(scores_b), 3) if scores_b else None,
        },
        "confidence_dist": conf_dist,
        "disagreements":   details,
    }


# ── EFFICIENCY RATIO ──────────────────────────────────────────────────────────

def efficiency_ratio(metrics_a: dict, metrics_b: dict) -> dict:
    """Quality per watt-hour — higher means more sustainable."""
    def ratio(m):
        q    = m["parse_success_pct"]
        e    = m["energy_wh"]
        tps  = m["avg_tokens_per_sec"]
        return {
            "quality_per_wh":      round(q / e, 2)    if e > 0 else None,
            "tokens_per_wh":       round(m["total_tokens"] / e, 1) if e > 0 else None,
            "quality_per_sec":     round(q / m["elapsed_sec"], 3)  if m["elapsed_sec"] > 0 else None,
        }

    return {
        metrics_a["model"]: ratio(metrics_a),
        metrics_b["model"]: ratio(metrics_b),
    }


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(ma: dict, mb: dict, agreement: dict, efficiency: dict):
    W = 75
    sep = "=" * W

    def row(label, a, b, unit=""):
        av = f"{a}{unit}" if a is not None else "n/a"
        bv = f"{b}{unit}" if b is not None else "n/a"
        print(f"  {label:<32} {av:>16} {bv:>16}")

    def header():
        print(f"  {'':32} {ma['model'][:16]:>16} {mb['model'][:16]:>16}")
        print("  " + "-" * (W - 2))

    print("\n" + sep)
    print("  SLM vs LLM — RESOURCE & SUSTAINABILITY COMPARISON")
    print(sep)

    # ── Performance
    print("\n  PERFORMANCE")
    header()
    row("Elapsed (s)",         ma["elapsed_sec"],          mb["elapsed_sec"],        "s")
    row("Total tokens",        ma["total_tokens"],         mb["total_tokens"])
    row("Avg tokens/sec",      ma["avg_tokens_per_sec"],   mb["avg_tokens_per_sec"], " tok/s")
    row("Parse success",       ma["parse_success_pct"],    mb["parse_success_pct"],  "%")

    # ── Resource utilisation
    ra, rb = ma.get("resource", {}), mb.get("resource", {})
    print("\n  RESOURCE UTILISATION (during inference window)")
    header()
    row("VM samples captured", ra.get("samples"),          rb.get("samples"))
    row("CPU avg %",           ra.get("cpu_avg_pct"),      rb.get("cpu_avg_pct"),    "%")
    row("CPU peak %",          ra.get("cpu_peak_pct"),     rb.get("cpu_peak_pct"),   "%")
    row("RAM avg (GB)",        ra.get("ram_avg_gb"),       rb.get("ram_avg_gb"),     " GB")
    row("GPU util avg %",      ra.get("gpu_avg_pct"),      rb.get("gpu_avg_pct"),    "%")
    row("GPU util peak %",     ra.get("gpu_peak_pct"),     rb.get("gpu_peak_pct"),   "%")
    row("VRAM avg (GB)",       ra.get("vram_avg_gb"),      rb.get("vram_avg_gb"),    " GB")
    row("VRAM peak (GB)",      ra.get("vram_peak_gb"),     rb.get("vram_peak_gb"),   " GB")
    row("GPU power avg (W)",   ra.get("gpu_avg_power_w"),  rb.get("gpu_avg_power_w")," W")
    row("GPU power peak (W)",  ra.get("gpu_peak_power_w"), rb.get("gpu_peak_power_w")," W")
    row("GPU temp avg (°C)",   ra.get("gpu_avg_temp_c"),   rb.get("gpu_avg_temp_c"), "°C")

    # ── Energy & cost
    print("\n  ENERGY & COST  " + f"({ma['energy_source']})")
    header()
    row("Energy used (Wh)",    ma["energy_wh"],             mb["energy_wh"],          " Wh")
    row("kWh per 1k tokens",   ma["energy_kwh_per_1k_tokens"], mb["energy_kwh_per_1k_tokens"])
    row("CO₂ per 1k tokens",   ma["co2_g_per_1k_tokens"],   mb["co2_g_per_1k_tokens"]," g")
    row("Cost per 1k tokens",  ma["cost_usd_per_1k_tokens"],mb["cost_usd_per_1k_tokens"]," $")

    # ── Sentiment quality
    print("\n  SENTIMENT QUALITY")
    header()
    avg = agreement["avg_score"]
    row("Avg sentiment score", avg.get(ma["model"]),        avg.get(mb["model"]))
    for model, dist in agreement["confidence_dist"].items():
        short = model[:16]
        label = f"  Confidence ({short[:12]})"
        vals  = "  ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        print(f"  {label:<32} {vals}")

    # ── Agreement
    print(f"\n  AGREEMENT BETWEEN MODELS")
    print(f"  Agreement rate    : {agreement['agreement_pct']}%  "
          f"({agreement['agree_count']} agree / {agreement['disagree_count']} disagree)")
    if agreement["disagreements"]:
        print(f"\n  Top disagreements (first 5):")
        for d in agreement["disagreements"][:5]:
            print(f"    \"{d['text']}...\"")
            print(f"      {ma['model'][:20]}: {d.get(ma['model'])}  "
                  f"{mb['model'][:20]}: {d.get(mb['model'])}")

    # ── Efficiency ratio
    print(f"\n  EFFICIENCY RATIO  (higher = more sustainable)")
    header()
    ea = efficiency.get(ma["model"], {})
    eb = efficiency.get(mb["model"], {})
    row("Quality per Wh",      ea.get("quality_per_wh"),   eb.get("quality_per_wh"))
    row("Tokens per Wh",       ea.get("tokens_per_wh"),    eb.get("tokens_per_wh"))
    row("Quality per second",  ea.get("quality_per_sec"),  eb.get("quality_per_sec"))

    # ── Verdict
    print("\n" + sep)
    print("  VERDICT")
    print(sep)
    if ea.get("quality_per_wh") and eb.get("quality_per_wh"):
        winner = ma["model"] if ea["quality_per_wh"] > eb["quality_per_wh"] else mb["model"]
        loser  = mb["model"] if winner == ma["model"] else ma["model"]
        ratio  = max(ea["quality_per_wh"], eb["quality_per_wh"]) / max(min(ea["quality_per_wh"], eb["quality_per_wh"]), 0.0001)
        print(f"  Most energy-efficient : {winner}")
        print(f"  Ratio                 : {ratio:.1f}x more quality per Wh than {loser}")
    if agreement["agreement_pct"] >= 80:
        print(f"  Sentiment agreement   : HIGH ({agreement['agreement_pct']}%) — SLM results are reliable")
    elif agreement["agreement_pct"] >= 60:
        print(f"  Sentiment agreement   : MODERATE ({agreement['agreement_pct']}%) — review disagreements")
    else:
        print(f"  Sentiment agreement   : LOW ({agreement['agreement_pct']}%) — significant divergence")
    print(sep + "\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="sentiment_results.json")
    parser.add_argument("--vm",      default="vm_metrics.csv")
    parser.add_argument("--output",  default="comparison_report.json")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"Results file not found: {args.results}")
        print("Run sentiment_analysis.py first.")
        return

    with open(args.results) as f:
        all_results = json.load(f)

    models = list(all_results.keys())
    if len(models) < 2:
        print(f"Need at least 2 models in results. Found: {models}")
        return

    vm_rows = load_vm_metrics(args.vm)
    print(f"VM metrics : {len(vm_rows)} samples {'loaded' if vm_rows else '— not found, energy metrics will be unavailable'}")
    print(f"Models     : {models[0]}  vs  {models[1]}\n")

    ma = extract_model_metrics(models[0], all_results[models[0]], vm_rows)
    mb = extract_model_metrics(models[1], all_results[models[1]], vm_rows)

    agreement  = agreement_analysis(models[0], all_results[models[0]],
                                    models[1], all_results[models[1]])
    efficiency = efficiency_ratio(ma, mb)

    print_report(ma, mb, agreement, efficiency)

    report = {"model_a": ma, "model_b": mb,
              "agreement": agreement, "efficiency": efficiency}
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Full report saved → {args.output}")


if __name__ == "__main__":
    main()
