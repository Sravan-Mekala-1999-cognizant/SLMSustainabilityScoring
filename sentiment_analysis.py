"""
Sentiment analysis using Ollama SLMs via Open WebUI proxy.
Scores 50 text samples across multiple models and saves results to JSON.
"""

import json
import time
from datetime import datetime
import ollama

# ── CONFIG ────────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://10.120.100.16/ollama"
API_KEY     = "sk-your-key-here"   # Open WebUI: Settings → Account → API Keys

# SLM vs LLM comparison pair — change these to try different combos
MODELS = [
    "llama3.2:latest",   # SLM — 3.2B parameters
    "llama3.3:70b",      # LLM — 70.6B parameters
]

# ── 50 SAMPLE TEXTS ──────────────────────────────────────────────────────────
TEXTS = [
    "I absolutely love this product, it exceeded all my expectations!",
    "The service was terrible and the staff were rude.",
    "It was okay, nothing special but nothing bad either.",
    "Best purchase I've made all year, highly recommend!",
    "Complete waste of money, broke after one week.",
    "Surprisingly good quality for the price.",
    "I'm on the fence about this — has pros and cons.",
    "Fantastic experience from start to finish.",
    "Would not recommend to anyone.",
    "Decent product but delivery took forever.",
    "The new update ruined everything that was good about this app.",
    "Customer support resolved my issue in minutes, very impressed.",
    "Average at best, expected much more.",
    "This changed my life, I use it every single day.",
    "Returned it immediately, not as advertised.",
    "Pretty solid, does exactly what it says on the tin.",
    "Mediocre performance and overpriced.",
    "Outstanding build quality and great value.",
    "Had high hopes but was ultimately disappointed.",
    "Works perfectly, no complaints whatsoever.",
    "The interface is confusing and unintuitive.",
    "Exceeded my expectations in every way possible.",
    "Barely functional and customer service ignored my emails.",
    "Good enough for everyday use.",
    "Incredible product, fast shipping, great communication.",
    "Nothing impressive here, just another generic option.",
    "Really happy with this — bought two more for friends.",
    "Constant crashes and bugs, completely unusable.",
    "Not bad, not great — right in the middle.",
    "The quality has noticeably declined since the last version.",
    "This is exactly what I was looking for!",
    "Regret buying this, total disappointment.",
    "Works as described, solid choice.",
    "Surprisingly terrible given all the good reviews.",
    "Love the design but the functionality needs work.",
    "Far better than competing products I've tried.",
    "Feels cheap despite the premium price tag.",
    "Smooth experience, zero issues after months of use.",
    "Instructions were unclear and setup was a nightmare.",
    "Great for beginners, intuitive and reliable.",
    "The battery life is unacceptable for a device this price.",
    "My team loves it — boosted our productivity significantly.",
    "Overhyped and underwhelming, save your money.",
    "Perfect fit for my needs, couldn't be happier.",
    "Keeps disconnecting, very frustrating.",
    "Solid mid-range option that delivers on promises.",
    "Absolutely terrible, the worst I've ever used.",
    "Does the job well enough, no major complaints.",
    "Mind-blowing performance, worth every penny.",
    "Packaging was damaged and product arrived broken.",
]

SYSTEM_PROMPT = (
    "You are a sentiment analysis engine. "
    "Analyze the sentiment of the user's text and reply ONLY with a valid JSON object — "
    "no explanation, no markdown, no extra text.\n"
    "Format: {\"sentiment\": \"positive\"|\"negative\"|\"neutral\", "
    "\"score\": <float 0.0–1.0>, \"confidence\": \"high\"|\"medium\"|\"low\"}\n"
    "score: 1.0 = most positive, 0.0 = most negative, 0.5 = neutral."
)

# ── CLIENT ────────────────────────────────────────────────────────────────────
client = ollama.Client(
    host=OLLAMA_HOST,
    headers={"Authorization": f"Bearer {API_KEY}"},
)


def list_available_models() -> list[str]:
    """Return model names from Ollama via SDK."""
    response = client.list()
    return [m.model for m in response.models]


def analyze_sentiment(text: str, model: str) -> dict:
    """Score a single text using the given model. Captures Ollama inference metadata."""
    wall_start = datetime.utcnow().isoformat() + "Z"
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            options={"temperature": 0.1, "num_predict": 60},
        )
        wall_end = datetime.utcnow().isoformat() + "Z"
        raw = response.message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        result = json.loads(raw.strip())

        eval_count     = response.eval_count     or 0
        eval_dur_ns    = response.eval_duration  or 0
        prompt_count   = response.prompt_eval_count    or 0
        prompt_dur_ns  = response.prompt_eval_duration or 0
        total_dur_ns   = response.total_duration or 0

        result["meta"] = {
            "wall_start":          wall_start,
            "wall_end":            wall_end,
            "total_duration_ms":   round(total_dur_ns   / 1e6, 2),
            "prompt_tokens":       prompt_count,
            "prompt_duration_ms":  round(prompt_dur_ns  / 1e6, 2),
            "completion_tokens":   eval_count,
            "completion_duration_ms": round(eval_dur_ns / 1e6, 2),
            "tokens_per_sec":      round(eval_count / (eval_dur_ns / 1e9), 2) if eval_dur_ns > 0 else 0,
        }
        return result

    except json.JSONDecodeError:
        return {"sentiment": "error", "score": -1.0, "confidence": "none",
                "meta": {"wall_start": wall_start, "wall_end": datetime.utcnow().isoformat() + "Z"}}
    except Exception as e:
        return {"sentiment": "error", "score": -1.0, "confidence": "none",
                "meta": {"wall_start": wall_start, "wall_end": datetime.utcnow().isoformat() + "Z"},
                "error": str(e)}


def run_analysis(model: str) -> list[dict]:
    """Score all 50 texts with the given model, print a live table."""
    results = []
    print(f"\n{'='*70}")
    print(f"  Model: {model}")
    print(f"{'='*70}")
    print(f"{'#':<4} {'Sentiment':<10} {'Score':<7} {'Conf':<8} Text preview")
    print("-" * 70)

    for i, text in enumerate(TEXTS, 1):
        result = analyze_sentiment(text, model)
        result["text"] = text
        results.append(result)

        sentiment  = result.get("sentiment", "?")
        score      = result.get("score", -1)
        confidence = result.get("confidence", "?")
        preview    = text[:48] + "…" if len(text) > 48 else text
        score_str  = f"{score:.2f}" if score >= 0 else "error"
        print(f"{i:<4} {sentiment:<10} {score_str:<7} {confidence:<8} {preview}")

    return results


def print_summary(model: str, results: list[dict]):
    counts = {}
    for r in results:
        s = r.get("sentiment", "error")
        counts[s] = counts.get(s, 0) + 1

    valid  = [r for r in results if r.get("score", -1) >= 0]
    avg    = sum(r["score"] for r in valid) / len(valid) if valid else 0

    print(f"\n  Summary — {model}")
    for label, count in sorted(counts.items()):
        bar = "█" * count
        print(f"    {label:<10} {count:>3}  {bar}")
    print(f"    avg score  {avg:.3f}  ({len(valid)}/{len(results)} parsed)")


def main():
    print("=" * 70)
    print(f"  Ollama Sentiment Scorer")
    print(f"  Host : {OLLAMA_HOST}")
    print("=" * 70)

    # Verify connection and list models
    try:
        available = list_available_models()
        print(f"\nModels on this Ollama instance ({len(available)} total):")
        for m in available:
            print(f"  • {m}")
    except Exception as e:
        print(f"\nCould not connect to {OLLAMA_HOST}: {e}")
        print("Check that the VM is reachable and OLLAMA_HOST is correct.")
        return

    # Confirm which models we'll run
    models_to_run = [m for m in MODELS if m in available]
    missing = [m for m in MODELS if m not in available]
    if missing:
        print(f"\nNot found (skipping): {missing}")
    if not models_to_run:
        print("None of the MODELS list are available. Update MODELS in the script.")
        return
    print(f"\nRunning analysis with: {models_to_run}")

    all_results = {}
    for model in models_to_run:
        t0         = time.time()
        wall_start = datetime.utcnow().isoformat() + "Z"
        results    = run_analysis(model)
        wall_end   = datetime.utcnow().isoformat() + "Z"
        elapsed    = time.time() - t0

        valid = [r for r in results if r.get("meta", {}).get("tokens_per_sec", 0) > 0]
        avg_tps = sum(r["meta"]["tokens_per_sec"] for r in valid) / len(valid) if valid else 0
        total_tokens = sum(
            r.get("meta", {}).get("prompt_tokens", 0) + r.get("meta", {}).get("completion_tokens", 0)
            for r in results
        )

        all_results[model] = {
            "wall_start":    wall_start,
            "wall_end":      wall_end,
            "elapsed_sec":   round(elapsed, 2),
            "total_tokens":  total_tokens,
            "avg_tokens_per_sec": round(avg_tps, 2),
            "results":       results,
        }
        print_summary(model, results)
        print(f"  time : {elapsed:.1f}s  avg {avg_tps:.1f} tok/s  total tokens: {total_tokens}")

    # Save to JSON
    output_file = "sentiment_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {output_file}")


if __name__ == "__main__":
    main()
