"""
Sentiment analysis using Ollama models via Open WebUI API.
Scores 50 text samples across multiple available SLMs.
"""

import json
import time
from openai import OpenAI

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE_URL = "http://<your-vm-ip>:3000/api/v1"   # replace with your VM address
API_KEY  = "sk-your-key-here"                   # replace with your Open WebUI key

# Models to test — update to match what your Ollama instance has
MODELS = [
    "llama3.2:latest",
    "mistral:latest",
    "gemma2:latest",
    # add/remove based on what's available at your endpoint
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

SYSTEM_PROMPT = """You are a sentiment analysis engine. Analyze the sentiment of the given text and respond ONLY with a valid JSON object in this exact format:
{"sentiment": "positive" | "negative" | "neutral", "score": <float 0.0-1.0>, "confidence": "high" | "medium" | "low"}

- score: 1.0 = most positive, 0.0 = most negative, 0.5 = neutral
- confidence: how certain you are of the label
No explanation, no extra text — only the JSON object."""

# ── CLIENT ────────────────────────────────────────────────────────────────────
client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def list_available_models() -> list[str]:
    """Fetch model list from the endpoint."""
    models = client.models.list()
    return [m.id for m in models.data]


def analyze_sentiment(text: str, model: str) -> dict:
    """Run sentiment analysis on a single text using the given model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0.1,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"sentiment": "error", "score": -1, "confidence": "none", "raw": raw}
    except Exception as e:
        return {"sentiment": "error", "score": -1, "confidence": "none", "error": str(e)}


def run_analysis(model: str) -> list[dict]:
    """Score all 50 texts with the given model."""
    results = []
    print(f"\n{'='*60}")
    print(f"  Model: {model}")
    print(f"{'='*60}")
    print(f"{'#':<4} {'Sentiment':<10} {'Score':<7} {'Conf':<8} Text")
    print("-" * 80)

    for i, text in enumerate(TEXTS, 1):
        result = analyze_sentiment(text, model)
        result["text"] = text
        results.append(result)

        sentiment  = result.get("sentiment", "?")
        score      = result.get("score", -1)
        confidence = result.get("confidence", "?")
        preview    = text[:50] + "..." if len(text) > 50 else text
        print(f"{i:<4} {sentiment:<10} {score:<7.2f} {confidence:<8} {preview}")
        time.sleep(0.1)  # small delay to avoid hammering the server

    return results


def print_summary(model: str, results: list[dict]):
    counts = {"positive": 0, "negative": 0, "neutral": 0, "error": 0}
    for r in results:
        counts[r.get("sentiment", "error")] = counts.get(r.get("sentiment", "error"), 0) + 1

    valid = [r for r in results if r.get("score", -1) >= 0]
    avg_score = sum(r["score"] for r in valid) / len(valid) if valid else 0

    print(f"\nSummary for {model}:")
    print(f"  Positive : {counts['positive']}")
    print(f"  Negative : {counts['negative']}")
    print(f"  Neutral  : {counts['neutral']}")
    print(f"  Errors   : {counts['error']}")
    print(f"  Avg score: {avg_score:.3f}")


def main():
    print("Connecting to Open WebUI / Ollama endpoint...")
    print(f"Endpoint: {BASE_URL}\n")

    # Show available models
    try:
        available = list_available_models()
        print("Available models:", available)
    except Exception as e:
        print(f"Could not fetch model list: {e}")
        available = []

    # Filter MODELS list to only those actually available
    models_to_run = [m for m in MODELS if m in available] or MODELS
    if not models_to_run:
        print("No matching models found. Check MODELS list or run list_available_models().")
        return

    all_results = {}
    for model in models_to_run:
        results = run_analysis(model)
        all_results[model] = results
        print_summary(model, results)

    # Save full results to JSON
    output_file = "sentiment_results.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results saved to {output_file}")


if __name__ == "__main__":
    main()
