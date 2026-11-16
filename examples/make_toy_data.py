"""Generate SIMULATED prediction files for demonstrating the toolkit.

These are *not* outputs of real language models. Two systems are simulated
answering capital-city questions with controlled error types, so the effect of
each metric (and of calibration) is easy to see:

* system_a: more accurate but sometimes verbose ("The capital is X.") and
  over-confident.
* system_b: less accurate, terse, and better calibrated.

    python examples/make_toy_data.py
"""

import json
from pathlib import Path

import numpy as np

CAPITALS = {
    "France": "Paris", "Germany": "Berlin", "Italy": "Rome", "Spain": "Madrid",
    "Portugal": "Lisbon", "Japan": "Tokyo", "India": "New Delhi", "Nepal": "Kathmandu",
    "Kenya": "Nairobi", "Egypt": "Cairo", "Brazil": "Brasilia", "Argentina": "Buenos Aires",
    "Canada": "Ottawa", "Australia": "Canberra", "Norway": "Oslo", "Sweden": "Stockholm",
    "Poland": "Warsaw", "Greece": "Athens", "Turkey": "Ankara", "Thailand": "Bangkok",
    "Vietnam": "Hanoi", "Peru": "Lima", "Chile": "Santiago", "Ghana": "Accra",
    "Morocco": "Rabat",
}


def simulate(name, p_correct, p_verbose, conf_correct, conf_wrong, rng, n=200):
    countries = list(CAPITALS)
    rows = []
    for i in range(n):
        country = countries[i % len(countries)]
        gold = CAPITALS[country]
        correct = rng.random() < p_correct
        answer = gold if correct else CAPITALS[rng.choice([c for c in countries if c != country])]
        if rng.random() < p_verbose:
            answer = f"The capital of {country} is {answer}."
        conf = float(np.clip(rng.normal(conf_correct if correct else conf_wrong, 0.08), 0.01, 0.99))
        rows.append({"id": f"q{i}", "question": f"What is the capital of {country}?",
                     "references": [gold], "prediction": answer, "confidence": round(conf, 3)})
    out = Path(__file__).parent / f"{name}.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    simulate("system_a", p_correct=0.85, p_verbose=0.30, conf_correct=0.93, conf_wrong=0.88, rng=rng)
    simulate("system_b", p_correct=0.75, p_verbose=0.00, conf_correct=0.80, conf_wrong=0.40, rng=rng)
