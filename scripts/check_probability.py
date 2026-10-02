#!/usr/bin/env python3
"""Reproduce Assignment 1 calculations using only the Python standard library.

Run from any directory with ``python3 /path/to/scripts/check_probability.py``.
The script reads the saved answer JSON for consistency and writes no files.
"""

from fractions import Fraction
import json
from math import isclose, log2
from pathlib import Path


def main() -> None:
    results: dict[int, dict[str, Fraction | float | bool]] = {}

    # Q1: independent events; exact rational arithmetic avoids rounding errors.
    a, b = Fraction(2, 5), Fraction(3, 10)
    intersection = a * b
    union = a + b - intersection
    assert intersection == Fraction(3, 25)
    assert union == Fraction(29, 50)
    results[1] = {"intersection": intersection, "union": union}
    print("Q1: Independent events")
    print(f"  (a) P(A intersection B) = 0.4 * 0.3 = {intersection} = {float(intersection):.2f}")
    print(f"  (b) P(A union B) = 0.4 + 0.3 - 0.12 = {union} = {float(union):.2f}\n")

    # Q2: compare the conditional and marginal; check the intersection as well.
    a, b, a_given_b = Fraction(1, 2), Fraction(2, 5), Fraction(7, 10)
    intersection = a_given_b * b
    product = a * b
    independent = a_given_b == a
    assert not independent
    assert intersection == Fraction(7, 25)
    assert product == Fraction(1, 5)
    assert intersection != product
    results[2] = {
        "independent": independent,
        "intersection": intersection,
        "product_of_marginals": product,
    }
    print("Q2: Testing independence")
    print("  P(A | B) = 0.7 differs from P(A) = 0.5: not independent.")
    print(f"  P(A intersection B) = 0.7 * 0.4 = {intersection} = {float(intersection):.2f}")
    print(f"  P(A) P(B) = 0.5 * 0.4 = {product} = {float(product):.2f}\n")

    # Q3: law of total probability followed by Bayes' rule.
    a = Fraction(3, 5)
    b_given_a, b_given_not_a = Fraction(1, 2), Fraction(1, 5)
    b = b_given_a * a + b_given_not_a * (1 - a)
    posterior = b_given_a * a / b
    assert b == Fraction(19, 50)
    assert posterior == Fraction(15, 19)
    results[3] = {"probability_b": b, "posterior": posterior}
    print("Q3: Bayes' rule")
    print(f"  P(B) = 0.5 * 0.6 + 0.2 * 0.4 = {b} = {float(b):.2f}")
    print(f"  P(A | B) = 0.30 / 0.38 = {posterior} = {float(posterior):.9f}\n")

    # Q4: the complement of specificity is the false positive rate.
    prevalence = Fraction(1, 50)
    sensitivity, specificity = Fraction(19, 20), Fraction(9, 10)
    false_positive_rate = 1 - specificity
    true_positive = sensitivity * prevalence
    false_positive = false_positive_rate * (1 - prevalence)
    positive = true_positive + false_positive
    posterior = true_positive / positive
    assert false_positive_rate == Fraction(1, 10)
    assert positive == Fraction(117, 1000)
    assert posterior == Fraction(19, 117)
    assert posterior == Fraction(190, 190 + 980)
    results[4] = {
        "false_positive_rate": false_positive_rate,
        "positive_probability": positive,
        "posterior": posterior,
    }
    print("Q4: Positive test result")
    print("  P(positive | healthy) = 1 - 0.90 = 0.10")
    print("  P(positive) = 0.95 * 0.02 + 0.10 * 0.98 = 0.019 + 0.098 = 0.117")
    print(f"  P(disease | positive) = 0.019 / 0.117 = {posterior} = {float(posterior):.9f}")
    print(f"  Percentage = {100 * float(posterior):.6f}%\n")

    # Q5: calculate the population distribution's moments, then the given sample.
    scores = (85, 90, 95, 100)
    probabilities = (Fraction(3, 8), Fraction(3, 8), Fraction(1, 8), Fraction(1, 8))
    sample = (85, 90, 85, 95, 90, 85, 100, 90)
    assert sum(probabilities) == 1
    expectation = sum(score * p for score, p in zip(scores, probabilities))
    second_moment = sum(score**2 * p for score, p in zip(scores, probabilities))
    variance = second_moment - expectation**2
    centered_variance = sum((score - expectation)**2 * p for score, p in zip(scores, probabilities))
    sample_sum = Fraction(sum(sample))
    sample_mean = sample_sum / len(sample)
    difference = sample_mean - expectation
    assert expectation == 90
    assert second_moment == 8125
    assert variance == centered_variance == 25
    assert sample_sum == 720
    assert sample_mean == 90
    assert difference == 0
    results[5] = {
        "expectation": expectation,
        "second_moment": second_moment,
        "variance": variance,
        "sample_sum": sample_sum,
        "sample_mean": sample_mean,
        "sample_mean_minus_expectation": difference,
    }
    print("Q5: Expected value, variance, and sample mean")
    for score, p in zip(scores, probabilities):
        print(f"  x = {score:3d}; p = {p}; x*p = {float(score * p):.3f}; x^2*p = {float(score**2 * p):.3f}")
    print(f"  (a) E[X] = sum(x*p) = {expectation}")
    print(f"  (b) E[X^2] = {second_moment}; Var(X) = {second_moment} - {expectation}^2 = {variance}")
    print(f"      Sum of weighted squared deviations = {centered_variance}")
    print(f"  (c) Sample mean = {sample_sum}/{len(sample)} = {sample_mean}; difference from E[X] = {difference}")
    print("      Equality holds for this supplied sample, not necessarily for every sample.\n")

    # Q6: probabilities are exact fractions; only logarithms need floating point.
    probabilities = (Fraction(2, 5), Fraction(3, 10), Fraction(1, 5), Fraction(1, 10))
    assert sum(probabilities) == 1
    terms = [-float(p) * log2(float(p)) for p in probabilities]
    entropy = sum(terms)
    uniform_entropy = -4 * 0.25 * log2(0.25)
    assert isclose(entropy, 1.8464393446710154, rel_tol=0, abs_tol=1e-12)
    assert uniform_entropy == log2(4) == 2
    assert entropy < uniform_entropy
    results[6] = {"entropy_bits": entropy, "uniform_entropy_bits": uniform_entropy}
    print("Q6: Entropy in bits")
    for p, term in zip(probabilities, terms):
        print(f"  -{float(p):.1f} * log2({float(p):.1f}) = {term:.12f} bits")
    print(f"  (a) H(X) = {entropy:.12f} bits")
    print(f"  (b) H(uniform) = -4 * 0.25 * log2(0.25) = {uniform_entropy:g} bits\n")

    # Verify the machine-readable deliverable against independently computed values.
    answers_path = Path(__file__).resolve().parents[1] / "assignment" / "probability_answers.json"
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    saved = {item["question"]: item["results"] for item in answers["questions"]}
    assert set(saved) == set(results), "Saved question numbers do not match."
    for question, computed_answers in results.items():
        assert set(saved[question]) == set(computed_answers), f"Q{question}: saved answer keys do not match."
        for name, computed in computed_answers.items():
            stored = saved[question][name]
            if isinstance(computed, bool):
                assert stored is computed, f"Q{question}, {name}: boolean mismatch."
            else:
                assert isclose(float(computed), stored["value"], rel_tol=0, abs_tol=1e-12), (question, name)
                if isinstance(computed, Fraction):
                    assert Fraction(stored["exact"]) == computed, (question, name, "exact fraction")
    print("All six questions passed: exact arithmetic, numerical checks, and saved JSON consistency.")


if __name__ == "__main__":
    main()
