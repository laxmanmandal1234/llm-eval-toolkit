import math

import numpy as np
import pytest

from llm_eval import (bootstrap_ci, brier_score, corpus_bleu, exact_match,
                      expected_calibration_error, normalize, paired_bootstrap_test,
                      reliability_bins, rouge_l, self_consistency, sentence_bleu, token_f1)


def test_normalize():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"


def test_exact_match_uses_normalization_and_multiple_refs():
    assert exact_match("the Paris.", ["London", "paris"]) == 1.0
    assert exact_match("Paris, France", ["Paris"]) == 0.0


def test_token_f1_known_value():
    # pred: {new, york, city}; ref: {new, york} -> P=2/3, R=1 -> F1=0.8
    assert token_f1("New York City", ["New York"]) == pytest.approx(0.8)
    assert token_f1("", [""]) == 1.0
    assert token_f1("cat", ["dog"]) == 0.0


def test_bleu_perfect_and_zero():
    s = "the quick brown fox jumps over the lazy dog"
    assert corpus_bleu([s], [[s]]) == pytest.approx(1.0)
    assert corpus_bleu(["completely different words here"], [[s]]) == 0.0


def test_bleu_brevity_penalty():
    ref = "a b c d e f g h"
    pred = "a b c d"           # all n-grams match, but half the length
    assert corpus_bleu([pred], [[ref]]) == pytest.approx(math.exp(1 - 8 / 4))


def test_bleu_clipping():
    # "the the the the" vs "the cat": unigram precision clipped to 1/4
    assert corpus_bleu(["the the the the"], [["the cat"]], max_n=1) == pytest.approx(0.25)


def test_sentence_bleu_is_smoothed():
    assert 0 < sentence_bleu("the cat sat", ["the cat sat on the mat"]) < 1


def test_rouge_l_known_value():
    # Normalisation drops "the": [police, killed, gunman] vs [police, kill, gunman]
    # LCS = 2 (police, gunman) -> P = R = 2/3 -> F = 2/3
    f = rouge_l("police killed the gunman", ["police kill the gunman"])
    assert f == pytest.approx(2 / 3)


def test_ece_perfectly_calibrated_is_zero():
    conf = [0.8] * 10
    correct = [True] * 8 + [False] * 2
    assert expected_calibration_error(conf, correct) == pytest.approx(0.0)


def test_ece_overconfident():
    conf = [0.9] * 10
    correct = [True] * 5 + [False] * 5
    assert expected_calibration_error(conf, correct) == pytest.approx(0.4)


def test_reliability_bins_and_bounds():
    bins = reliability_bins([0.0, 0.05, 0.95, 1.0], [False, False, True, True], n_bins=10)
    assert sum(b.count for b in bins) == 4
    with pytest.raises(ValueError):
        reliability_bins([1.2], [True])


def test_brier():
    assert brier_score([1.0, 0.0], [True, False]) == 0.0
    assert brier_score([0.5], [True]) == 0.25


def test_self_consistency():
    answer, share = self_consistency(["Paris", "paris.", "Lyon", "The Paris"])
    assert normalize(answer) == "paris" and share == 0.75


def test_bootstrap_ci_contains_mean():
    scores = np.random.default_rng(0).random(200)
    ci = bootstrap_ci(scores)
    assert ci.low < ci.mean < ci.high


def test_paired_bootstrap():
    rng = np.random.default_rng(1)
    b = rng.random(300)
    assert paired_bootstrap_test(b + 0.1, b) < 0.01      # clearly better
    assert paired_bootstrap_test(b, b + 0.1) > 0.99      # clearly worse
    with pytest.raises(ValueError):
        paired_bootstrap_test([1, 2], [1])
