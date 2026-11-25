"""
Tests for src/conformal.py — run them from the project folder with:   pytest -q
Each test checks one small promise of the code with numbers we can work out by hand.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import conformal


def test_softmax_rows_add_up_to_one():
    probs = conformal.softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[1], 1 / 3)             # equal logits -> equal probabilities


def test_higher_temperature_makes_probabilities_flatter():
    logits = np.array([[1.0, 3.0]])
    assert conformal.softmax(logits, 2.0).max() < conformal.softmax(logits, 1.0).max()


def test_group_probabilities_add_class_probabilities():
    probs = np.array([[0.5, 0.3, 0.2]])
    class_groups = [0, 1, 1]                         # class 0 -> group 0, classes 1 and 2 -> group 1
    assert np.allclose(conformal.group_probabilities(probs, class_groups, 2), [[0.5, 0.5]])


def test_prediction_set_uses_the_thresholds():
    probs = np.array([[0.7, 0.2, 0.1]])              # group scores: 0.3, 0.8, 0.9
    sets = conformal.prediction_sets(probs, [0, 1, 2], 3, thresholds=[0.5, 0.85, 0.5])
    assert sets.tolist() == [[True, True, False]]


def test_energy_is_lower_for_confident_logits():
    confident = np.array([[10.0, 0.0, 0.0]])
    unsure = np.array([[0.1, 0.0, 0.0]])
    assert conformal.energy_score(confident) < conformal.energy_score(unsure)


def test_decision_rule():
    assert conformal.decide([1]) == ("advise", 1)
    assert conformal.decide([1, 3]) == ("refer", None)
    assert conformal.decide([]) == ("unknown", None)
    assert conformal.decide([1], is_unfamiliar=True) == ("unknown", None)
