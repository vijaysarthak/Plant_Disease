"""
conformal.py — turns the model's scores into a SAFE decision.

The thresholds and the temperature were LEARNED in the Colab notebook and saved in
saved_model/model_info.json. This file only USES them.

Words used here:
  logits        : the raw scores the network outputs, one per class
  temperature T : one number that makes the probabilities honest
  probability   : softmax(logits / T) — numbers between 0 and 1 that add up to 1
  group probability: probability of a GROUP = sum of the probabilities of its classes
  score         : s(x, g) = 1 - group probability.  Small score = group is likely.
  threshold     : one cut-off per group
  prediction set: all groups whose score is <= their threshold

Decision rule:
  * the set has exactly 1 group   -> ADVISE that group's treatment
  * the set has 2 or more groups  -> REFER the farmer to an expert (KVK)
  * the set is empty, or the photo looks unfamiliar (high energy) -> UNKNOWN

These functions are the same as in Step 14 of the Colab notebook.
"""
import numpy as np


def softmax(logits, temperature=1.0):
    """logits (photos x classes) -> probabilities (photos x classes), each row adds up to 1."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)    # subtract the biggest value: avoids overflow
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def group_probabilities(probs, class_groups, num_groups):
    """probs (photos x 38 classes) -> group probabilities (photos x 6 groups)."""
    class_groups = np.asarray(class_groups)
    result = np.zeros((len(probs), num_groups))
    for g in range(num_groups):
        result[:, g] = probs[:, class_groups == g].sum(axis=1)
    return result


def group_scores(probs, class_groups, num_groups):
    """s(x, g) = 1 - group probability.  Small score = this group is likely."""
    return 1.0 - group_probabilities(probs, class_groups, num_groups)


def prediction_sets(probs, class_groups, num_groups, thresholds):
    """True/False table (photos x groups): is group g in the prediction set of photo i?"""
    scores = group_scores(probs, class_groups, num_groups)
    return scores <= np.asarray(thresholds)


def energy_score(logits, temperature=1.0):
    """Energy = -T * log(sum(exp(logits / T))).
    LOW for photos that look like the training photos, HIGH for unfamiliar photos."""
    scaled = np.asarray(logits, dtype=np.float64) / temperature
    biggest = scaled.max(axis=1, keepdims=True)
    log_sum_exp = biggest[:, 0] + np.log(np.exp(scaled - biggest).sum(axis=1))
    return -temperature * log_sum_exp


def decide(groups_in_set, is_unfamiliar=False):
    """
    groups_in_set: list of group numbers in the prediction set of ONE photo.
    Returns ("advise", group) or ("refer", None) or ("unknown", None).
    """
    if is_unfamiliar or len(groups_in_set) == 0:
        return "unknown", None
    if len(groups_in_set) == 1:
        return "advise", groups_in_set[0]
    return "refer", None
