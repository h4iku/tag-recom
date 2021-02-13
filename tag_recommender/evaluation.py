import itertools
import json
import operator
import pickle
import random
from pathlib import Path

import numpy as np
from scipy import optimize

import multilabel_classification
import textual_similarity
from datasets import DATASET

# @ indicator for p, r, and f1
at = None


def combine_rank_scores(coeffs, *rank_scores):
    """Combining the rank score of different algorithms"""

    final_score = []
    for scores in zip(*rank_scores):
        combined_score = coeffs @ np.array(scores)
        final_score.append(combined_score.tolist())

    return final_score


def cost(coeffs, tags_order, test_set, *rank_scores):

    final_scores = combine_rank_scores(coeffs, *rank_scores)

    # List for recall value of each iteration
    recall = []

    for i, test_q in enumerate(test_set):
        tag_ranks, _ = zip(*sorted(zip(tags_order, final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))

        # Calculating recall
        tp = set(test_q.tags) & set(tag_ranks[:at])
        recall.append(len(tp) / len(test_q.tags))

    return -1 * np.mean(recall)


def evaluate(coeffs, tags_order, test_set, *rank_scores):

    final_scores = combine_rank_scores(coeffs, *rank_scores)

    # Lists for p, r and f1 of each iteration
    precision = []
    recall = []
    f_measure = []

    for i, test_q in enumerate(test_set):
        tag_ranks, _ = zip(*sorted(zip(tags_order, final_scores[i]),
                                   key=operator.itemgetter(1), reverse=True))

        # Calculating precision, recall and f-measure
        tp = set(test_q.tags) & set(tag_ranks[:at])
        if not tp:
            precision.append(0)
        else:
            precision.append(len(tp) / len(tag_ranks[:at]))
        recall.append(len(tp) / len(test_q.tags))
        if not (precision[i] + recall[i]):
            f_measure.append(0)
        else:
            f_measure.append(
                2 * (precision[i] * recall[i]) / (precision[i] + recall[i]))

    return (np.mean(precision), np.mean(recall), np.mean(f_measure))
    # return (precision, recall, f_measure)


def estiamte_params(tags_order, test_set, *rank_scores):
    """Estimating linear combination parameters"""

    res = optimize.differential_evolution(
        cost, bounds=[(0, 1)] * len(rank_scores),
        args=(tags_order, test_set, *rank_scores)
    )

    return res.x.tolist()


def prepare_estimate_date(train_set):
    """Prepares data for parameter estimation"""

    test_size = int(0.2 * len(train_set))
    test_set = [train_set.pop(random.randrange(len(train_set)))
                for _ in range(test_size)]
    tags_order, multilabel_scores = multilabel_classification.multilabel_clf(
        train_set, test_set)
    sm = textual_similarity.Similarity(train_set)
    similarity_scores = sm.find_similars(test_set)

    return (tags_order, test_set, multilabel_scores, similarity_scores)


def recommend(param_estimate_data):

    with DATASET.test_set.open('rb') as file:
        test_set = pickle.load(file)

    with open(DATASET.fold_root / 'tags_order.json') as file:
        tags_order = json.load(file)
    with open(DATASET.fold_root / 'mul_clf_proba.pickle', 'rb') as file:
        multilabel_scores = pickle.load(file)
    with open(DATASET.fold_root / 'title_simis.pickle', 'rb') as file:
        similarity_scores = pickle.load(file)

    params = estiamte_params(*param_estimate_data)
    results = evaluate(params, tags_order, test_set,
                       multilabel_scores, similarity_scores)

    return results
