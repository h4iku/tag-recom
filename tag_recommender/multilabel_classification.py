import pickle
import json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
import numpy as np

from datasets import DATASET
 

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selecting appropriate feature set in the pipeline"""
    
    def __init__(self, key):
        self.key = key
    
    def fit(self, x, y=None):
        return self
    
    def transform(self, data):
        if self.key == 'title':
            return [' '.join(q.title) for q in data]
        elif self.key == 'body':
            return [' '.join(q.body) for q in data]
        elif self.key == 'codes':
            return [' '.join(q.codes) for q in data]
        else:
            return [[q.has_code, q.has_link, q.has_math] for q in data]


def multilabel_clf(train_set, test_set):
    """Multilabel Classification using LinearSVM"""
    
    train_tags = [q.tags for q in train_set]
    
    # Classes need to be binarized for the classifier
    mlb = MultiLabelBinarizer()
    train_labels = mlb.fit_transform(train_tags)
    
    classifier = Pipeline([
        ('feats', FeatureUnion([
            ('title_ngram', Pipeline([
                ('title', FeatureSelector('title')),
                ('title_tfidf', TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True))
            ])),
            ('body_ngram', Pipeline([
                ('body', FeatureSelector('body')),
                ('body_tfidf', TfidfVectorizer(sublinear_tf=True))
            ])),
            ('codes_ngram', Pipeline([
                ('codes', FeatureSelector('codes')),
                ('codes_tfidf', TfidfVectorizer(sublinear_tf=True))
            ])),
            ('meta_feats', FeatureSelector('meta'))
        ])),
        ('clf', OneVsRestClassifier(CalibratedClassifierCV(LinearSVC(), cv=3)))
    ])
    
    classifier.fit(train_set, train_labels)

    # Getting probabilities for all tags in each test questions    
    probas = classifier.predict_proba(test_set)
    tags_order = mlb.classes_.tolist()
    
    min_max_scaler = MinMaxScaler()
    results = []
    for proba in probas:
        prob = np.array([float(p)
                         for p in proba]).reshape(-1, 1)
        normalized_proba = np.concatenate(
            min_max_scaler.fit_transform(prob)
        ).tolist()
        results.append(normalized_proba)
    
    return tags_order, results


def main():
    with DATASET.train_set.open('rb') as file:
        train_set = pickle.load(file)
    with DATASET.test_set.open('rb') as file:
        test_set = pickle.load(file)

    tags_order, predicted_labels = multilabel_clf(train_set, test_set)
    
    with open(DATASET.fold_root / 'tags_order.json', 'w') as file:
        json.dump(tags_order, file)
    with open(DATASET.fold_root / 'mul_clf_proba.pickle', 'wb') as file:
        pickle.dump(predicted_labels, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()