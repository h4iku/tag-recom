import pickle
import json
from collections import Counter
from itertools import chain

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import jenkspy

from datasets import DATASET


class Similarity(object):
    
    __slots__ = ['train_set', 'train_str']
    
    def __init__(self, train_set):
        self.train_set = train_set
        self.train_str = [' '.join(q.title) for q in train_set]
    
    
    def calculate_similarity(self, tfidf_matrix, test_tfidf):
        """Calculatnig cosine similarity between test and train questions"""
        
        with open(DATASET.fold_root / 'tags_order.json') as file:
            tags_order = json.load(file)
        
        min_max_scaler = MinMaxScaler()
        
        n_clus = 2
        simis = []
        for test_q in test_tfidf:
            s = cosine_similarity(tfidf_matrix, test_q)
            
            # Sorting and getting indices of sorted similarities
            simi = s.transpose()[0]
            simi_values = np.sort(simi)[::-1][:200]
            simi_indices = simi.argsort()[::-1]

            breaks = jenkspy.jenks_breaks(simi_values, n_clus)
            simi_count = len(simi_values[breaks[-2] <= simi_values])

            q_tags = [self.train_set[i].tags for i in simi_indices][:simi_count]
            
            tags_votes = Counter(chain(*q_tags))
            all_count = sum(tags_votes.values())
            tags_likelihood = [tags_votes.get(tag, 0) / all_count for tag in tags_order]
            
            lh = np.array([float(x)
                           for x in tags_likelihood]).reshape(-1, 1)
            normalized_lh = np.concatenate(
                min_max_scaler.fit_transform(lh)
            ).tolist()
            
            simis.append(normalized_lh)
            
        return simis
        
    
    
    def find_similars(self, test_set):
        """Calculating tf-idf vectors for test and train sets
        to find similar questions for each test question.
        """
        
        tfidf = TfidfVectorizer(lowercase=False, sublinear_tf=True)
        tfidf_matrix = tfidf.fit_transform(self.train_str)
        
        # Calling only transform on test so that idf calculated on train data
        test_str = [' '.join(q.title) for q in test_set]
        test_tfidf = tfidf.transform(test_str)
        
        simis = self.calculate_similarity(tfidf_matrix, test_tfidf)
        return simis
         


def main():
    # Unpickle preprocessed data
    with DATASET.train_set.open('rb') as file:
        train_set = pickle.load(file)
    with DATASET.test_set.open('rb') as file:
        test_set = pickle.load(file)
    
    sm = Similarity(train_set)
    simis = sm.find_similars(test_set)
    
    with open(DATASET.fold_root / 'title_simis.pickle', 'wb') as file:
        pickle.dump(simis, file, protocol=pickle.HIGHEST_PROTOCOL)
    
    
if __name__ == '__main__':
    main()