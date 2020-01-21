import pickle
from collections import defaultdict
import numpy as np

from datasets import DATASET
import preprocessing
import generate_folds
import multilabel_classification
import textual_similarity
import evaluation

# Set these booleans to preprocess, classify, or compute similarities.
preprocess = True
gen_folds = True
classify = True
similarity = True

cuts = (5, 10)  # Final rank cuts

precisions = defaultdict(list)
recalls = defaultdict(list)
f1_scores = defaultdict(list)

if preprocess:
    print('Preprocessing')
    preprocessing.main()

if gen_folds:
    print('Generating folds')
    generate_folds.main()

for fold in range(DATASET.total_folds):
    
    DATASET.set_fold(fold)
    
    if classify:
        print(f'Multi-label classification fold {fold}')
        multilabel_classification.main()
    
    if similarity:
        print(f'Textual similarity fold {fold}')
        textual_similarity.main()
    
    print(f'Evaluating fold {fold}')
    with DATASET.train_set.open('rb') as file:
        train_set = pickle.load(file)
    param_estimate_data = evaluation.prepare_estimate_date(train_set)
    
    for at in cuts:
        evaluation.at = at
        pr, re, f1 = evaluation.recommend(param_estimate_data)
        precisions[at].append(pr)
        recalls[at].append(re)
        f1_scores[at].append(f1)

for at in cuts:
    print()
    
    print(f'@{at}')
    print('p:', precisions[at])
    print('mean:', np.mean(precisions[at]))
    print('std:', np.std(precisions[at], ddof=1))
    
    print('r:', recalls[at])
    print('mean:', np.mean(recalls[at]))
    print('std:', np.std(recalls[at], ddof=1))
    
    print('f:', f1_scores[at])
    print('mean:', np.mean(f1_scores[at]))
    print('std:', np.std(f1_scores[at], ddof=1))
