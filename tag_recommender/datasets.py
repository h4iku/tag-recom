from pathlib import Path


class Data:
    
    __slots__ = ['name', 'root', 'data', 'fold',
                 'fold_root', 'train_set', 'test_set']
    
    total_folds = 10
    
    # Dataset root directory
    _DATASET_ROOT = Path(__file__).parent / '../data'
    
    def __init__(self, name):
        self.name = name
        self.root = Data._DATASET_ROOT / name
        self.data = self.root / f'{name}.csv'

        # TODO: Also add pickled things here
        self.fold = None
        self.fold_root = None
        self.train_set = None
        self.test_set = None
        
    def set_fold(self, fold):
        self.fold = fold
        self.fold_root = self.root / str(self.fold)
        self.train_set = self.fold_root / 'train_set.pickle'
        self.test_set = self.fold_root / 'test_set.pickle'


# Software information sites
datasets_names = [
    'apple',
    'askubuntu',
    'codereview',
    'dba',
    'serverfault',
    'softwareengineering',
    'stackoverflow',
    'stats',
    'superuser',
    'tex',
    'wordpress',
]

datasets = {name : Data(name) for name in datasets_names}

# Current dataset in use. (change this variable to change the dataset)
DATASET = datasets['tex']

if __name__ == '__main__':
    DATASET.set_fold(0)
    print(DATASET.name, DATASET.root, DATASET.train_set, DATASET.test_set, DATASET.total_folds)
