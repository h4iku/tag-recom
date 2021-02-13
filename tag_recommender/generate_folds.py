import pickle
import random

from datasets import DATASET


def fold_generator(preprocessed_data, test_size):

    for fold in range(DATASET.total_folds):

        DATASET.set_fold(fold)
        temp_qs = preprocessed_data[:]

        # Randomly select objects of the whole data as the test set
        # and the remaining data is the train set
        test_set = [temp_qs.pop(random.randrange(len(temp_qs)))
                    for _ in range(test_size)]

        # Creating the fold directory
        DATASET.fold_root.mkdir()

        # Writing the trianing set
        with DATASET.train_set.open('wb') as train_file:
            pickle.dump(temp_qs, train_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Writing the generated test set
        with DATASET.test_set.open('wb') as test_file:
            pickle.dump(test_set, test_file, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    test_size = 10000
    with open(DATASET.root / 'preprocessed_data.pickle', 'rb') as file:
        preprocessed_data = pickle.load(file)
    fold_generator(preprocessed_data, test_size)


if __name__ == '__main__':
    main()
