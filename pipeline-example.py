'''
Brief example of how to load custom data structures into tensorflow
'''
DATASET_PATH='/data/mnist'
TEST_LABELS_FILE='test-labels.csv'
TRAIN_LABELS_FILE='train-labels.csv'
import os


def encode_label(label):
    return int(label)

def read_label_file(source_file):
    with open(source_file, "r") as f:
        filepaths, labels = [], []
        for line in f:
            filepath, label = line.split(",")
            filepaths.append(filepath)
            labels.append(label)

    return filepaths, labels

if __name__ == '__main__':
    train_filepaths, train_labels = read_label_file(os.path.join(DATASET_PATH, TRAIN_LABELS_FILE))
    test_filepaths, test_labels = read_label_file(os.path.join(DATASET_PATH, TEST_LABELS_FILE))

    # full path
    train_filepaths = [os.path.join(DATASET_PATH, file_path) 
        for file_path in train_filepaths]

    test_filepaths =  [os.path.join(DATASET_PATH, file_path) 
        for file_path in test_filepaths]

    # join and take sampe
    all_filepaths = (train_filepaths + test_filepaths)[:20]
    all_labels = (train_labels + test_labels)[:20]

    print(all_labels)
    print(all_filepaths)
