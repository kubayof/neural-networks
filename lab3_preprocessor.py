import numpy as np


def load():
    word2index = {}
    max_index = 0
    resulting_entries = []
    with open('lab3_data/data.txt', 'r') as f:
        for line in f.readlines():
            label = line[:1]
            words = line[2:].split(" ")
            word_indices = []
            for word in words:
                if word in word2index:
                    index = word2index[word]
                else:
                    index = max_index
                    max_index += 1
                    word2index[word] = max_index
                word_indices.append(index)
            resulting_entries.append([int(label), np.array(word_indices)])
    return np.array(resulting_entries)


def shuffle(resulting_entries_np):
    np.random.shuffle(resulting_entries_np)
    return resulting_entries_np


def split(resulting_entries_np):
    rate = 5
    x = []
    y = []
    for entry in resulting_entries_np:
        y.append(entry[0])
        x.append(entry[1])

    test_count = len(x) // rate
    x_train = np.array(x[test_count:])
    y_train = np.array(y[test_count:])
    x_test = np.array(x[:test_count])
    y_test = np.array(y[:test_count])
    return x_train, y_train, x_test, y_test


entries_np = load()
entries_np_shuffled = shuffle(entries_np)
x_train_np, y_train_np, x_test_np, y_test_np = split(entries_np_shuffled)
type(x_train_np)

all_np = np.array([x_train_np, y_train_np, x_test_np, y_test_np])

np.save("lab3_data/data.npy", all_np)
