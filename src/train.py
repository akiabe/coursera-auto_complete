import config
import dataset
import model

def train():
    # load row dataset
    with open(config.RAW_DATA_DIR, "r") as f:
        data = f.read()

    # get tokenized data
    tokenized_data = dataset.get_tokenized_data(data)

    # split to train and test data
    train_data, test_data = dataset.split_to_train_and_test_sets(tokenized_data)

    # get vocabulary and processed train and test data
    minimum_freq = 2
    train_data_processed, test_data_processed, vocabulary = dataset.preprocess_data(train_data,
                                                                                    test_data,
                                                                                    minimum_freq)
    print(type(vocabulary))


if __name__ == "__main__":
    train()