import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


TRAIN_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TRAIN.tsv"
TEST_URL = "https://maxime-devanne.com/datasets/ECG200/ECG200_TEST.tsv"


def load_data():
    train_df = pd.read_csv(TRAIN_URL, sep="\t")
    test_df = pd.read_csv(TEST_URL, sep="\t")

    train_df = train_df.dropna()
    test_df = test_df.dropna()

    X_train = train_df.iloc[:, 1:].values.astype(np.float32)
    y_train = train_df.iloc[:, 0].values

    X_test = test_df.iloc[:, 1:].values.astype(np.float32)
    y_test = test_df.iloc[:, 0].values

    # Conversion des labels -1 en 0 
    y_train = (y_train == 1).astype(int)
    y_test = (y_test == 1).astype(int)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def reshape_for_sequence_models(X):
    return X.reshape((X.shape[0], X.shape[1], 1))
