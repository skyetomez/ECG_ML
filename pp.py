import pandas as pd
from pandas import DataFrame
import numpy as np
import os


def parse(file: str) -> dict[str, str]:
    paths = dict()

    with open(file, "r") as f:
        for line in f.readlines():
            name_dirty, value_dirty = line.split("=")
            name, value = str(name_dirty.strip()), str(value_dirty.strip())
            paths[name] = str(value)

    return paths


def prep_dirs(
    HOME: str,
    config: str,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    os.chdir(HOME)
    paths = parse(config)

    train_mit_path = paths["train_mit"]
    test_mit_path = paths["test_mit"]
    abnormal_pb_path = paths["abnormal_pb"]
    normal_pb_path = paths["normal_pb"]

    dirs = [train_mit_path, test_mit_path, abnormal_pb_path, normal_pb_path]
    verify = [os.path.exists(dir) for dir in dirs]

    if all(verify):
        print("All dirs ready")
        os.chdir(HOME)

        tmit = pd.read_csv(train_mit_path, encoding="ascii")
        print(train_mit_path)
        tsmit = pd.read_csv(test_mit_path, encoding="ascii")
        print(test_mit_path)
        apb = pd.read_csv(abnormal_pb_path, encoding="ascii")
        print(abnormal_pb_path)
        npb = pd.read_csv(normal_pb_path, encoding="ascii")
        print(normal_pb_path)
        ans = (tmit, tsmit, apb, npb)
        return ans
    else:
        raise ValueError


def data_aug(data: pd.DataFrame) -> np.ndarray:
    data_arr = data.copy()
    data_arr = data_arr.to_numpy()
    row_dim, col_dim = data_arr.shape

    aug1 = data_arr.copy()
    shape = data_arr.shape

    # add noise
    noise = np.random.randn(row_dim, col_dim)
    aug1 += noise

    # dilate signal

    aug2 = np.zeros(shape=shape)
    aug3 = np.zeros(shape=shape)

    augs = list()

    factors = [2, 0.5]

    tmp = np.zeros(shape=shape)

    for scalar in factors:

        scaled_indices = np.arange(col_dim, dtype=int) * scalar
        scaled_indices = np.where(scaled_indices < col_dim, scaled_indices, -1)

        if scalar % 1 != 0:
            scaled_indices = np.floor(scaled_indices).astype(int)

        remainder = col_dim - len(scaled_indices)
        scaled_indices = np.pad(
            scaled_indices, (0, remainder), "constant", constant_values=-1
        )

        for row in range(row_dim):
            tmp[row, :] = np.where(
                scaled_indices != -1, data_arr[row, scaled_indices], 0
            )

        augs.append(tmp)

    aug2, aug3 = augs

    return np.vstack((data_arr, aug1, aug2, aug3))


if __name__ == "__main__":
    # HOME = os.environ["WORK"]
    # hp = os.environ["HYPERPARAMS"]

    HOME = "/home/gear/Desktop/ECG_Heartbeat/"
    train_mit_path, test_mit_path, normal_pb_path, abnormal_pb_path = prep_dirs(
        HOME, "/home/gear/Github/ECG_ML/hp.txt"
    )

    tmp = data_aug(train_mit_path)

    # factors = 0.5

    # row_dim, col_dim = train_mit_path.shape
    # data_arr = train_mit_path.to_numpy()
    # shape = train_mit_path.shape

    # tmp = np.zeros(shape=shape)

    # scaled_indices = np.arange(col_dim, dtype=int) * factors
    # scaled_indices = np.where(scaled_indices < col_dim, scaled_indices, -1)
    # scaled_indices = np.floor(scaled_indices).astype(int)
    # remainder = col_dim - len(scaled_indices)
    # scaled_indices = np.pad(
    #     scaled_indices, (0, remainder), "constant", constant_values=-1
    # )

    # for row in range(row_dim):
    #     tmp[row, :] = np.where(scaled_indices != -1, data_arr[row, scaled_indices], 0)
