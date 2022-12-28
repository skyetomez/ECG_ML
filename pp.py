import pandas as pd
from pandas import DataFrame
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


if __name__ == "__main__":
    # HOME = os.environ["WORK"]
    # hp = os.environ["HYPERPARAMS"]

    HOME = "/home/gear/Desktop/ECG_Heartbeat/"
    train_mit_path, test_mit_path, normal_pb_path, abnormal_pb_path = prep_dirs(
        HOME, "/home/gear/Github/ECG_ML/hp.txt"
    )

    # train_mit = pd.read_csv(train_mit_path)
    # test_mit = pd.read_csv(test_mit_path)
    # abnormal_pb = pd.read_csv(abnormal_pb_path)
    # normal_pb = pd.read_csv(normal_pb_path)

    # mit_data = pd.concat([train_mit, test_mit], axis=0)
