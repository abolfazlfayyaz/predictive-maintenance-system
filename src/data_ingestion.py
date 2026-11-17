from pathlib import Path
import pandas as pd

COLUMN_NAMES = [
    "unit_number",
    "time_in_cycle",
    "op_settings_1",
    "op_settings_2",
    "op_settings_3",
]


SENSOR_COLUMNS = [
    f"sensor_{i}" for i in range(1, 22)
]
COLUMN_NAMES.extend(SENSOR_COLUMNS)

def load_sensor_data(file_path):

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None
    )

    df = df.dropna(axis=1, how="all")

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"Expected 26 columns, got {df.shape[1]}"
        )

    df.columns = COLUMN_NAMES

    return df


def load_fd001(data_dir="data/raw"):

    data_dir = Path(data_dir)

    train_path = data_dir / "train_FD001.txt"
    test_path = data_dir / "test_FD001.txt"
    rul_path = data_dir / "RUL_FD001.txt"

    train_df = load_sensor_data(train_path)

    test_df = load_sensor_data(test_path)

    rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    if len(rul_df) != test_df["unit_number"].nunique():
        raise ValueError(
            "RUL labels do not match the number of test engines."
        )

    return train_df, test_df, rul_df



if __name__ == "__main__":

    train, test, rul = load_fd001()

    print("TRAIN DATA")
    print(train.head())

    print("\nTEST DATA")
    print(test.head())

    print("\nRUL DATA")
    print(rul.head())

    print("\nDATASET SHAPES")

    print("Train:", train.shape)
    print("Test:", test.shape)
    print("RUL:", rul.shape)

    print("\nUNIQUE ENGINES")

    print("Train engines:",
          train["unit_number"].nunique())

    print("Test engines:",
          test["unit_number"].nunique())