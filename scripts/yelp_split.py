import argparse
import pathlib

import pandas as pd

from sklearn.model_selection import train_test_split


def main(args):
    path = "data/yelp/"

    # Load
    print("loading...")
    df = pd.read_json(path + "yelp_academic_dataset_review.json", lines=True)

    # Get only relevant features
    df = df.loc[:, ["text", "stars"]]

    # Perform 80:10:10 split
    train, other = train_test_split(df, test_size=0.2, stratify=df["stars"])
    val, test = train_test_split(other, test_size=0.5, stratify=other["stars"])

    # Save results and print to confirm
    print("compressing...")
    pathlib.Path(path + "reviews/").mkdir(exist_ok=True)
    for name, df in zip(("train", "val", "test"), (train, val, test)):
        # df.to_json(path + f"reviews/{name}.json")
        df.to_pickle(path + f"reviews/{name}.xz", compression="xz")

        print(f"\n--- {name} ---")
        print("Shape:", df.shape)
        print(df.stars.value_counts())

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    main(args)
