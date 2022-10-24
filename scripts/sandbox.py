import pandas as pd


def main():
    # decompress_reviews()
    # read_json()

    pass


def decompress_reviews():
    path = "data/yelp/"

    train = pd.read_pickle("data/yelp/reviews/train.xz")
    test = pd.read_pickle("data/yelp/reviews/test.xz")
    dev = pd.read_pickle("data/yelp/reviews/dev.xz")

    for name, df in zip(("train", "dev", "test"), (train, dev, test)):
        df.to_json(path + f"reviews/{name}.json")


def read_json():
    df = pd.read_json("data/yelp/reviews/dev.json")
    print(df)


if __name__ == "__main__":
    main()
