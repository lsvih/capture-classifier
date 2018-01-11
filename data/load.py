import pickle


def loader() -> dict:
    with open("data/dataset", "rb") as f:
        data = pickle.load(f, encoding="bytes")

    return data


if __name__ == "__main__":
    print(loader())
