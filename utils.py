from sklearn.utils import shuffle

import pickle
import csv


def read_MELD():
    data = {}
    emotion = ['neutral', 'joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    def read(mode):
        x, y = [], []
        with open("data/MELD/" + mode + "_sent_emo.csv", 'r', encoding="utf-8", errors="ignore") as f:
            f_csv = csv.reader(f)
            for i, line in enumerate(f_csv):
                if i == 0:
                    continue
                x.append(line[1].split())
                y.append(emotion.index(line[3]))

        x, y = shuffle(x, y)
        data[mode + "_x"], data[mode + "_y"] = x, y

    read("train")
    read("dev")
    read("test")
    print(data.keys())
    for mode in ["train", "dev", "test"]:
        print("\n-------" + mode + "--------")
        print("%d %d" % (len(data[mode + "_x"]), len(data[mode + "_y"])))
        for i in range(2):
            print(str(data[mode + "_x"][i]) + "   " + str(data[mode + "_y"][i]))
    return data


def read_TREC():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")

    return data


def read_MR():
    data = {}
    x, y = [], []

    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def save_model(model, params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")


def load_model(params):
    path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()


def glove2word():
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec("/users5/yjtian/Downloads/glove.840B.300d.txt", "/users5/yjtian/Downloads/glove.840B.300d.w2v.txt")


if __name__ == '__main__':
    read_MELD()
    # read_MR()
    # glove2word()
