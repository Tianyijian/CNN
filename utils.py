from sklearn.utils import shuffle

import pickle
import csv
from nltk import word_tokenize

max_length = 0


def read_MELD():
    data = {}
    emotion = ['neutral', 'joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

    def read(mode):
        x, y = [], []
        global max_length
        with open("data/MELD/" + mode + "_sent_emo.csv", 'r', encoding="utf-8", errors="ignore") as f:
            f_csv = csv.reader(f)
            for i, line in enumerate(f_csv):
                if i == 0:
                    continue
                # x.append(line[1].split())
                words = word_tokenize(line[1])
                if len(words) > max_length:
                    max_length = len(words)
                x.append(words)
                y.append(emotion.index(line[3]))

        x, y = shuffle(x, y)
        data[mode + "_x"], data[mode + "_y"] = x, y

    read("train")
    read("dev")
    read("test")
    print(data.keys())
    print("max_sent_length:%d " % max_length)
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


def test():
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    y = [0, 1, 2, 0, 1, 2]
    pred = [0, 2, 1, 0, 0, 1]
    accuracy = accuracy_score(y, pred)
    macro_p = precision_score(y, pred, average="macro")
    macro_r = recall_score(y, pred, average="macro")
    macro_f1 = f1_score(y, pred, average="macro")
    micro_p = precision_score(y, pred, average="micro")
    micro_r = recall_score(y, pred, average="micro")
    micro_f1 = f1_score(y, pred, average="micro")
    print("acc: {}\nmacro: p {}, r {}, f1: {}\nmicro: p {}, r {}, f1 {}".format(accuracy, macro_p, macro_r, macro_f1,
                                                                                micro_p, micro_r, micro_f1))


if __name__ == '__main__':
    read_MELD()
    # read_MR()
    # glove2word()
    # test()
    # import nltk
    # nltk.download()
