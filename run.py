from model import CNN
import utils
import os

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy


def train(data, params):
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        if os.path.exists("glove.txt"):
            print("load glove.txt")
            word_vectors = KeyedVectors.load_word2vec_format("glove.txt")
        else:
            word_vectors = KeyedVectors.load_word2vec_format(params["W2V_PATH"])

        words = {}
        w_in_w2v = 0
        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
                words[word] = word_vectors.word_vec(word)
                w_in_w2v += 1
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # vocab_num:13802, in w2v: 6506, ratio:0.4713809592812636
        print("vocab_num:{}, in w2v: {}, ratio:{}".format(len(data["vocab"]), w_in_w2v,
                                                          float(w_in_w2v) / len(data["vocab"])))

        if not os.path.exists("glove.txt"):
            print("write glove vector to glove.txt")
            with open("glove.txt", "w", encoding="utf-8") as f:
                f.write(str(len(words)) + " 300\n")
                for word in words:
                    f.write("{} {}\n".format(word, " ".join("%s" % v for v in words[word])))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params).cuda(params["GPU"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    max_dev_acc = 0
    max_test_acc = 0
    max_train_acc = 0
    best_cnt = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = [[data["word_to_idx"][w] for w in sent] +
                       [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                       for sent in data["train_x"][i:i + batch_range]]
            batch_y = [data["classes"].index(c) for c in data["train_y"][i:i + batch_range]]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_acc = test(data, model, params, mode="dev")
        test_acc = test(data, model, params, mode="test")
        train_acc = test(data, model, params, mode="train")
        print("epoch:", e + 1, "/ train_acc:", train_acc, "/ dev_acc:", dev_acc, "/ test_acc:", test_acc)

        if params["EARLY_STOPPING"] and dev_acc <= max_dev_acc:
            best_cnt += 1
            if best_cnt >= 3:
                print("early stopping by dev_acc!")
                break
        else:
            best_cnt = 0
            max_train_acc = train_acc
            max_dev_acc = dev_acc
            max_test_acc = test_acc
            print("New best model!")
            best_model = copy.deepcopy(model)

    print("max train acc:", max_train_acc, "max dev acc:", max_dev_acc, "test acc:", max_test_acc)
    return best_model


def test(data, model, params, mode="test"):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]
    elif mode == "train":
        x, y = data["train_x"], data["train_y"]

    x = [[data["word_to_idx"][w] if w in data["vocab"] else params["VOCAB_SIZE"] for w in sent] +
         [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
         for sent in x]

    x = Variable(torch.LongTensor(x)).cuda(params["GPU"])
    y = [data["classes"].index(c) for c in y]

    pred = np.argmax(model(x).cpu().data.numpy(), axis=1)
    acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)

    return acc


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="non-static",
                        help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="MELD", help="available datasets: MR, TREC, MELD")
    parser.add_argument("--save_model", default=True, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=True, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--gpu", default=0, type=int, help="the number of gpu to be used")
    parser.add_argument("--w2v_path", default="/users5/yjtian/Downloads/glove.840B.300d.w2v.txt",
                        help="word2vec file path")

    options = parser.parse_args()
    data = getattr(utils, f"read_{options.dataset}")()

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] + data["test_x"] for w in sent])))
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [50, 50, 50],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu,
        "W2V_PATH": options.w2v_path
    }

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("MAX_SENT_LEN:", params["MAX_SENT_LEN"])
    print("CLASS_SIZE:", params["CLASS_SIZE"])
    print("FILTERS:", params["FILTERS"])
    print("FILTER_NUM:", params["FILTER_NUM"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        if params["SAVE_MODEL"]:
            utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        model = utils.load_model(params).cuda(params["GPU"])

        test_acc = test(data, model, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()
