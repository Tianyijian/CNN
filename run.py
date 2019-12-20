from model import CNN
import utils
import os

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

tb_writer = SummaryWriter(log_dir="logs")

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy


def train(data, params, global_step):
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

    max_dev_res = {"weighted_f1": 0}
    max_test_res = {}
    max_train_res = {}
    max_epoch = 0
    best_cnt = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in range(0, len(data["train_x"]), params["BATCH_SIZE"]):
            global_step += 1
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
            print("global_step: %d, epoch: %d, loss: %f" % (global_step, e + 1, loss.item()))
            tb_writer.add_scalar('loss', loss.item(), global_step)

        print("epoch:", e + 1)
        train_res = test(data, model, params, "train", global_step)
        dev_res = test(data, model, params, "dev", global_step)
        test_res = test(data, model, params, "test", global_step)
        # print("epoch:", e + 1, "/ train_res:", train_res, "/ dev_res:", dev_res, "/ test_res:", test_res)

        if dev_res["weighted_f1"] > max_dev_res["weighted_f1"]:
            max_train_res = train_res
            max_dev_res = dev_res
            max_test_res = test_res
            max_epoch = e + 1
            print("New best model!")
            if params["SAVE_MODEL"]:
                best_model = copy.deepcopy(model)

        if params["EARLY_STOPPING"] and dev_res["weighted_f1"] <= max_dev_res["weighted_f1"]:
            best_cnt += 1
            if best_cnt >= 3:
                print("early stopping by dev_weighted_f1!")
                break
        else:
            best_cnt = 0

    print("BEST MODEL epoch: " + str(max_epoch))
    print("train\t" + " ".join(["%s: %.4f" % (k, max_train_res[k]) for k in max_train_res]))
    print("dev\t" + " ".join(["%s: %.4f" % (k, max_dev_res[k]) for k in max_dev_res]))
    print("test\t" + " ".join(["%s: %.4f" % (k, max_test_res[k]) for k in max_test_res]))
    if params["SAVE_MODEL"]:
        utils.save_model(best_model, params)
    return global_step, max_train_res, max_dev_res, max_test_res, max_epoch


def test(data, model, params, mode, global_step):
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
    # acc = sum([1 if p == y else 0 for p, y in zip(pred, y)]) / len(pred)
    res = {}
    res["acc"] = accuracy_score(y, pred)
    res["macro_p"] = precision_score(y, pred, average="macro")
    res["macro_r"] = recall_score(y, pred, average="macro")
    res["macro_f1"] = f1_score(y, pred, average="macro")
    res["micro_p"] = precision_score(y, pred, average="micro")
    res["micro_r"] = recall_score(y, pred, average="micro")
    res["micro_f1"] = f1_score(y, pred, average="micro")
    res["weighted_f1"] = f1_score(y, pred, average="weighted")
    print(
        "{}\tacc: {:.4f}\tmacro: p {:.4f}, r {:.4f}, f1: {:.4f}\tmicro: p {:.4f}, r {:.4f}, f1 {:.4f}\tweighted_f1:{:.4f}".format(
            mode, res["acc"], res["macro_p"], res["macro_r"], res["macro_f1"], res["micro_p"], res["micro_r"],
            res["micro_f1"], res["weighted_f1"]))
    for k in res:
        tb_writer.add_scalar(mode + "_" + k, res[k], global_step)
    return res


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="non-static",
                        help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="MELD", help="available datasets: MR, TREC, MELD")
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=200, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=0.1, type=float, help="learning rate")
    parser.add_argument("--gpu", default=9, type=int, help="the number of gpu to be used")
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
        "BATCH_SIZE": 256,
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
        v = ["acc", "macro_f1", "micro_f1", "weighted_f1"]
        iter = 5
        test_w_f1 = 0.0
        best_epoch = []
        global_step = 0
        with open("res.csv", "w", encoding="utf-8") as f:
            f.write("id," + ",".join(
                ["train_%s" % s for s in v] + ["dev_%s" % s for s in v] + ["test_%s" % s for s in v]) + "\n")
            # f.write("id,acc,macro_f1,micro_f1,weighted_f1\n")
            for i in range(1, 1 + iter):
                print("=" * 10 + "ROUND " + str(i) + "=" * 10)
                global_step, max_train_res, max_dev_res, max_test_res, max_epoch = train(data, params, global_step)
                test_w_f1 += max_test_res["weighted_f1"]
                best_epoch.append(max_epoch)
                f.write(str(i) + "," + ",".join(["%f" % max_train_res[k] for k in max_train_res if k in v]))
                f.write("," + ",".join(["%f" % max_dev_res[k] for k in max_dev_res if k in v]))
                f.write("," + ",".join(["%f" % max_test_res[k] for k in max_test_res if k in v]) + "\n")
                # f.write(str(i) + "," + ",".join(["%f" % max_test_res[k] for k in max_test_res if k in v]) + "\n")
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
        test_w_f1 = test_w_f1 / iter
        print("best epoch: %s, avg test weighted f1: %f" % (str(best_epoch), test_w_f1))
        tb_writer.add_scalar('avg_test_w_f1', test_w_f1, global_step)
        tb_writer.close()
    else:
        model = utils.load_model(params).cuda(params["GPU"])

        test_acc = test(data, model, params)
        print("test acc:", test_acc)


if __name__ == "__main__":
    main()
