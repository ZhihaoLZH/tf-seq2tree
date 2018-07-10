import random

parent = "./data/seq2tree_geoqueries"

train_src_name = "src_train.txt"
train_tgt_name = "tgt_train.txt"

train_src_f = open(parent + "/tmp/" + train_src_name, "r", encoding="utf-8")
train_tgt_f = open(parent + "/tmp/" + train_tgt_name, "r", encoding="utf-8")

train_src = train_src_f.readlines()
train_tgt = train_tgt_f.readlines()

train_src_f.close()
train_tgt_f.close()

l = min(len(train_src), len(train_tgt))
DEV_NUM = int(0.1 * l)

dev_idxs = random.sample(range(l), DEV_NUM)

new_tr_src = []
new_tr_tgt = []

new_dev_src = []
new_dev_tgt = []

for i in range(len(train_src)):

    if i in dev_idxs:
        # belong to dev
        new_dev_src.append(train_src[i])
        new_dev_tgt.append(train_tgt[i])
    else:
        # belong to train
        new_tr_src.append(train_src[i])
        new_tr_tgt.append(train_tgt[i])

train_src_f = open(parent + "/tmp2/" + train_src_name, "w", encoding="utf-8")
train_tgt_f = open(parent + "/tmp2/" + train_tgt_name, "w", encoding="utf-8")

dev_src_f = open(parent + "/tmp2/src_dev.txt", "w", encoding="utf-8")
dev_tgt_f = open(parent + "/tmp2/tgt_dev.txt", "w", encoding="utf-8")

fs = [train_src_f, train_tgt_f, dev_src_f, dev_tgt_f]
ds = [new_tr_src, new_tr_tgt, new_dev_src, new_dev_tgt]

for f, d in zip(fs, ds):

    for line in d:
        f.write(line)
    f.close()
