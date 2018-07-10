from sys import argv
import os

parent = "../data/seq2tree_geoqueries"

tmp_dir = parent + "/" + "tmp"

if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

f_name = argv[1]

f = open(parent + "/" + f_name, "r", encoding="utf-8")

srcs = []
tgts = []

for line in f:
    line = line.strip().split("\t")
    srcs.append(line[0].strip())
    tgts.append(line[1].strip())

f.close()

src_f = open(tmp_dir + "/" + "src_" + f_name, "w", encoding="utf-8")

for src in srcs:
    src_f.write(src + "\n")
src_f.close()

tgt_f = open(tmp_dir + "/" + "tgt_" + f_name, "w", encoding="utf-8")

for tgt in tgts:
    tgt_f.write(tgt + "\n")
tgt_f.close()
