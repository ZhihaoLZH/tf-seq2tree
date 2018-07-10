from utils.Tree import Tree
from utils.utils import *
from utils.SymbolManager import SymbolManager

parent = "./data/seq2tree_geoqueries/tmp"
parent2 = "./data/seq2tree_geoqueries"

train_src_name = "src_train.txt"
train_tgt_name = "tgt_train.txt"

test_src_name = "src_test.txt"
test_tgt_name = "tgt_test.txt"


tr_src_f = open(parent + "/" + train_src_name, "r", encoding="utf-8")
tr_tgt_f = open(parent + "/" + train_tgt_name, "r", encoding="utf-8")
test_src_f = open(parent + "/" + test_src_name, "r", encoding="utf-8")
test_tgt_f = open(parent + "/" + test_tgt_name, "r", encoding="utf-8")

tgt_symbol_manager = SymbolManager(vocab_path=parent2 + "/" + "vocab.f.txt")

tr_srcs = [src.strip() for src in tr_src_f.readlines()]
tr_tgts = [tgt.strip() for tgt in tr_tgt_f.readlines()]

test_srcs = [src.strip() for src in test_src_f.readlines()]
test_tgts = [tgt.strip() for tgt in test_tgt_f.readlines()]

tr_src_f.close()
tr_tgt_f.close()

test_src_f.close()
test_tgt_f.close()

non_terminal_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.NT)
sos_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.SOS)
eos_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.EOS)
left_bracket_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.LF_B)
right_bracket_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.RT_B)

tr_src_lens = [len(src)  for src in tr_srcs]
tr_tgt_lens = [len(tree2hierarchy_target_stringv2(convert_to_tree(tgt_symbol_manager.symbols2ids_list(tgt), 0, len(tgt)-1, tgt_symbol_manager) , non_terminal_id, eos_id)) for tgt in tr_tgts]

test_src_lens = [len(src) for src in test_srcs]
test_tgt_lens = [len(tree2hierarchy_target_stringv2(convert_to_tree(tgt_symbol_manager.symbols2ids_list(tgt), 0, len(tgt)-1, tgt_symbol_manager) , non_terminal_id, eos_id)) for tgt in test_tgts]

total_src_lens = tr_src_lens + test_src_lens
total_tgt_lens = tr_tgt_lens + test_tgt_lens

min_tr_src_len = min(tr_src_lens)
max_tr_tgt_len = max(tr_tgt_lens)

min_test_src_len = min(test_src_lens)
max_test_tgt_len = max(test_tgt_lens)

total_min_src_len = min(total_src_lens)
total_max_tgt_len = max(total_tgt_lens)

print("min-train-src-len:%d"%(min_tr_src_len))
print("max-train-tgt-len:%d"%(max_tr_tgt_len))
print("min-test-src-len:%d"%(min_test_src_len))
print("max-test-tgt-len:%d"%(max_test_tgt_len))

print("train set ratio:%d"%(max_tr_tgt_len/min_tr_src_len))
print("test set ratio:%d"%(max_test_tgt_len/min_test_src_len))
print("total ratio:%d"%(total_max_tgt_len/total_min_src_len))
