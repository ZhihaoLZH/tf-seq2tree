import tensorflow as tf


from utils.TreeDataset import TreeDataset
from utils.NormalDataset import NormalDataset
from utils.SymbolManager import SymbolManager
from utils.utils import string2treev2, convert_to_string, compute_accuracy

from utils.bleu import compute_bleu

from seq2tree_model import Seq2TreeModel

data_parent = "./data/seq2tree_geoqueries"

src_train_f_name = "src_train.txt"
tgt_train_f_name = "tgt_train.txt"

src_test_f_name = "src_test.txt"
tgt_test_f_name = "tgt_test.txt"

vocab_src_name = "vocab.q.txt"
vocab_tgt_name = "vocab.f.txt"

src_symbol_manager = SymbolManager(vocab_path=data_parent + "/" + vocab_src_name)
tgt_symbol_manager = SymbolManager(vocab_path=data_parent + "/" + vocab_tgt_name)

print("src_vocab_size:", src_symbol_manager.size())
print("tgt_vocab_size:", tgt_symbol_manager.size())

BATCH_SIZE = 20

learning_rate = 0.005

def create_dataset(src_data_path,
        tgt_data_path,
        batch_size,
        src_symbol_manager,
        tgt_symbol_manager):

        src_dataset = NormalDataset(data_path=src_data_path,
                                    batch_size=batch_size,
                                    symbol_manager=src_symbol_manager)

        tgt_dataset = TreeDataset(data_path=tgt_data_path,
                                batch_size=batch_size,
                                symbol_manager=tgt_symbol_manager)

        return src_dataset, tgt_dataset

def eval(sess, eval_model, src_dataset, tgt_dataset, symbol_manager):

    predicts = []

    references = []

    ori_predicts = []

    non_terminal_id = symbol_manager.get_symbol2id(symbol_manager.NT)
    eos_id = symbol_manager.get_symbol2id(symbol_manager.EOS)
    left_bracket_id = symbol_manager.get_symbol2id(symbol_manager.LF_B)
    right_bracket_id = symbol_manager.get_symbol2id(symbol_manager.RT_B)

    while not src_dataset.is_empty() and not tgt_dataset.is_empty():
        src_batch = src_dataset.get_batch()
        tgt_batch = tgt_dataset.get_batch()

        _, batch_hierachy_tgt_str_ids, _ = tgt_batch

        batch_predict_ids = eval_model.decode(sess, src_batch).tolist()

        batch_predict_words = [symbol_manager.ids2symbols_list(convert_to_string(string2treev2(predict_ids, non_terminal_id, eos_id), left_bracket_id, right_bracket_id))  for predict_ids in batch_predict_ids]
        batch_hierachy_tgt_words = [symbol_manager.ids2symbols_list(convert_to_string(string2treev2(hierachy_tgt_str_ids, non_terminal_id, eos_id), left_bracket_id, right_bracket_id)) for hierachy_tgt_str_ids in batch_hierachy_tgt_str_ids]

        batch_ori_predict_words = [symbol_manager.ids2symbols_list(predict_ids) for predict_ids in batch_predict_ids]

        predicts.extend(batch_predict_words)
        references.extend(batch_hierachy_tgt_words)

        ori_predicts.extend(batch_ori_predict_words)

    pred_gold_f = open(data_parent + "/tmp/" + "pred_gold.txt", "w", encoding="utf-8")

    ori_pred_f = open(data_parent + "/tmp/" + "ori_pred.txt", "w", encoding="utf-8")

    for p, r, op in zip(predicts, references, ori_predicts):
        p = ' '.join(p)
        r = ' '.join(r)
        op = ' '.join(op)

        pred_gold_f.write(p + "\t" + r + "\n")
        ori_pred_f.write(op + "\n")

    pred_gold_f.close()
    ori_pred_f.close()

    src_dataset.reset()
    tgt_dataset.reset()

    acc, not_match = compute_accuracy(predicts, references)

    not_match_f = open(data_parent + "/tmp/" + "not_match.txt", "w", encoding="utf-8")
    for c, r in not_match:
        c = ' '.join(c)
        r = ' '.join(r)
        not_match_f.write(c + "\t" + "r" + "\n")

    not_match_f.close()

    return acc


sos_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.SOS)
eos_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.EOS)
non_terminal_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.NT)
left_bracket_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.LF_B)
right_bracket_id = tgt_symbol_manager.get_symbol2id(tgt_symbol_manager.RT_B)

src_train_dataset, tgt_train_dataset = create_dataset(
                        src_data_path=data_parent + "/tmp/" + src_train_f_name,
                        tgt_data_path=data_parent + "/tmp/" + tgt_train_f_name,
                        batch_size=BATCH_SIZE,
                        src_symbol_manager=src_symbol_manager,
                        tgt_symbol_manager=tgt_symbol_manager,
                        )

src_test_dataset, tgt_test_dataset = create_dataset(
                        src_data_path=data_parent + "/tmp/" + src_test_f_name,
                        tgt_data_path=data_parent + "/tmp/" + tgt_test_f_name,
                        batch_size=BATCH_SIZE,
                        src_symbol_manager=src_symbol_manager,
                        tgt_symbol_manager=tgt_symbol_manager,
                        )

train_model = Seq2TreeModel(
    mode=tf.contrib.learn.ModeKeys.TRAIN,
    learning_rate=learning_rate,
    src_vocab_size=src_symbol_manager.size(),
    tgt_vocab_size=tgt_symbol_manager.size(),
    embedding_size=200,
    hidden_size=200,
    sos_id=sos_id,
    non_terminal_id=non_terminal_id,
    eos_id=eos_id,
    left_bracket_id=left_bracket_id,
    right_bracket_id=right_bracket_id,
    learning_rate_decay=0.97,
    learning_rate_decay_after=5,
)

eval_model = Seq2TreeModel(
    mode=tf.contrib.learn.ModeKeys.EVAL,
    learning_rate=learning_rate,
    src_vocab_size=src_symbol_manager.size(),
    tgt_vocab_size=tgt_symbol_manager.size(),
    embedding_size=200,
    hidden_size=200,
    sos_id=sos_id,
    non_terminal_id=non_terminal_id,
    eos_id=eos_id,
    left_bracket_id=left_bracket_id,
    right_bracket_id=right_bracket_id,
    learning_rate_decay=0.97,
    learning_rate_decay_after=5,
)

EPOCH = 110

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    best_acc = 0.0
    # best_train_acc = 0.0

    for i in range(EPOCH):

        epoch_change = True

        while not src_train_dataset.is_empty() and not tgt_train_dataset.is_empty():
            src_train_batch = src_train_dataset.get_batch()
            tgt_train_batch = tgt_train_dataset.get_batch()

            loss, predict_ids = train_model.train(sess, src_train_batch, tgt_train_batch, i, epoch_change)

            print("epoch%d-loss:%f"%(i+1, loss))

            epoch_change = False

        src_train_dataset.reset()
        tgt_train_dataset.reset()

        if (i+1) % 1 == 0:
            # train_acc = eval(sess, eval_model, src_train_dataset, tgt_train_dataset, tgt_symbol_manager)
            acc = eval(sess, eval_model, src_test_dataset, tgt_test_dataset, tgt_symbol_manager)

            print("epoch%d-acc:%f"%(i+1, acc))
            # print("epoch%d-train-acc:%f"%(i+1, train_acc))

            if acc > best_acc:
                best_acc = acc

            # if train_acc > best_train_acc:
            #     best_train_acc = train_acc

            print("best-acc:%f"%(best_acc))
            # print("best-train-acc:%f"%(best_train_acc))
