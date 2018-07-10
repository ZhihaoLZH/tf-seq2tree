import random
from utils.utils import convert_to_tree

class NormalDataset:

    def __init__(self, data_path, batch_size, symbol_manager):
        data_f = open(data_path, "r", encoding="utf-8")

        data = []

        for line in data_f:
            line = line.strip().split()
            data.append(symbol_manager.symbols2ids_list(line))

        # random.shuffle(data)
        self.data = data
        self.batch_size = batch_size
        self.symbol_manager = symbol_manager
        self.current_data_pos = 0


    def get_batch(self):

        batch_data = []
        seq_lens = []

        # data_pos = self.current_data_pos

        for _ in range(self.batch_size):

            d = self.data[self.current_data_pos]
            batch_data.append(d)
            seq_lens.append(len(d))

            self.current_data_pos += 1

            if self.is_empty():
                break

        # self.current_data_pos = data_pos

        max_len = max(seq_lens)
        eos_id = self.symbol_manager.get_symbol2id(self.symbol_manager.EOS)

        for i, seq in enumerate(batch_data):
            batch_data[i] = seq + [eos_id] * (max_len-len(seq))

        return (batch_data, seq_lens)

    def reset(self):
        self.current_data_pos = 0

    def is_empty(self):
        return self.current_data_pos >= len(self.data)
