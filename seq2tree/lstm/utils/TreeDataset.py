import random
from utils.utils import convert_to_tree, tree2hierarchy_input_stringv2, tree2hierarchy_target_stringv2

class TreeDataset:

    def __init__(self, data_path, batch_size, symbol_manager):
        data_f = open(data_path, "r", encoding="utf-8")

        self.symbol_manager = symbol_manager

        data = []

        for line in data_f:
            line = line.strip().split()
            data.append(symbol_manager.symbols2ids_list(line))

        data_f.close()

        # random.shuffle(data)
        self.data = data
        self.batch_size = batch_size

        self.current_data_pos = 0


    def get_batch(self):

        batch_data = []

        batch_hierachy_input_strs = []
        batch_hierachy_target_strs = []

        batch_hierachy_input_lens = []

        non_terminal_id = self.symbol_manager.get_symbol2id(self.symbol_manager.NT)
        sos_id = self.symbol_manager.get_symbol2id(self.symbol_manager.SOS)
        eos_id = self.symbol_manager.get_symbol2id(self.symbol_manager.EOS)
        left_bracket_id = self.symbol_manager.get_symbol2id(self.symbol_manager.LF_B)
        right_bracket_id = self.symbol_manager.get_symbol2id(self.symbol_manager.RT_B)

        # data_pos = self.current_data_pos

        for _ in range(self.batch_size):

            d = self.data[self.current_data_pos]
            t = convert_to_tree(d, 0, len(d)-1, self.symbol_manager)

            hierachy_input_str = tree2hierarchy_input_stringv2(t, non_terminal_id, sos_id, left_bracket_id)
            hierachy_target_str = tree2hierarchy_target_stringv2(t, non_terminal_id, eos_id)

            batch_hierachy_input_lens.append(len(hierachy_input_str))

            batch_hierachy_input_strs.append(hierachy_input_str)
            batch_hierachy_target_strs.append(hierachy_target_str)

            self.current_data_pos += 1
            if self.is_empty():
                break

        # self.current_data_pos = data_pos

        # padding eos
        max_len = max(batch_hierachy_input_lens)

        for i, input_str in enumerate(batch_hierachy_input_strs):
            batch_hierachy_input_strs[i] = input_str + [eos_id]*(max_len-len(input_str))

        for i, input_str in enumerate(batch_hierachy_target_strs):
            batch_hierachy_target_strs[i] = input_str + [eos_id]*(max_len-len(input_str))

        return (batch_hierachy_input_strs, batch_hierachy_target_strs, batch_hierachy_input_lens)

    def reset(self):
        self.current_data_pos = 0

    def is_empty(self):
        return self.current_data_pos >= len(self.data)
