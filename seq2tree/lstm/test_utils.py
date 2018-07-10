from utils.SymbolManager import SymbolManager
from utils.utils import *

data_parent = "./data/seq2tree_geoqueries"
vocab_tgt_name = "vocab.f.txt"

symbol_manager = SymbolManager(data_parent + "/" + vocab_tgt_name)

def to_str(id_list, symbol_manager):
    return symbol_manager.ids2symbols_list(id_list)

sos_id = symbol_manager.get_symbol2id(symbol_manager.SOS)
eos_id = symbol_manager.get_symbol2id(symbol_manager.EOS)
non_terminal_id = symbol_manager.get_symbol2id(symbol_manager.NT)
left_bracket_id = symbol_manager.get_symbol2id(symbol_manager.LF_B)
right_bracket_id = symbol_manager.get_symbol2id(symbol_manager.RT_B)

#lambda $0 e ( exists $1 ( and ( mountain:t $1 ) ( loc:t $1 $0 ) ) )
#lambda $0 e ( and ( state:t $0 ) ( next_to:t $0 s0 ) )
#lambda $0 e ( and ( state:t $0 ) ( exists $1 ( and ( city:t $1 ) ( named:t $1 n0 ) ( loc:t $1 $0 ) ) ) )
data = "lambda $0 e ( and ( state:t $0 ) ( exists $1 ( and ( city:t $1 ) ( named:t $1 n0 ) ( loc:t $1 $0 ) ) ) )"

print("source_string:", data)

data = data.strip().split()

data_ids = symbol_manager.symbols2ids_list(data)

tree = convert_to_tree(data_ids, 0, len(data_ids)-1, symbol_manager)

tree2string_ids = convert_to_string(tree, left_bracket_id, right_bracket_id)
print("tree2string_ids:", tree2string_ids)
print("original_tree_string:", ' '.join(to_str(tree2string_ids, symbol_manager)))

hierachy_input_string = tree2hierarchy_input_stringv2(tree, non_terminal_id, sos_id, left_bracket_id)
print("hierachy_input_string:", ' '.join(to_str(hierachy_input_string, symbol_manager)))

hierachy_target_string = tree2hierarchy_target_stringv2(tree, non_terminal_id, eos_id)
print("hierachy_target_string：", ' '.join(to_str(hierachy_target_string, symbol_manager)))

transformed_tree = string2treev2(hierachy_target_string, non_terminal_id, eos_id)
transformed_str_ids = convert_to_string(transformed_tree, left_bracket_id, right_bracket_id)
print("transformed_tree_string:", ' '.join(to_str(transformed_str_ids, symbol_manager)))
