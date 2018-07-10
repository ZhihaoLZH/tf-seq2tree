from utils.Tree import Tree


def convert_to_tree(r_list, i_left, i_right, symbol_manager):
    t = Tree()
    level = 0
    left = -1
    for i in range(i_left, i_right+1):
        if r_list[i] == symbol_manager.get_symbol2id('('):
            if level == 0:
                left = i

            level = level + 1
        elif r_list[i] == symbol_manager.get_symbol2id(')'):
            level = level - 1
            if level == 0:
                c = convert_to_tree(r_list, left+1, i-1, symbol_manager)
                t.add_child(c)

        elif level == 0:
            t.add_child(r_list[i])

    return t

def convert_to_string(tree, left_bracket_id, right_bracket_id):
    global level
    l = []
    for child in tree.childern:

        if type(child) == Tree:
            sub_l = convert_to_string(child, left_bracket_id, right_bracket_id)
            l = l + [left_bracket_id] + sub_l + [right_bracket_id]
        else:
            l.append(child)

    return l

#convert a tree to a string following the hierachy order
def tree2hierarchy_input_string(tree, non_terminal_id, sos_id, left_bracket_id):
    queue = [tree]

    result = []

    tree_start_flag = 1

    while len(queue) != 0:
        t = queue.pop(0)

        result.append(sos_id)

        if tree_start_flag == 1:
            # result.append(sos_id)
            tree_start_flag = 0
        else:
            result.append(left_bracket_id)

        for child in t.childern:
            if type(child) == Tree:
                queue.append(child)
                result.append(non_terminal_id)
            else:
                result.append(child)

        # result.append(eos_id)

    return result

def tree2hierarchy_input_stringv2(tree, non_terminal_id, sos_id, left_bracket_id):
    queue = [tree]

    result = []

    tree_start_flag = 1

    while len(queue) != 0:
        t = queue.pop(0)

        if tree_start_flag == 1:
            result.append(sos_id)
            tree_start_flag = 0
        else:
            result.append(left_bracket_id)

        for child in t.childern:
            if type(child) == Tree:
                queue.append(child)
                result.append(non_terminal_id)
            else:
                result.append(child)

        # result.append(eos_id)

    return result


def tree2hierarchy_target_string(tree, non_terminal_id, eos_id, left_bracket_id, right_bracket_id):
    queue = [tree]

    result = []

    tree_start_flag = 1

    while len(queue) != 0:
        t = queue.pop(0)

        if tree_start_flag != 1:
            result.append(left_bracket_id)

        for child in t.childern:
            if type(child) == Tree:
                queue.append(child)
                result.append(non_terminal_id)
            else:
                result.append(child)

        if tree_start_flag == 1:
            result.append(eos_id)
            tree_start_flag = 0
        else:
            result.append(right_bracket_id)

    return result

def tree2hierarchy_target_stringv2(tree, non_terminal_id, eos_id):
    queue = [tree]

    result = []

    tree_start_flag = 1

    while len(queue) != 0:
        t = queue.pop(0)

        # if tree_start_flag != 1:
        #     result.append(left_bracket_id)

        for child in t.childern:
            if type(child) == Tree:
                queue.append(child)
                result.append(non_terminal_id)
            else:
                result.append(child)

        result.append(eos_id)

        # if tree_start_flag == 1:
        #     result.append(eos_id)
        #     tree_start_flag = 0
        # else:
        #     result.append(right_bracket_id)

    return result

def string2tree(str_list, non_terminal_id, eos_id, left_bracket_id, right_bracket_id):
    root = Tree()

    t = root

    queue = []

    for ch in str_list:
        if ch == non_terminal_id:
            subtree = Tree()
            queue.append(subtree)
            t.add_child(subtree)
        elif ch == eos_id:
            if len(queue) <= 0:
                break
        elif ch == left_bracket_id:
            t = queue.pop(0)
        elif ch == right_bracket_id:
            if len(queue) <= 0:
                break
        else:
            t.add_child(ch)

    return root

def string2treev2(str_list, non_terminal_id, eos_id):
    root = Tree()

    t = root

    queue = []

    for ch in str_list:
        if ch == non_terminal_id:
            subtree = Tree()
            queue.append(subtree)
            t.add_child(subtree)
        elif ch == eos_id:
            if len(queue) <= 0:
                break
            else:
                t = queue.pop(0)
        else:
            t.add_child(ch)

    return root

def is_all_same(candidate, reference):
    if len(candidate) == len(reference):
        all_same = True

        for i in range(len(candidate)):
            if candidate[i] != reference[i]:
                all_same = False
                break

        return all_same
    else:
        return False

def compute_accuracy(candidate_list, reference_list):
    if len(candidate_list) != len(reference_list):
        print("length: cadidate list(%d) != reference list(%d)"%(len(candidate_list), len(reference_list)))

    l = min(len(candidate_list), len(reference_list))

    count = 0
    not_match = []

    for i in range(l):
        if is_all_same(candidate_list[i], reference_list[i]):
            count += 1
        else:
            not_match.append((candidate_list[i], reference_list[i]))
    # print("count:", count)
    # print("total:", l)
    # print("compute_acc:", count/l)
    return float(count) / float(l), not_match
