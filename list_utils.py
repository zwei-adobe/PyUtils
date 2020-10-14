
def get_common_list_2(list1, list2):
    return list(set(list1) & set(list2))

    # return new_list

def any_item_in_dict(input_list, target_dict):
    for s_item in input_list:
        if s_item in target_dict:
            return True
    return False


def item_idx_in_dict(input_list, target_dict):
    result = []
    for s_idx, s_item in enumerate(input_list):
        if s_item in target_dict:
            result.append(s_idx)

    return result