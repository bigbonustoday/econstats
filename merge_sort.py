def merge_sort(list):
    if len(list) == 1:
        return list
    start = 0
    end = len(list) - 1
    first_half_end = (start + end) / 2

    list1 = merge_sort(list[start: first_half_end + 1])
    list2 = merge_sort(list[first_half_end + 1: end + 1])
    list = list1 + list2

    # merge two sorted lists
    i = start
    j = first_half_end + 1
    sorted_list = []
    while (i <= first_half_end) or (j <= end):
        if (i <= first_half_end) and (j <= end):
            if compare(list[i], list[j]) <= 0:
                sorted_list.append(list[i])
                i += 1
            else:
                sorted_list.append(list[j])
                j += 1
        else:
            if i <= first_half_end:
                sorted_list.append(list[i])
                i += 1
            else:
                sorted_list.append(list[j])
                j += 1
    return sorted_list


def compare(a, b):
    if len(a) < len(b):
        return -1
    elif len(a) == len(b):
        return 0
    else:
        return 1
