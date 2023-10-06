def process_dict(input_dict):
    # 删除学号尾号为偶数的元素
    output_dict = {a: b for a, b in input_dict.items() if int(a[-1]) % 2 != 0}
    return output_dict

input_dict = {'1': 'a', '2': 'b', '3': 'c', '4': 'd'}
print(process_dict(input_dict))
