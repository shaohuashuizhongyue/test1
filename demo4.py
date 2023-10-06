def listanalysis(input_list):
    # 删除字符串
    output_list = [i for i in input_list if isinstance(i, int)]
    # 升序排序
    output_list.sort()
    return output_list

input_list = [1, 3, 'a', 4, 'b', 2]
print(listanalysis(input_list))
