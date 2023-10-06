def count_numbers(list):
    count = {}
    for num in list:
        if num in count:
            count[num] += 1
        else:
            count[num] = 1
    return count

list = [1, 2, 3, 1, 1, 4, 5, 4, 2]
print(count_numbers(list))
