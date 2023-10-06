#way 1
num=[1,2,3]
num.sort(reverse=True)
for number in num:
    print(number)
#way2
nums = [1, 2, 3]
for i in range(len(nums)-1, -1, -1):  #(数组长度-1，索引为0结束，递减1)
    for j in range(i):
        if nums[j] < nums[j+1]:
            nums[j], nums[j+1] = nums[j+1], nums[j]

for num in nums:
    print(num)
