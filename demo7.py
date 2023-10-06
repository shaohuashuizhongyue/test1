class Product:
    def __init__(self, id, name, price, total, remaining):
        self.__id = id
        self.__name = name
        self.__price = price
        self.__total = total
        self.__remaining = remaining

    def display(self):
        print("ID: ", self.__id)
        print("Name: ", self.__name)
        print("Price: ", self.__price)
        print("Total: ", self.__total)
        print("Remaining: ", self.__remaining)

    def income(self):
        sold = self.__total - self.__remaining
        return sold * self.__price

    def setdata(self, id, name, price, total, remaining):
        self.__id = id
        self.__name = name
        self.__price = price
        self.__total = total
        self.__remaining = remaining

# 创建一个商品实例
p = Product(1, 'egg', 1, 10, 1)
# 显示商品信息
p.display()
# 计算已售出商品价值
print("Income: ", p.income())
# 修改商品信息
p.setdata(2, 'Orange', 1, 20, 1)
# 显示修改后的商品信息
p.display()
