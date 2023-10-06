# 输入一个字符串
String = input()

# 判断字符串中是否含有子串"ol"
if String.find("ol") >= 0:
    # 将所有的"ol"替换为"fzu"
    String = String.replace("ol", "fzu")
# 倒序输出符串
String = String[::-1]
print(String)