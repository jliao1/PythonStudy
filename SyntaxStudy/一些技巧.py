
# 把字符串str，一位一位地，转成数字
for c in str:
    num = num * 10 + ord(c) - ord('0')

# 翻转整数 n
reverse = 0
while n > 0:
    reverse = reverse * 10 + n % 10
    n = n // 10