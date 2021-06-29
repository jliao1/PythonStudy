'''
python初始化list列表（1维、2维）
1.初始化递增的list：

list1 = range(10)
#print list1
#[0,1,2,...,9]

2.初始化每项为0的一维数组：

list2 = [0] * 5
#print list2
#[0,0,0,0,0]


3.初始化固定值的一维数组：

initVal = 1
listLen = 5
list3 = [ initVal for i in range(5)]
#print list3
#[1,1,1,1,1]
list4 = [initVal] * listLen
#print list4
#[1,1,1,1,1]

4.初始化一个5x6每项为0（固定值）的数组（推荐使用）：

multilist = [[0 for col in range(5)] for row in range(6)]

5.初始化一个5x6每项为0（固定值）的数组

multilist = [[0] * 5 for row in range(3)]



看到了以上的方法，那初始化一个二维数组时，是否可以这样做呢：
multi = [[0] * 5] * 3

其实，这样做是不对的，因为[0] * 5是一个一维数组的对象，* 3的话只是把对象的引用复制了3次，比如，我修改multi[0][0]：
multi = [[0] * 5] * 3
multi[0][0] = 'Hello'
print multi

输出的结果将是：
[['Hello', 0, 0, 0, 0], ['Hello', 0, 0, 0, 0], ['Hello', 0, 0,0, 0]]
我们修改了multi[0][0]，却把我们的multi[1][0]，multi[2][0]也修改了。这不是我们想要的结果。

但是如下写法是对的：
multilist = [[0] * 5 for row in range(3)]
multilist[0][0] = 'Hello'
print multilist
我们看输出结果：
[['Hello', 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
恩，没问题。但是，由于使用 * 的方法比较容易引起混淆导致Bug，所以还是推荐使用上面方法4，即：

multilist = [[0 for col in range(5)] for row in range(6)]


for range 倒序
for i in range(100,0,-1):
    print(i) # 打印出从 100 到 1


快速开数组/开 list 的写法
比如快速生成含有 26个0 的list：list = [0 for _ in range(26)]
                   或者写成：list = [0] * 26

'''