'''
under the hood 在引擎盖下面；在底层

所有的排序都可以默认是 NlogN

【String】
好像string的slice时间复杂度很复杂

input_str.split() 时间复杂度是 O(n) n是input_str的长度

output = input_str.join(list_of_words) 用来concatenate字符串最高效，时复是 O(n), n is the total length of the output
                                       而如果用 + 来连接，每 + 一次都要创建一个新的string把旧string全部copy一次 For concatenation of strings with varying lengths, it should be O(N + M) where N and M are the lengths of the two strings being concatenated

【list】
List结构查找是O(n), 在头增加/删除是 O(n)，在尾部增加/删除 是O(1)
如果有个 list 名称叫 L
L.pop() 是默认删除最后一个是O(1)
python 里 len(L) 也是 O(1) 的，长度是list结构的一个属性
然后 iter(list)时间复杂度应该是O(1), 查看了list 似乎是自带了 __iter__ 方法，应该是在创建 list 的时候就完成了。怎么查看？print( dir(list))

【collections.deque】
A deque (double-ended queue) is represented internally as a doubly linked list.
所以To implement a queue, use collection.deque which was designed to have fast appends and pops from both ends.

具体的时间复杂度:
Operation	Average Case	Amortized Worst Case
Copy		O(n)		O(n)
append		O(1)		O(1) # 插入 right end
appendleft	O(1)		O(1)
pop		    O(1)		O(1) # pop from right end
popleft		O(1)		O(1)
extend		O(k)		O(k)
extendleft	O(k)		O(k)
rotate		O(k)		O(k)
remove		O(n)		O(n)
参考网页1：https://wiki.python.org/moin/TimeComplexity
参考网页2：https://www.geeksforgeeks.org/queue-in-python/
'''


'''
def concat_strings():
    """
    This is a program to remove spaces in a string
    :return:
    """

    input_string = "Th is is an ex am pl ew it hs pa ce"

    output_string = ""

    for i in input_string:
        if i == " ":
            pass
        else:
            output_string += i

    print(output_string)


concat_strings()
Output:
Thisisanexamplewithspace

'''