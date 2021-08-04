'''
A lot of times when dealing with iterators, we also get a need to keep a count of iterations.
Python eases the programmers’ task by providing a built-in function enumerate() for this task.
Enumerate() method adds a counter to an iterable and returns it in a form of enumerate object.
This enumerate object can then be used directly in for loops or be converted into a list of tuples using list() method.
'''

if __name__ == '__main__':


    list = ["eat", "sleep", "repeat"]
    for each in enumerate(list):
        # 这个 each 是一个 enumerate object

        # 取counter
        print(each[0])
        # 取具体元素
        print(each[0])


    l1 = ["eat", "sleep", "repeat"]
    s1 = "geek"

    # creating enumerate objects
    obj1 = enumerate(l1)
    obj2 = enumerate(s1)

    print("Return type:", type(obj1))
    print(obj1)
    print(list(enumerate(l1))) #  be converted into a list of tuples using list() method.

    print(enumerate(l1))

    # changing start index to 2

    print(list(enumerate(l1, 2)))
    print(list(enumerate(s1, 2)))

    # Python program to illustrate
    # enumerate function in loops
    l1 = [[1,2,3], [], [5,6]]

    # printing the tuples in object directly
    for v in l1:
        for each in enumerate(v):
            print(each)
        '''
        默认从0才开始cout, 打印出来是
        (0, 'eat')
        (1, 'sleep')
        (2, 'repeat')
        '''

    # changing index and printing separately
    for count, ele in enumerate(l1, 100): # 从100开始计
        print(count, ele)
        '''
        从100开始count, 打印出来是
                        100 eat
                        101 sleep
                        102 repeat
        '''