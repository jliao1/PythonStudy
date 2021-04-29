if __name__ == '__main__':
# ----------------------------------------------------------------------------------------------------
    '''
    if not 怎么用？
    当 A = [] 或者 A == None 时， if not A 都会被执行
    还可以看看Stack.py里的is_empty()函数
    '''
    myList = []
    if not myList:  # 译为：如果myList为空，如果没有myList。
                    # 或者这么理解：空的myList 是 False，not myList 就是 true
        print('\n当 myList = [] 空, if not stack 执行了')
    myList.append(0)
    if myList:
        print('myList 里有数据了！')

    root = None
    if not root:
        print('\n当 root = None, if not root 执行了!')
    if root is None:
        print('当 root = None, if root is None 也也执行了!')
    if root == None:
        print('当 root = None, if root == None 也也执行了! ( 但最好别这么写，不太专业)')