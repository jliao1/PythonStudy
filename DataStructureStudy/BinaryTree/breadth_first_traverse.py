# coding=utf-8
# 小 queue 代表模块，就是 python的一个文件
from queue import Queue

# import queue   # 如果这样就导入整个 queue 模块了

class TreeNode:

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# 跟 depth_first_traverse.py 里的一样也是建立个二叉树
def build_tree():
    node_1 = TreeNode(8)
    node_2 = TreeNode(3)
    node_3 = TreeNode(10)
    node_4 = TreeNode(1)
    node_5 = TreeNode(6)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(13)

    node_1.left = node_2
    node_1.right = node_3

    node_2.left = node_4
    node_2.right = node_5

    node_3.right = node_6

    node_5.left = node_7
    node_5.right = node_8

    node_6.left = node_9

    return node_1

# 宽度优先遍历
'''
宽度优先遍历，横向遍历先left再right，
这用queue来实现
why？既然用queue肯定要用queue的特性，那就是先进先出
why先进先出？在某层中，如果先遍历到某一个节点，那在下一层中，依旧会先遍历它的子节点，这就先进先出了
'''
def breadth_first_traverse(root):
    if not root:  # 译为 如果没有root  等价于这句 if root is None    但写成 if not root比较简洁啦
        return

    que = Queue(maxsize=0)       # why maxsize=0 ?  If maxsize is less than or equal to zero, the queue size is infinite
    que.put(root)                # 根节点放到队列里

    while not que.empty():       # 译为：当 que 非空时
        cur = que.get()          # 从 que 中取出一个元素
        print(cur.val)  # 取出以后 打印下它值(开始遍历它嘛)
        # 完事儿后开始遍历它 子节点了
        if cur.left:             # 如果它left存在
            que.put(cur.left)    # 就把它left放进que里
        if cur.right:            # 如果它right存在
            que.put(cur.right)   # 就把它right放que里
# 总结：所有节点进que了一次，也出que了一次，进/出顺序 就是这个二叉树遍历的顺序

'''
上面那种方法遍历出来，不知道哪些节点属于同一层，做题时很多时需要知道
二叉树的分层遍历：（比以上写法只多了一层循环）
这样分层遍历能知道哪些节点在同一层，但是也不能确定二叉树的完整结构
那这还有什么应用意义呢？ 二叉树的层数，记录了 root 节点 到当前 node 的 路径长度. 可以由 level -> 到根节点路径长度
                    这在宽度优先搜索里比较有用，因为宽度优先搜索往往是求 在tree中，根节点到当前节点 最小路径长度是多少
'''
def breadth_first_traverse_by_level(root):
    if not root:
        return

    que = Queue()
    que.put(root)

    level = 1

    while not que.empty():    # 当 que 不是空的
        n = que.qsize()       # 看 que 里有多少元素
        print('Level', level)

        # 这个for循环主要是 先把当前层都遍历打印一次，然后把下一层依照次序放进que
        for i in range(n):    # 这是长度为 n 的for循环，i：0 -> n-1
            cur = que.get()
            print(cur.val, end=' ')
            if cur.left:
                que.put(cur.left)
            if cur.right:
                que.put(cur.right)

        print()              # 一个 for 循环结束后换行
        level = level + 1

'''
二叉树宽度优先遍历
breadth_first_traverse_by_level(root) 和 breadth_first_traverse(root)
时间复杂度：
        无论是否分层，都是 O(n)   因为每一个节点只 进/出队列一次(每个节点看一次)
空间复杂度：
        (1)由于没有使用递归，所以 call stack 的空间我们认为是constant，可以忽略不计
        (2) que = Queue() 这条语句 队列这个对象会放在 heap 空间里，往队列里放元素就会消耗heap空间，
            二叉树的宽度优先遍历过程中，队列中最多放入多少个节点?
                            队列中始终有元素，但又不好说是多少个
                            但是可以确定的是，队列中元素个数一定小于两层节点数（因为一定不可能两层节点都在队列中）
                                           很多时候队列中元素又可以包含一层所有节点数（比如上一层所有节点都出队列了，下一层所有节点可以在里边儿）
                            所以 二叉树宽度优先遍历当中， 某一层 ≤ 队列里面元素个数 < 两层，由整个二叉树中节点数最多的那一层决定的(不会超过它的二倍)
                            那就分析worst case，节点数最多的那一层会有多少个节点呢? 当这个二叉树特别满，最多层的节点数 与 总节点数 的关系。
                                              比如总节点 n = 15个，二叉树很满时，最底层节点有8个，规律是 最多层节点数 = (n+1)/2
        所以答案是，由节点数最多的那一层的节点数决定，O(n)
                                              
'''

# leetcode:
# Binary Tree Level Order Traversal II
def levelOrderBottom(root):
    # write your code here
    res = []    # 存结果

    if not root:
        return res

    que = Queue()
    que.put(root)

    while not que.empty():
        tmp = []   # 注意下层级结构和local scope， tmp 在每次进入 while not que.empty() 时会初始化
        n = que.qsize();

        for i in range(n):
            cur = que.get()
            tmp.append(cur.val)
            if cur.left:
                que.put(cur.left)
            if cur.right:
                que.put(cur.right)

        res.append(tmp)  # 结束后 res 里放的是一个正序的遍历结果

    res.reverse()
    return res
    # 或以上两句写成 return list(reversed(res))  # reversed(res) returns a reverse iterator 然后这个迭代器要传给list
                                               # 其实 range(n) 也是迭代器，是个对象，在对象里可以一点点访问它的值
                                                # list(reversed(res)) 这里想获取list就得强制转换一下




if __name__ == '__main__':
    root = build_tree()
    # breadth_first_traverse(root)
    # breadth_first_traverse_by_level(root)
    # l = levelOrderBottom(root)
    # print(l)


