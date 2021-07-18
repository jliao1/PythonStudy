import collections
from queue import Queue
from enum import Enum
from collections import deque
class TreeNode:

    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None
class Sequence(Enum):
    positive = 1
    negative = 2
# 宽度优先遍历 时间空间复杂度讲解
"""
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
"""

# 基本的 宽度优先遍历写法
def breadth_first_traverse(root):
    """
    宽度优先遍历，一层一层的，横向遍历先，left到right，
    这用queue来实现
    why？既然用queue肯定要用queue的特性，那就是先进先出
    why先进先出？在某层中，如果先遍历到某一个节点，那在下一层中，依旧会先遍历它的子节点，这就先进先出了

    总结：所有节点进que了一次，也出que了一次，进/出顺序 就是这个二叉树遍历的顺序
    """
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

# 分层的 宽度优先遍历（记录了每一个节点，在哪一层）
def breadth_first_traverse_by_level(root):
    """
    上面那种方法遍历出来，不知道哪些节点属于同一层，做题时很多时需要知道
    二叉树的分层遍历：（比以上写法只多了一层循环）
    这样分层遍历能知道哪些节点在同一层，但是也不能确定二叉树的完整结构
    那这还有什么应用意义呢？ 二叉树的层数，记录了 root 节点 到当前 node 的 路径长度. 可以由 level -> 到根节点路径长度
                        这在宽度优先搜索里比较有用，因为宽度优先搜索往往是求 在tree中，根节点到当前节点 最小路径长度是多少

    无论是否分层，时间复杂度都是O(n)，因为每个节点都只 进队列/出队列 1次
    空间复杂度，队列中最大的空间，一定小于2层点数，大于某一层的的节点数，所以空间复杂度由最多层的节点数 决定，一般情况下说它是 O(n) 如果它是full tree的话
    """
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

# 题集
class Solution:

    # Lintcode Medium 70 · Binary Tree Level Order Traversal II
    def levelOrderBottom(self, root):
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

    # Lintcode Easy 242 · Convert Binary Tree to Linked Lists by Depth 考了二叉树和链表
    def binaryTreeToLists(self, root):
        # Write your code here
        res = []

        if not root:
            return res

        que = Queue()
        que.put(root)


        while not que.empty():
            n = que.qsize()

            dummy = ListNode(-1)
            tail = dummy

            for i in range(n):
                cur = que.get()
                tail.next = ListNode(cur.val)
                tail = tail.next    # 始终保持 tail 指向链表的尾部

                if cur.left:
                    que.put(cur.left)
                if cur.right:
                    que.put(cur.right)

            res.append(dummy.next)

        return res

    # lintcode Medium 71 · Binary Tree Zigzag Level Order Traversal  用双端队列，双端进出做的
    def zigzagLevelOrder1(self, root):
        """
        我自己首先用双端队列做的，我觉得也比较好懂
        但可能……有点藕合所以代码有点长  可以看看方法2和3
        """
        if not root:
            return []

        q = deque()
        q.append(root)

        res = []
        order = Sequence.positive

        while len(q) != 0:

            length = len(q)

            if length != 0 and order == Sequence.positive:
                level = []
                for i in range(length):

                    cur = q.popleft()
                    level.append(cur.val)

                    if cur.left:
                        q.append(cur.left)
                    if cur.right:
                        q.append(cur.right)

                order = Sequence.negative

                res.append(level)

            length = len(q)
            if length != 0 and order == Sequence.negative:
                level = []
                for i in range(length):
                    cur = q.pop()
                    level.append(cur.val)

                    if cur.right:
                        q.appendleft(cur.right)
                    if cur.left:
                        q.appendleft(cur.left)

                order = Sequence.positive

                res.append(level)

        return res

    # lintcode Medium 71 · Binary Tree Zigzag Level Order Traversal 九章老师用2个stack做的(好像双端队列都可以转换成2个stack做)
    def zigzagLevelOrder2(self, root):
        list = []
        if not root:
            return list

        stack1 = []
        stack2 = []
        is_left_to_right = True  # 由于是从第一层开始，默认是 true

        stack1.append(root)

        # BFS
        while len(stack1) != 0:
            curr_level_nodes = []

            while len(stack1) != 0:
                cur = stack1.pop()

                if not cur:   # 要先检车如果cur是None, 就跳过接下来的，继续下个循环
                    continue  # 为什么要check呢？因为后面append的时候，是会把 None node append进去的
                else:         # 如果cur不是None，就加到本层listNodes里
                    curr_level_nodes.append(cur.val)

                if is_left_to_right:
                    stack2.append(cur.left)
                    stack2.append(cur.right)
                else:
                    stack2.append(cur.right)
                    stack2.append(cur.left)

            # check本层列表是不是空的，非空再append
            if curr_level_nodes:
                '''
                为什么要check空？
                因为 curr_level_nodes 初始是空的
                在最后一层level里，由于都是None nodes，所以添加不进 curr_level_nodes
                导致 curr_level_nodes 依然是空的，就不能 append 进 list
                '''
                list.append(curr_level_nodes)

            # 翻转遍历方向了
            is_left_to_right = not is_left_to_right

            # stack的迭代（此时stack1已空，stack2添加好了，把里面的元素交换一下）
            stack1, stack2 = stack2, stack1

        return list

    # lintcode Medium 71 · Binary Tree Zigzag Level Order Traversal 这个只是用正常的bfs做，需要逆序的时候翻转一下列表再append，思路最简单吧
    def zigzagLevelOrder3(self, root):
        """
        题目主要思路： BFS + Queue + 翻转
        """
        if not root:
            return []

        # bfs 开始了
        lists = []

        q = collections.deque()
        q.append(root)
        is_left_to_right = True

        while q:
            this_level_nodes = []
            size = len(q)

            for i in range(size):
                cur = q.popleft()
                this_level_nodes.append(cur.val)
                '''
                可不可以接下来，不检查左右孩子是否为空，直接入队？
                可以，但需要
                1。加一个逻辑，如果cur取出的是null，就continue
                2。如果this_level_nodes为空，不append进lists
                '''
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)

            if not is_left_to_right:
                # 如果需要逆序，那就把level_nodes去reverse 一下再 append
                this_level_nodes.reverse()

            lists.append(this_level_nodes)

            # 翻转读取顺序
            is_left_to_right = not is_left_to_right

        return lists



# 建立个二叉树，测试用
def build_tree1():
    """
            8
           /  \
          3    10
         / \     \
        1   6     14
            /\    /
           4  7  13

    """
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
def build_tree2():
    """
        1
       / \
      2   3
         / \
        4   5
    """
    node_1 = TreeNode(1)
    node_2 = TreeNode(2)
    node_3 = TreeNode(3)
    node_4 = TreeNode(4)
    node_5 = TreeNode(5)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(13)

    node_1.right = node_2
    node_1.left = node_3
    node_3.left = node_4
    node_3.right = node_5

    return node_1
def build_tree3():
    """
        1
       / \
      2   3
     /
    4
    """
    node_1 = TreeNode(1)
    node_2 = TreeNode(2)
    node_3 = TreeNode(3)
    node_4 = TreeNode(4)

    node_1.left = node_2
    node_1.right = node_3
    node_2.left = node_4

    return node_1

if __name__ == '__main__':

    root = build_tree3()
    sol = Solution()
    l = sol.addOneRow2(root, 5, 4)
    print(l)




