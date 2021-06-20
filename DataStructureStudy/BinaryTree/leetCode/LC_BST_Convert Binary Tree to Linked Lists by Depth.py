# Leetcode: Convert Binary Tree to Linked Lists by Depth
# 本题考察了，链表，binary tree的宽度优先分层遍历
"""
Definition of TreeNode:
"""
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None




"""
Definition for singly-linked list.
"""

class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
        self.tail = None

    def add(self, node):
        if not self.head:
            self.head = node
            self.tail = node
        else:
            self.tail.next = node
            self.tail = node      # 把 tail 更新，new_node 变成新尾巴

    def get_head(self):
        return self.head




from queue import Queue   # 在使用前import就好，import在哪里无所谓




class Solution:

    # @param {TreeNode} root the root of binary tree
    # @return {ListNode[]} a lists of linked list
    def binaryTreeToLists(self, root):
        # 写法1： 我新建了一个 ListNode 的 class 来写的 （实际上也不用这么做，那就看 # 写法2）
        # res = []
        #
        # if not root:
        #     return res
        #
        # que = Queue()
        # que.put(root)
        #
        # while not que.empty():
        #     n = que.qsize()
        #
        #     linked_list = LinkedList()
        #
        #     for i in range(n):
        #         cur = que.get()
        #
        #         node = ListNode(cur.val)
        #         linked_list.add(node)
        #
        #         if cur.left:
        #             que.put(cur.left)
        #         if cur.right:
        #             que.put(cur.right)
        #
        #     # debug 时辅助打印看一下
        #     self.traverse(linked_list.get_head())
        #
        #     res.append(linked_list.get_head())
        #
        # return res

        # 写法2： 不在外面写LinkedList，直接 接元素
        res = []

        if not root:
            return res

        que = Queue()
        que.put(root)

        while not que.empty():
            n = que.qsize()

            dummy = ListNode(-1)    # 知识点讲解：dummy哨兵节点 往下看
            tail = dummy

            for i in range(n):
                cur = que.get()

                node = ListNode(cur.val)
                tail.next = node
                tail = node   # update tail to the last node

                if cur.left:
                    que.put(cur.left)
                if cur.right:
                    que.put(cur.right)

            # debug 时辅助打印看一下
            self.traverse(dummy.next)

            res.append(dummy.next)

        return res

    # 写这个是debug测试用的
    def traverse(self, node):
        cur = node
        while cur is not None:
            print(cur.val, end=' ')
            cur = cur.next
        print()

'''
知识点讲解： dummy哨兵节点
对链表的操作，头节点，非头节点，是不一样的，往往分开操作
所以会判断是不是头节点 (location是不是等于0)，但这样操作会有点累
所以会在链表前加一个dummy节点，这个dummy节点值是多少都无所谓，但一般写-1（下标-1，不存在），这个值永远用不到
这样的话，再对链表操作，都转换成 对链表中间节点的 操作了 （不对头单独操作了） 
这样代码写起来更统一了，也不用再单独判断头部了
操作完后呢，还要把链表返回，返回谁呢？ 返回 dummy.next 才是真正的原链表
'''





# 测试用
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







# 测试用
if __name__ == '__main__':



    testRoot1 = build_tree()
    sol = Solution()
    list = sol.binaryTreeToLists(testRoot1)
    print(list)



