"""

dfs 和 bfs 其实就是另一种特殊的顺序 for 循环每个节点
但分治法不一般的for循环去打擂台
而是 左一半 中间 右一半 去整

DFS主要是 遍历法+分治法
然后令狐冲讲的好像是说 DFS = Backtracking

什么样的数据结构适合分治法？
数组
二叉树（主要），二叉树的高度是 logN   二叉树问题无非就是分治和遍历法。二叉树一般是欠套结构。

【默认规则】
subtree：n个节点有n个 subtree
叶子节点也算子数的，

【题型】
（1）二叉树上求值，求路径
    Maximum / Minimum/ Average / Sum / Path

二叉树，BST 的高度一般用 O(h)
平衡二叉树/heap/AVL/red black tree 的高度才是 O(logN)     不一定要用 tree map 来做面试题

BST的  前序和中序遍历，非递归版本，要背！！！
后续遍历选择性掌握，但最好掌握。因为分治法，其实就是后续遍历，先处理左右儿子，最后再combine root一起处理
面试中考BST，如果考前序，中序遍历的话，都考非递归写法的

不过一般考分治法的话，就允许用递归了

O(a+b) = O(max(a,b)) 这是个数学结论，记住就行

二叉树的增加/查找/修改必须会，删除不作要求
后续遍历iterative写法 与 Morris 算法 不必须掌握，但学了可以提高 coding 能力
"""

class Solution:

    # 领扣 E 596 · Minimum Subtree 这个教了如何尽量避免使用全局变量，值得好好体会
    def findSubtree(self, root):
        """
        视频讲解是 00:50:00 - 01:05:30
        思路是
        for subtree in all_subtrees:
            min (打擂台)
        用没用到遍历的一个核心点就是有没有用到全局变量，
        但其实全局变量尽可能少使用，因为会导致
        （1）函数不纯粹。最好能影响函数return结果的只有input，不然容易出bug
        （2）不利于多线程化，对共享变量加锁带来效率下降
        因此把它改成纯 分治的写法吧，改的方法
        （1）全局变量 改成 input 参数
        （2）直接把 全局变量放在 return 的 整个结果 里 (变成几个人打擂台就好)
        """
        root_sum, min_sum, min_root = self.helper(root)  # 什么时候要用 helper function？当自己分治的时候跟题目要求return的不一致
        return min_root
    def helper(self, root):
        if not root:
            return 0, float('inf'), None

        left_sum, left_min_sum, left_min_root = self.helper(root.left)
        right_sum, right_min_sum, right_min_root = self.helper(root.right)

        # 就 3 个人在打擂台
        root_sum = left_sum + right_sum + root.val
        # 第一个是整个 parent node 的 tree
        # 第二个是 左边是 left sub-tree
        # 第三个是 右边是 right sub-tree
        current_min_sum = min(root_sum, left_min_sum, right_min_sum)

        if left_min_sum == current_min_sum:
            # return 啥都是越写越发现什么，就补上就好
            return root_sum, current_min_sum, left_min_root

        if right_min_sum == current_min_sum:
            return root_sum, current_min_sum, right_min_root

        return root_sum, current_min_sum, root

    # 领扣 E 474 · Lowest Common Ancestor II 哈希表做法。这道题是 LCA 有简称的题目都是经典题，面试官很可能重新面你的那种！！！
    def lowestCommonAncestorII1(self, root, A, B):
        """
        视频讲解在 01:06:00 - 01:09:35
        思路是A和B两个节点一路往父亲节点走
        先把A所有的沿路父节点加入hashset里，然后B的父节点检查在不在hashset里
        这就是考察一个 hashset 的应用

        如果不能用 hashset 怎么办？看方法2
        因为 hashset 是基于内存的，而 list 是可以基于内存/硬盘的。所以hashset放在硬盘上, 访问起来会毕竟慢
        """
        Set = set()

        # Add A's parents into Set
        temp_parent = A
        while temp_parent:
            Set.add(temp_parent.val)
            temp_parent = temp_parent.parent

        # Iterate B's parent node， its first parent appeared in Set is their common ancestor
        temp_parent = B
        while temp_parent:
            if temp_parent.val in Set:
                return temp_parent
            temp_parent = temp_parent.parent

        return None

    # 领扣 E 474 · Lowest Common Ancestor II  List 做法
    def lowestCommonAncestorII2(self, root, A, B):
        """
        如果不能用 hashset，只能用 list该怎么做？ 视频讲解在 01:06:00 - 01:12:25

        1.A,B 分别利用指向父亲节点的指针反向而上直到空节点，把经历的每一个节点存到两个列表中。
        2.将两个列表，从尾到头，一一比对
        3.同时扫描两个列表，pop出头，直到分叉
        4.返回最后一次相同的那个节点

        如果不能用 parent 指针怎么办？看方法3
        """
        pathA = []
        pathB = []
        node = A
        while node:
            pathA.append(node)
            node = node.parent
        node = B
        while node:
            pathB.append(node)
            node = node.parent

        res = None

        for _ in range(min(len(pathA), len(pathB))):
            if pathA[-1] == pathB[-1]:
                res = pathA.pop()
                pathB.pop()
            else:
                return res

        return res

    # 领扣 E 474 · Lowest Common Ancestor II 如果不能用parent指针，分治法该怎么做   这个方法太几把饶头了
    def lowestCommonAncestorII3(self, root, A, B):
        """
        如果没有 parent 指针该怎么做？ 这道题的前提，是确保了 A 和 B 都在树里
        分治法做  视频讲解 01:18:05 - 01:28:00
        A 和 B 是 reference，是唯一的，因为树里的 reference 是唯一的

        原始思路：
        分治法做的话，不知道分治后return啥，一开始就想return我想要的东西LCA
        所以会想在left/right去找LCA，找到return LCA，没有的话return None
        但这种方式，没办法处理，当A和B在left/right一边一个时会return None，A和B两边都没有也return None，区分不出来这两种情况
        那么信息量就不够了，那就要多return一点东西

        总的思想是：有啥 return 啥
        定义返回值：
        当一个子树
        A B 都有 -> 找到LCA了 -> return LCA
        只有A -> return A
        只有B -> return B
        A B都没 -> return null
        （以上很好理解 因为 A B 在一个树中存在的方式是 都存在，只存在A，只存在B，都不存在）
        """
        if root is None:
            return None

        # case1：特殊情况 A 或 B刚好在根节点
        # 那么这个跟节点就是LCA该return  #    A         B
        if root == A or root == B:   #   / \       / \
            return root              #  B null   null A

        left = self.lowestCommonAncestorII(root.left, A, B)
        right = self.lowestCommonAncestorII(root.right, A, B)

        # case2：如果 left 和 right都有内容
        if left and right:
            '''
            left 和 right 都有内容
            这种情况不可能是 左A   右LCA 这样就出现了2个A
                 也不可能是 左LCA 右B   这样就出现了2个B
                 也不可能是 左LCA 右LCA 这样就出现了2个A和2个B  
            只可能是 A B 两边各一个
            这就说明当前的 root 就是 LCA 
            '''
            return root

        # case3：只有左边有内容，那就有可能左边是 A/B/LCA
        if left:
            return left

        # case4：只有右边有内容，那就有可能右边是 A/B/LCA
        if right:
            return right

        # case5：能走到这，说明left和right都是空，没A也没B，return None
        return None

    # 领扣 E 474 · Lowest Common Ancestor II 如果A和B不保证一定出现在树root里, 也不能用parent指针，怎么做？
    def lowestCommonAncestorII4(self, root, A, B):
        """
        如果A和B不保证一定出现在树root里怎么做？ 视频讲解01:26:51 - 01:30:53

        怎么在只有A和只有B时，确保最后return的是None呢？那就把A和B是否存在的信息也记录下来
        整个树root里面有没有A，等于左子树有没有A，或者右子树有没有A，或根节点就是A

        """
        a, b, lca = self.helper474(root, A, B)
        if a and b:
            # 如果 a 和 b 都存在，那么lca才代表真正的lca
            return lca
        else:
            # 其他只存在 a 或 b 或者 ab都不存在，就return null
            return None
    def helper474(self, root, A, B):
        if not root:
            return None, None, None

        is_a_in_left, is_b_in_left, left_node = self.helper474(root.left, A, B)
        is_a_in_right, is_b_in_right, right_node = self.helper474(root.right, A, B)

        # a 和 b return 的是 boolean type
        is_a_existing = is_a_in_left or is_a_in_right or root == A  # 检测整个树root里面有没有A，等于左子树有没有A，或者右子树有没有A，或根节点就是A
        is_b_existing = is_b_in_left or is_b_in_right or root == B  # 检测整个树root里面有没有A，等于左子树有没有A，或者右子树有没有A，或根节点就是A

        # 接下来是检测，有LCA就优先return LCA，有A就return A，有B就returnB

        # case1: A 或 B 刚好在根节点
        if root == A or root == B:
            return is_a_existing, is_b_existing, root

        # case2: 如果 left 和 right 都有内容, 返回的 root 就是 lca，跟解法3一样
        if left_node and right_node:
            return is_a_existing, is_b_existing, root

        # case3：只有左边有内容，那就有可能左边是 A/B/LCA
        if left_node:
            return is_a_existing, is_b_existing, left_node

        # case4：只有右边有内容，那就有可能右边是 A/B/LCA
        if right_node:
            return is_a_existing, is_b_existing, right_node

        # case5: left_node 和 right_node 都没内容，说明该返回return
        return is_a_existing, is_b_existing, None

    # 领扣 E 453 · Flatten Binary Tree to Linked List 这是很容易写的一个错误版本, 全局变量导致函数不纯粹
    def flatten1(self, root):
        """
        一开始很容易想到先得到 pre-order traversal 的顺序再连成 linked list 但这会耗费O(n)的堆空间
        所以就不如直接 in—place 地做吧

        然后很容易写成下面这种
        错在哪里？  使用了全局变量让函数不纯粹了，容易出bug。实在要这样写的话，正确写法请看方法2
        怎么改？可以把需要修改的变量，作为参数传入到函数里
               或是放在 return value里 （比如领口的596题）
        """
        self.prev_node = None  # 错在：使用了全局变量，很容易搞错的

        def flattening(root):
            if not root:
                return

            if self.prev_node:
                self.prev_node.left = None
                self.prev_node.right = root

            self.prev_node = root
            flattening(root.left)
            # 上步，由于要进到 prev_node.right = root 里导致 root.right变了
            # flattening 不只是跟 input 的参数有关，还跟全局变量 prev_node 有关了，这个递归函数不纯粹了！！！
            # 错在，等走下步的时候 root.right 已经不是原来的root.right了，原来的丢失了
            flattening(root.right)

        flattening(root)

    # 领扣 E 453 · Flatten Binary Tree to Linked List 硬要使用全局变量，的正确版本
    def flatten2(self, root):
        self.prev_node2 = None  # 全局变量

        def flattening(root):
            if not root:
                return

            if self.prev_node2:
                self.prev_node2.left = None
                self.prev_node2.right = root

            self.prev_node2 = root
            # 由于一会儿要进到root.left里, prev_node2.right = root 导致 root.right变了
            # 所以要先把 root.right的信息保存起来，再进到 root.left 里
            right = root.right
            flattening(root.left)
            flattening(right)

        flattening(root)

    # 领扣 E 453 · Flatten Binary Tree to Linked List 分治法做，绕死我脑袋了
    def flatten3(self, root):
        """
        这个方法3比方法2改进的点在于
        （1）依赖return value（不依赖全局变量了，全局变量放return里）
        （2）依赖于局部去处理子树，处理完后，return结果，然后我们去合并这个结果
            分治思路，不是对它进行前序遍历，而是先flatten左边，再flatten右边，再处理合并一下
                                       分治法就是一分为二后，做同样的事情
              1                 1
             / \               / \
            2   5    ->       2   5
           /\   /\             \   \
          3  4 6  7             3   6
                                 \   \
                                  4   7
        建议版的写法，请看方法4
        """
        if root is None:
            return None, None

        # 先 flatten 左边，返回左子树的根，和左子树的尾巴
        left_n, left_tail = self.flatten(root.left)
        # 再 flatten 右边，返回右子树的根，和右子树的尾巴
        right_n, right_tail = self.flatten(root.right)

        if not left_n and not right_n:
            return root, root

        if not left_n and right_n:
            return root, right_tail

        if left_n and not right_n:
            root.right = left_n
            root.left = None
            return root, left_tail

        # 走到这里就说明 有 left_n 和 right_n
        root.right = left_n
        root.left = None
        left_tail.right = right_n

        return root, right_tail

    # 领扣 E 453 · Flatten Binary Tree to Linked List 分治法做，代码更简洁
    def flatten4(self, root):
        """
        这个方法4跟方法3一样的思路，但更简洁
        """
        self.flatten_and_return_tail_node(root)
    def flatten_and_return_tail_node(self, root):
        # 注意这个函数，虽然 flatten 了
        # 但  root  root.left  root.right  这三个相对位置没变，所以其实值需要 return tail就好
        if root is None:
            return None

        # 注意前序遍历，是先 root，再left，再right
        left_sub_tree_tail = self.flatten_and_return_tail_node(root.left)
        right_sub_tree_tail = self.flatten_and_return_tail_node(root.right)

        # 开始 flatten!
        if left_sub_tree_tail:  # 如果left返回的是空的话，说明没有左子树，那就也不需要继续往右子树插了
            left_sub_tree_tail.right = root.right
            root.right = root.left
            root.left = None

        #      这条语句，回返回第一个为真的值
        return right_sub_tree_tail or left_sub_tree_tail or root



        # 把BST的3种遍历方法的 interatve版本写下来

    # 领扣 M 902 · Kth Smallest Element in a BST
    def kthSmallest(self, root, k):
        """
        这个题其实就是做 inorder traversal
        stack里元素弹出的顺序就是 inorder traversal 的顺序
        要求第K大，那就把 K-1 个元素从 stack 里 pop掉，最后 return stack[-1] 栈顶元素就好
        用的 iterative 写法

        时间复杂度是 O(max(k,h)) 因为 当 k=1 时复是O(h), 当 k=n 时复是O(n)

        但 follow up的问题是，如何使它速度，更快呢？
        那我们就要知道 以每个节点为root的subtree有多少个
        所以可以用一个 hashMap<treeNode, Integer> 来存储某个节点为代表的subtree的节点个数
        在增删查改的过过程中 update 受影响的节点的 counter
        然后用 counter 去算该往哪边走（另一边就不走了）
        这样最后的时间复杂度就是 O(h) 了
        """
        if not root: return -1

        self.stack = []
        self.to_left_most(root)

        for _ in range(1, k):
            # 把 K-1 个元素从 stack 里 pop掉
            curr = self.stack.pop()
            if curr.right:
                self.to_left_most(curr.right)

        # 最后 return stack[-1] 栈顶元素就好
        return self.stack[-1].val
    def to_left_most(self, root):
        if not root: return

        while root:
            self.stack.append(root)
            root = root.left

    # 领扣 E 900 · Closest Binary Search Tree Value
    def closestValue(self, root, target):
        """
        通过这个题主要是去了解二叉树的特性
        如果使用 lowerBound / upperBound 的方法，时间复杂度是多少？
        """
        upper = root
        lower = root
        while root:
            # update lower and upper bound
            if target > root.val:
                lower = root
                root = root.right
            elif target < root.val:
                upper = root
                root = root.left
            else:
                return root.val

        if abs(upper.val - target) <= abs(lower.val - target):
            return upper.val

        return lower.val

    # 领扣 H 901 · Closest Binary Search Tree Value II  BST最难也就不过如此了

class Traversal: # iterative 写法

    # lintcode Easy 66 · Binary Tree Preorder Traversal
    def preorderTraversal_iterative(self, root):
        """
        时间O(n)
        空间O(h)  depending on the tree structure, we could keep up to the entire tree, therefore, the space complexity is O(n)
        """
        if root is None:
            return []

        stack = [root]
        res = []

        while stack:
            node = stack.pop()
            res.append(node.val)

            # 因为 stack 是先进后出。要先访问 left 后访问 right，所以 right 应该先进 stack
            if node.right:
                stack.append(node.right)

            # 然后 left 再进 stack (这样才能在 pop 的时候，才能 pop)
            if node.left:
                stack.append(node.left)

        return res

    # lintcode(力扣94) Easy 67 · Binary Tree Inorder Traversal
    def inorderTraversal_iterative(self, root):
        if not root:
            return []

        def visit_left_nodes(root):
            while root:           # 如果 root 是 None 就会退出这个小函数
                stack.append(root)
                root = root.left

        res = []
        stack = []
        visit_left_nodes(root)

        while stack:
            curr = stack.pop()
            res.append(curr.val)
            # 去看一下curr的右子树有没有，有的话 visit_left_nodes(curr.right)
            if curr.right:       # 这句话其实可以没有，但写上，让代码更可读
                visit_left_nodes(curr.right)

        return res

    # lintcode(力扣145) Easy 68 · Binary Tree Postorder Traversal
    def postorderTraversal_iterative(self, root):
        if not root:
            return []

        def to_leftMost_and_to_rightMost(node):
            while node:
                stack.append(node)
                # 先向左走到底，走到底之后，再从左底一路走到右底
                node = node.left if node.left else node.right

        res, stack = [], []
        to_leftMost_and_to_rightMost(root)

        while stack:
            curr = stack.pop()
            res.append(curr.val)
            # 如果 stack非空， curr 是不是 栈顶节点stack[-1] 的左孩子 （如果不是就继续while循环弹栈）
            if stack and stack[-1].left == curr:
                # 栈顶节点stack[-1]，先hold住不出栈
                # curr往 栈顶节点(也就是curr的父节点) 的右孩子走一步
                curr = stack[-1].right
                to_leftMost_and_to_rightMost(curr)

        return res

# Definition of TreeNode:
class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class ParentTreeNode:
    def __init__(self, val):
        self.val = val
        self.parent, self.left, self.right = None, None, None

def build_tree1():
    """
          1
         / \
        2   5
       /\   /\
      3  4 6  7
    """
    node_1 = TreeNode(1)
    node_2 = TreeNode(2)
    node_3 = TreeNode(5)
    node_4 = TreeNode(3)
    node_5 = TreeNode(4)
    node_6 = TreeNode(6)
    node_7 = TreeNode(7)

    node_1.left = node_2
    node_1.right = node_3
    node_2.left = node_4
    node_2.right = node_5
    node_3.left = node_6
    node_3.right = node_7
    node_4.right = TreeNode(8)

    return node_1
def build_tree2():
    """
          1
         / \
            2
       /\   /\
              3
    """
    node_1 = TreeNode(1)
    node_2 = TreeNode(2)
    node_3 = TreeNode(3)

    node_1.right = node_2
    node_2.right = node_3

    return node_1
def build_tree3():
    """
          4
         / \
        2   5
       / \
      1  3
    """
    node_1 = TreeNode(4)
    node_2 = TreeNode(2)
    node_3 = TreeNode(5)
    node_4 = TreeNode(1)
    node_5 = TreeNode(3)

    node_1.left = node_2
    node_1.right = node_3
    node_2.left = node_4
    node_2.right = node_5

    return node_1

if __name__ == '__main__':

    root = build_tree3()
    sol1 = Solution()
    sol2 = Traversal()
    l = sol1.kthSmallest(root, 3)
    print(l)
