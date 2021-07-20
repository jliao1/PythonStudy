# Definition of TreeNode:

from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

class Solution:

    # lintcode(力扣110) Easy 93 ·判断tree是否balanced Balanced Binary Tree 这题我觉得挺生疏的。用DFS做的，跟 maxDepth 有异曲同工之妙
    def isBalanced1(self, root):
        """
        九章令狐冲的答案
        对每个DFS，计算以每个node的左右子树的高度，顺便判断左右子树高度是否差≤1
        这个解法是利用，dfs可以一边遍历每个节点，一边计算每个节点（和它子节点）的高度

        由于也是用dfs做的，所以时间复杂度是O(n),空间复杂度O(h)
        """
        is_balanced, _ = self.helper_for_isBalanced1(root)
        return is_balanced
    def helper_for_isBalanced1(self, root):
        if not root:
            return True, 0

        is_left_balanced, left_height = self.helper_for_isBalanced1(root.left)
        is_right_balanced, right_height = self.helper_for_isBalanced1(root.right)

        root_height = max(left_height, right_height) + 1

        if not is_left_balanced or not is_right_balanced:
            return False, root_height

        if abs(left_height - right_height) > 1:
            return False, root_height

        return True, root_height

    # Lintcode(力扣104) Easy 97 · Maximum Depth of Binary Tree 其实就是求二叉树高度
    def maxDepth1(self, root):
        """
        借助递归函数参量
        思路是，求出所有节点的深度，然后取个最大值
        时间复杂度是O(n)， 空间复杂度是 O(h)
        """
        self.maxD = 0
        self.dfsForMaxDepth1(root, 1)
        return self.maxD
    def dfsForMaxDepth1(self, root, height):
        if not root:
            return

        self.maxD = max(self.maxD, height)

        self.dfsForMaxDepth1(root.left, height + 1)
        self.dfsForMaxDepth1(root.right, height + 1)

    # Lintcode(力扣104) Easy 97 · Maximum Depth of Binary Tree 一句话搞定的递归写法
    def maxDepth2(self, root):
        if root is None:
            return 0

        return max(self.maxDepth2(root.left), self.maxDepth2(root.right)) + 1

    # Lintcode(力扣104) Easy 97 · Maximum Depth of Binary Tree 自己想的思路，不高级但还是写一下
    def maxDepth3(self, root):
        """
        这个解法不是最好的。但这是我一开始想的思路
        就是算到每个 叶子节点的高度，然后取个最大值（不像方法1那样每个节点都取最大值）
        """
        self.maxD = 0
        self.dfsForMaxDepth3(root, 1)
        return self.maxD
    def dfsForMaxDepth3(self, root, height):
        if not root:  # 这句不能省的，因为有的节点，虽然不是叶子节点，但有可能有1个孩子，进到另一个孩子时root是Noone的
            return
        if not root.left and not root.right:
            self.maxD = max(self.maxD, height)
            return
        self.dfsForMaxDepth3(root.left, height + 1)
        self.dfsForMaxDepth3(root.right, height + 1)

    # Lintcode(力扣111) Easy 155 · Minimum Depth of Binary Tree 自己最开始的写法用dfs，不高级但思路清晰
    def minDepth1(self, root):
        if not root:
            return 0

        self.minD = float('inf')
        self.dfsForMinDepth1(root, 1)

        return self.minD
    def dfsForMinDepth1(self, root, depth):
        if not root:
            return 0

        if not root.left and not root.right:
            self.minD = min(self.minD, depth)
            return

        self.dfsForMinDepth1(root.left, depth + 1)
        self.dfsForMinDepth1(root.right, depth + 1)

    # Lintcode(力扣111) Easy 155 · Minimum Depth of Binary Tree 力扣上的第一个solution写法，写法清奇，可以参考部分
    def minDepth2(self, root):
        if not root:
            return 0

        # 接下来3行这么写，可读性挺高，判断如果左右孩子都没有
        children = [root.left, root.right]
        if not any(children):  # if we're at leaf node
            return 1

        min_depth = float('inf')

        for c in children:
            if c:
                min_depth = min(self.minDepth2(c), min_depth)

        return min_depth + 1

        # Lintcode Easy 482 · Binary Tree Level Sum 自己用BFS做的

    # Lintcode(力扣111) Easy 155 · Minimum Depth of Binary Tree 九章的一个答案
    def minDepth3(self, root):
        """
        可以大概理解成4种情况
         1. 当前节点有左右子树，分别计算左右子树的minimum depth，返回其中最小值 + 1
         2. 当前节点只有左子树，返回左子树的minimum depth + 1
         3. 当前节点只有右子树，返回右子树的minimum depth + 1
         4. 当前节点没有左右子树（叶子节点），返回1

        我们知道，在求最大深度时，递归条件是：每个节点的深度等于其左右子树最大深度值加上 1。
        在求最小深度时，如果类比过来，令每个节点的深度等于其左右子树最小深度值加上 1，这样做是不对的。
          tree = 1,2,#,3   以此为例，这是三层的树，
                /          最小深度应该是3
               2           但如果按照上述做法
              /            节点1的右子树为空
             3             我们会得出节点1的最小深度是1的结论，跟答案不符。
                           正确做法是要判断一下左右子树是否有空子树，如果有，那么最小深度等于另一颗子树的深度加1。

        时间复杂度O(n)  空间只考试call stack与树高度有关，最坏O(N)最好O(logN)
        """
        if root is None:
            return 0
        leftDepth = self.minDepth3(root.left)
        rightDepth = self.minDepth3(root.right)
        # 当左子树或右子树为空时，最小深度取决于另一颗子树
        if leftDepth == 0 or rightDepth == 0:
            return leftDepth + rightDepth + 1
        return min(leftDepth, rightDepth) + 1

    # Lintcode(力扣111) Easy 155 · Minimum Depth of Binary Tree 用BFS做的
    def minDepth4(self, root):
        """
        这题其实比较适合用 BFS 做（使用 DFS 会需要访问所有的子树并比较长度, 稍慢了一些）
        採用 level-order BFS, 记录目前为止树的深度
        只要遇到 leaf node 就 return
        """
        if not root:
            return 0

        # create queue to store nodes in current level
        queue = deque([root])
        depth = 0

        # BFS
        while queue:
            depth += 1
            for _ in range(len(queue)):
                node = queue.popleft()

                # if node is leaf node
                if not node.left and not node.right:
                    return depth

                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

    # Lintcode Easy 482 · Binary Tree Level Sum 用BFS做的
    def levelSum1(self, root, level):
        """
        用 BFS 做的，时间复杂度固定是O(n), 空间复杂度是 O(n)
        我这方法是遍历所有的，存所有层的，好处是想看哪层，改变一下input的level就行
        方法2是用DFS做的，时间空间复杂度和 level 相关(level更大的层的nodes不需要遍历了)
        """
        if not root:
            return 0

        all_level_lists = []

        q = deque()

        q.append(root)

        while len(q) != 0:

            cur_level_list = []
            n = len(q)

            for _ in range(n):
                cur_node = q.popleft()
                cur_level_list.append(cur_node.val)

                if cur_node.left:
                    q.append(cur_node.left)
                if cur_node.right:
                    q.append(cur_node.right)

            if cur_level_list:
                all_level_lists.append(cur_level_list)

        # 这个很容易忘记
        return sum(all_level_lists[level - 1]) if level <= len(all_level_lists) and level >= 1 else 0

    # Lintcode Easy 482 · Binary Tree Level Sum 用DFS做的
    def levelSum2(self, root, level):
        """
        这是用DFS和divide and conquer做的
        为什么可以用DFS做？ 好像看 左右 child nodes 的，都可以用DFS
        """
        if root is None:
            return 0
        elif level == 1:
            return root.val
        return self.levelSum2(root.left, level-1) + self.levelSum2(root.right, level-1)

    # Lintcode(力扣623) Medium 1122 · Add One Row to Tree 自己用BFS做的
    def addOneRow1(self, root, val, depth):
        if depth == 1:
            new_root = TreeNode(val)
            new_root.left = root
            return new_root

        q = deque()
        q.append(root)
        cnt = 1

        # BFS 在这里
        while cnt <= depth - 2:
            length = len(q)
            for _ in range(length):
                cur = q.popleft()
                if cur.left:
                    q.append(cur.left)
                if cur.right:
                    q.append(cur.right)
            cnt += 1

        # 开始处理目标层
        while q:
            cur = q.popleft()

            new1 = TreeNode(val)
            new2 = TreeNode(val)

            if cur.left:
                new1.left = cur.left

            cur.left = new1

            if cur.right:
                new2.right = cur.right

            cur.right = new2

        return root

    # Lintcode(力扣623) Medium 1122 · Add One Row to Tree 九章答案用dfs做的
    def addOneRow2(self, root, v, d):
        """
        decrease and conquer的 dfs 方法
        当 d == 2的时候，就可以连接新node了

        时间空间复杂度与方法1一样的
        """
        if not root:
            return None
        if d==1:
            new_root = TreeNode(v)
            new_root.left = root
            return new_root
        if d==2:
            root.left, root.left.left = TreeNode(v), root.left
            root.right, root.right.right = TreeNode(v), root.right
            return root
        elif d>2:
            root.left = self.addOneRow2(root.left, v, d - 1)
            root.right = self.addOneRow2(root.right, v, d - 1)
        return root

    # Lintcode(力扣199) Medium 760 · Binary Tree Right Side View  用BFS做的
    def rightSideView1(self, root):
        """
        方法1的思路比较常规，是：BFS，分层遍历，取每层的最后一个放进list里就好
        """
        if not root:
            return []
        ans = []
        # 一般要用队列的数据结构，但是由于这个代码不需要pop，所以直接用list啦，直接用list不需要导包
        queue = [root]
        while queue:
            next_q = []
            for index, cur in enumerate(queue):
                if index == len(queue) - 1:
                    ans.append(cur.val)
                if cur.left:
                    next_q.append(cur.left)
                if cur.right:
                    next_q.append(cur.right)
            # 由于这句话的使用，上面不需要用队列的数据结构啦
            queue = next_q
        return ans

    # Lintcode(力扣199) Medium 760 · Binary Tree Right Side View 用DFS的赶脚
    def rightSideView2(self, root):
        """
        这个思路  愣是把DFS用出BFS的赶脚
        不如方法1好想到
        """
        def collect(node, depth):
            if node:
                # 如果深度刚好等于res的长度，说明是这层深度放进去的第一个value
                if depth == len(res):
                    res.append(node.val)

                # 而这里先遍历右边才左边，相同深度下，第一个遍历到的元素，是每层最右边的元素
                collect(node.right, depth + 1)
                collect(node.left, depth + 1)
        res = []
        collect(root, 0)
        return res

    # Lintcode hard 87 · Remove Node in Binary Search Tree  主要是 edge case太多，太烦了
    def removeNode1(self, root, value):
        """
        这是sheila教的方法，
        先找到 value 的 parent 和这个 value
        然后看要删除的 target node 是不是 叶子节点，或之有1个孩子，或有2个孩子，分这3种情况讨论
        写出来累死了…  我写得不太好的地方在于可能很多地方重复判断了？
        可以看下方法2 iterative 写法要简练点，把remove都整合到一个函数里了
        方法3 比 方法2 更简洁一些
        """
        dummy = TreeNode(-float('inf'))
        dummy.right = root
        target_node, parent = self.find_target_node_and_its_parent(dummy, value)

        # find nothing，doing nothing
        if not parent and not target_node:
            return root

        # target node is a leaf node
        if not target_node.left and not target_node.right:
            self.remove_leaf_node(parent, target_node)
        # target node only has one child
        elif target_node.left and not target_node.right:
            self.remove_node_with_one_child(parent, target_node)
        elif target_node.right and not target_node.left:
            self.remove_node_with_one_child(parent, target_node)
        # target node has 2 children
        elif target_node.right and target_node.left:
            self.remove_node_with_2_children(target_node)

        return dummy.right
    def remove_node_with_2_children(self, target):
        """
        策略是 find inorder predecessor
        从 target node 的位置开始，先go left once，然后 go as right as possible
        就可以找到 target 的 inorder predecessor
        然后把 inorder predecessor 位置的 value 和 target 位置的 value swap
        然后删掉 inorder predecessor 位置的 node
        """
        # go left once
        curr = target.left
        # go as right as possible
        curr_prev = target
        while curr.right:
            curr_prev = curr
            curr = curr.right
        # find target's inorder_predecessor
        inorder_predecessor = curr
        inorder_predecessor_parent = curr_prev
        # swap
        inorder_predecessor.val, target.val = target.val, inorder_predecessor.val
        # delete inorder_predecessor
        if inorder_predecessor.left:
            # inorder_predecessor has one child
            self.remove_node_with_one_child(inorder_predecessor_parent, inorder_predecessor)
        else:
            # inorder_predecessor has no child
            self.remove_leaf_node(inorder_predecessor_parent, inorder_predecessor)
    def remove_node_with_one_child(self, parent,  target):
        # 要删除的 target node 如果只有一个孩子，那么直接让 target 的 parent 指向 target 的唯一孩子
        if target.left:
            # 如果 target 唯一的孩子是左孩子
            target_child = target.left
        else:
            # 如果 target 唯一的孩子是右孩子
            target_child = target.right
        if parent.left and parent.left == target:
            # 如果 parent 的左孩子 是 target node
            parent.left = target_child
        else:
            # 如果 parent 的右孩子 是 target node
            parent.right = target_child
    def remove_leaf_node(self, parent, target):
        # 要删除的 target node 如果是 叶子节点，那么直接让 target 的 parent 指向 None
        if parent.left and parent.left == target:
            # 如果 target node 是 parent 的左孩子
            parent.left = None
        else:
            # 那么 target node 是 parent 的左孩子
            parent.right = None
    def find_target_node_and_its_parent(self, root, value):
        if not root:
            return None

        if root.left:
            if root.left.val == value:
                return root.left, root
            elif value < root.val:
                return self.find_target_node_and_its_parent(root.left, value)

        if root.right:
            if root.right.val == value:
                return root.right, root
            elif value >= root.val:
                return self.find_target_node_and_its_parent(root.right, value)

        return None, None

    # Lintcode hard 87 · Remove Node in Binary Search Tree 这个思路和方法1一样，但是iterative写法，写得比方法1简练点
    def removeNode2(self, root, value):
        # # 这个部分是处理null case 不过其实以下这2句不写也是可以的，因为 check parent of the node to delete
        # #            时已经处理当parent是None的情况了（也就是cur指向root，但由于root是None导致parent也是None根本没移动）
        # if root is None:
        #     return root

        # 这个部分是find the node to delete and its parent
        parent, target = None, root
        # 找cur把它指向我们想要的value node
        # 循环停止条件是：找到 target.val 就是 value 停止
        #              或者 target 是 None 为止 (cur如果是None了说明tree里根本不存在value)
        # 但有2种情况根本就不会进入while循环：
        #         (1) 但如果 root是None ，那么target一开始也就是None
        #         (2) value 就在 root
        #         这两种情况，由于根本不进入while循环，那么parent = None
        while target and target.val != value:
            parent = target
            if target.val > value:
                target = target.left
            else:
                target = target.right

        # 这部分开始 remove 了，remove()返回的其实都是删除了target后上位的节点，正好可以连到 parent 下，
        # 其实 check parent是不是None, 就是在处理 root 是 null case 的情况（就是最开始2行代码）
        #                            或者 value 就在 root 为止找到的情况，删除 target = root 根节点
        if parent is None:
            return self.remove_helper(target)
        # 上面说了，如果tree里没找到value，cur会是None，parent.left就是None，就相当于调用self.remove(None)，remove函数也只会返回个None，不会对结果有啥影响
        # 如果找到value了，要么 parent左孩子是 target value node，要么右孩子是，就删除它
        elif parent.left is target:
            parent.left = self.remove_helper(target)
        else:
            parent.right = self.remove_helper(target)

        return root
    def remove_helper(self, target):
        # case1：target is null: 就是上面说的 根本没找到value的情况，target就会是None，再返回target相当于啥也没干
        if target is None:
            return target

        # case2: if target is leaf
        if target.left is None and target.right is None:
            return None   # 返回 None 因为是让 parent 的左孩子/右孩子 连上 None

        # case3: if target has only one child
        if target.left is None:  # 如果左孩子不存在，那就返回右孩子，好让parent上target的右孩子
            return target.right
        if target.right is None:  # 如果右孩子不存在，那就返回左孩子，好让parent上target的左孩子
            return target.left

        # case4：if target has 2 children, find max node in left subtree（target 的 inorder predecessor）
        prev, cur = None, target.left  # 因为反正target也是有2孩子的，cur先向target的左走1次
        while cur.right:               # 然后 cur go as right as possible
            prev = cur
            cur = cur.right
        # 走到这里说明 cur 已指向了 inorder predecessor

        # make inorder predecessor as new root (move the pointer not copy value)
        # 这步好难想……
        cur.right = target.right
        if target.left is not cur:  # 如果这条语句是True，说明cur除了往左target左走了一步后还向右下继续走了几步
            prev.right = cur.left
            cur.left = target.left
        return cur

    # Lintcode hard 87 · Remove Node in Binary Search Tree 这个思路是很棒的分治法体现！！也是解这题最简单的方法
    def removeNode3(self, root, value):
        """
        这个方法3其实比方法1和2更简练些
        recursion去找要删除的node，找到后直接进行删除操作，再返回连接上原parent
        O(h) time, O(h) space, h: height of tree
        """
        # null case
        if root is None:
            return root

        # check if node to delete is in left/right subtree
        if value < root.val:
            root.left = self.removeNode3(root.left, value)
        elif value > root.val:
            root.right = self.removeNode3(root.right, value)
        else:  # 说明 root.val == value 找到了！就开始做删除root的操作了！
            # if root is has 2 children
            if root.left and root.right:
                left_max = self.find_Max_of_left_tree(root)
                # 把 left_max.val 的值 给 root.val 的位置
                root.val = left_max.val
                # 然后从 root 的 left tree 进入去删掉 这个 left_max
                root.left = self.removeNode3(root.left, left_max.val)
            # if root has only one child
            elif root.left:
                root = root.left
            elif root.right:
                root = root.right
            # if root has no child, it's leaf node
            else:
                root = None

        return root
    # find max node in left subtree of root，就是找删除target的 inorder_predecessor
    def find_Max_of_left_tree(self, target):
        # 反正target也有2个孩子，向左移动1次
        node = target.left
        # 然后go right as possible
        while node.right:
            node = node.right
        return node

    # Lintcode hard 87 · Remove Node in Binary Search Tree
    def removeNode4(self, root, value):
        """
        这个思路是对原列表左一次inorder traversal 但是要删除 value 后生成一个 list
        然后依据这个list 重建一个(高度尽量小的) 二叉树
        但这种可能不太符合题目说的，因为题目要求是，没找到value的话就doing nothing
        """
        self.ans = []
        self.inorder_traversal_list(root, value)
        return self.build_tree_from_inorder_traversal(0, len(self.ans) - 1)
    def inorder_traversal_list(self, root, value):
        if root is None:
            return

        self.inorder_traversal_list(root.left, value)
        if root.val != value:
            self.ans.append(root.val)
        self.inorder_traversal_list(root.right, value)
    def build_tree_from_inorder_traversal(self, l, r):
        if l == r:
            node = TreeNode(self.ans[l])
            return node

        if l > r:
            return None

        mid = (l + r) // 2
        node = TreeNode(self.ans[mid])
        node.left = self.build_tree_from_inorder_traversal(l, mid - 1)
        node.right = self.build_tree_from_inorder_traversal(mid + 1, r)
        return node

# 建立个二叉树，测试用
def build_tree1():
    """
            8
           /  \
          3    10
         / \     \
        1   6     14
            /\      \
           4  7     12
          /\   \
       3.5 4.5  7.5
    """
    node_1 = TreeNode(8)
    node_2 = TreeNode(3)
    node_3 = TreeNode(10)
    node_4 = TreeNode(1)
    node_5 = TreeNode(6)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(12)
    node_10 = TreeNode(3.5)
    node_11 = TreeNode(7.5)
    node_12 = TreeNode(4.5)

    node_1.left = node_2
    node_1.right = node_3

    node_2.left = node_4
    node_2.right = node_5

    node_3.right = node_6

    node_5.left = node_7
    node_5.right = node_8

    node_6.right = node_9

    node_7.left = node_10
    node_7.right = node_12
    node_8.right = node_11

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

    node_1.left = node_2
    node_1.right = node_3
    node_3.left = node_4
    node_3.right = node_5

    return node_1
def build_tree3():
    """
        0
       / \
     -1   1
    """
    node_1 = TreeNode(0)
    node_2 = TreeNode(-1)
    node_3 = TreeNode(1)
    node_4 = TreeNode(4)
    node_5 = TreeNode(5)
    node_6 = TreeNode(14)
    node_7 = TreeNode(4)
    node_8 = TreeNode(7)
    node_9 = TreeNode(13)

    node_1.left = node_2
    node_1.right = node_3

    return node_1

if __name__ == '__main__':
    root = build_tree1()
    sol = Solution()
    l = sol.removeNode3(root, 4)
    print(l)

    n1 = TreeNode(1)
    n2 = TreeNode(2)
    n3 = TreeNode(3)
    n1.left = n2
    n1.right = n3
    n2 = None

    pass


    pass