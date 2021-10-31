class Solution:
    # 力扣ME 921. Minimum Add to Make Parentheses Valid
    def minAddToMakeValid(self, s: str) -> int:
        stack = []
        for char in s:
            if char == ')' and stack and stack[-1] == '(':
                stack.pop()
                continue

            if char in '()':
                stack.append(char)
        return len(stack)

    # 力扣MM 1249 Minimum Remove to Make Valid Parentheses
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        for i in range(len(s)):

            if s[i] == ')' and stack and s[stack[-1]] == '(':
                stack.pop()
                continue

            if s[i] in '()':
                stack.append(i)

        S = list(s)
        for i in reversed(stack):
            S.pop(i)
        """
        以上两句也可以写成 这样更高效 
        while stack:
            S[stack.pop()] = ''
        """

        return ''.join(S)

    # 力扣EM 680 Valid Palindrome II
    def validPalindromeII(self, s: str) -> bool:
        left = 0
        right = len(s) - 1

        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return self.isValid680(s, left+1, right) or self.isValid680(s, left, right-1 )

        return True
    def isValid680(self, s: str, left:int, right:int) -> bool:
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -= 1
            else:
                return False

        return True

    # 力扣ME 1762 Buildings With an Ocean View
    def findBuildings(self, heights: "List[int]") -> "List[int]":
        current_max_height = -float('inf')
        List = []
        for i in range(len(heights) - 1, -1, -1):
            height = heights[i]
            if height > current_max_height:
                current_max_height = height
                List.append(i)

        return reversed(List)

    # 力扣EE 938 Range Sum of BST 用BFS做的
    def rangeSumBST(self, root: 'Optional[TreeNode]', low: int, high: int) -> int:
        from collections import deque
        result = 0

        if not root:
            return result

        queue = deque([root])

        while queue:
            node = queue.popleft()

            if low <= node.val <= high:
                result += node.val

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result

    # 力扣EM 953. Verifying an Alien Dictionary 把字符顺序->数字顺序
    def isAlienSorted(self, words: 'List[str]', order: str) -> bool:
        # define alien language order with hashmap
        char_numOrder = {char: i for i, char in enumerate(order)}

        # 把字符顺序->数字顺序
        words_numberOrder = []
        for word in words:
            word_number_order = []
            for char in word:
                number_order = char_numOrder[char]
                word_number_order.append(number_order)
            words_numberOrder.append(word_number_order)

        # 转成数字顺序后，sort后，比较一下
        sorted_words_order = sorted(words_numberOrder)
        return sorted_words_order == words_numberOrder

    # 力扣MM 1650 Lowest Common Ancestor of a Binary Tree III 有父节点的。用hashset做
    def lowestCommonAncestor(self, A: 'Node', B: 'Node') -> 'Node':
        Set = set()

        # Add A's parents into Set
        temp_parent = A
        while temp_parent:
            Set.add(temp_parent)
            temp_parent = temp_parent.parent

        # Iterate B's parent node， its first parent appeared in Set is their common ancestor
        temp_parent = B
        while temp_parent:
            if temp_parent in Set:
                return temp_parent
            temp_parent = temp_parent.parent

        return None

    # 力扣M 236 Lowest Common Ancestor of a Binary Tree 没父指针，A和B保证出现在tree里，分治法该怎么做？   这个方法太几把饶头了.可以看方法2
    def lowestCommonAncestorII1(self, root, A, B):
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

        # case1：特殊情况 A 或 B刚好在根节点  #    A         B       C  这三种情况都是可以处理的
        if root == A or root == B:        #   / \       / \     /\
            return root                   #  B null   null A   A  B

        left = self.lowestCommonAncestorII1(root.left, A, B)
        right = self.lowestCommonAncestorII1(root.right, A, B)

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

        # case3：只有左边有啥return啥 (有可能是A或B或LCA)
        if left:
            return left

        # case4：只有右边有啥return啥 (有可能是A或B或LCA)
        if right:
            return right

        # case5：能走到这，说明left和right都是空，没A也没B，return None
        return None

    # 力扣MM 426 · Convert Binary Search Tree to Sorted Doubly Linked List 核心思想是中序recursive写法  想通了之后其实还是挺简单的
    def treeToDoublyList(self, root):
        """
        Each node in a doubly linked list has a predecessor and successor

        The left pointer 指向前 of the tree node should point to its predecessor,
        and the right pointer 指向后 should point to its successor

        这是我自己左的，感觉挺好
        """
        if not root:
            return root

        head, tail = self.dfs_treeToDoublyList(root)

        # 连接首尾
        tail.right = head
        head.left = tail

        return head
    def dfs_treeToDoublyList(self, root): # 中序recursive写法
        """
        是中序遍历，
        思路：
        由于进到右子树，需要返回右子树变成双链表的头 给上一层级用
            进到左子树，需要返回左子树变成双链表的尾 给上一层级用
        然后又要一直记录 双链表的最前端返回给主函数用
        而且在主函数内也要连接整个双链表的 最头 最尾
        所以这个函数就 返回 头 尾 一直记录着好啦
        """
        # 它俩要先赋值，因为如果 下面2个if语句都没进入执行，它俩得有个初始值
        head, tail = root, root

        # 如果左孩子存在
        if root.left:
            # dfs 返回的 后续会用到 sub_left_tail 接回 root
            head, left_tree_tail = self.dfs_treeToDoublyList(root.left)
            # 连上（处理root左边情况）
            left_tree_tail.right = root
            root.left = left_tree_tail

        # 如果右孩子存在
        if root.right:
            # dfs 返回的 后续会用到 sub_right_head 接回 root
            right_tree_head, tail = self.dfs_treeToDoublyList(root.right)
            # 连上（处理root右边情况）
            right_tree_head.left = root
            root.right = right_tree_head

        return head, tail

    # 力扣ME 314. Binary Tree Vertical Order Traversal 脑筋急转弯
    def verticalOrder(self, root: 'Optional[TreeNode]') -> 'List[List[int]]':
        if not root:
            return []

        from collections import deque

        q = deque([(root, 0)])
        num_nodes = {}

        while q:
            node, num = q.popleft()

            num_nodes.setdefault(num, [])
            num_nodes[num].append(node.val)

            if node.left:
                q.append((node.left, num - 1))

            if node.right:
                q.append((node.right, num + 1))

        return [num_nodes[i] for i in sorted(num_nodes)]
        #                   这一步是会把dict的key来sort一下组成一个list，比如
        #                       dictionary = { 1 : 1.1, 2 : 2.5, 0 : 0.5}
        #                       List = sorted(dictionary) # = [0,1,2]
        # 这样就可以按key的大小提取corresponding value

    # 力扣MM 560 Subarray Sum Equals K 前缀和 + hashmap 先找再往map里放,就不会重复查找了
    def subarraySum(self, nums: 'List[int]', k: int) -> int:
        """时间空间都是O(n)"""
        prefix_sum = 0
        prefixSum_indicesList = {prefix_sum: [-1]}  # 为什么value是个list?因为prefix可能有相同好几个
        # 这题用不到 sub_arrays = []  # 装所有的sub_array的index范围
        count = 0

        for i, num in enumerate(nums):
            prefix_sum += num
            target = prefix_sum - k
            # 先找
            if target in prefixSum_indicesList:
                count += len(prefixSum_indicesList[target])
                # for index in prefixSum_indicesList[target]:
                #     sub_arrays.append((index+1,i))  # append一个tuple(start_index, end_index)
            # 再往hashmap里方
            prefixSum_indicesList.setdefault(prefix_sum, [])
            prefixSum_indicesList[prefix_sum].append(i)

        return count

    # 力扣ME 973. K Closest Points to Origin
    def kClosest(self, points: 'List[List[int]]', k: int) -> 'List[List[int]]':
        min_heap = []
        from heapq import heappush, heappop
        for x, y in points:
            heappush(min_heap, (x * x + y * y, x, y))

        result = []
        for _ in range(k):
            distance, x, y = heappop(min_heap)
            result.append([x, y])

        return result

    # 力扣215 Kth Largest Element in an Array
    def findKthLargest(self, nums: 'List[int]', k: int) -> int:
        if not nums or k < 1 or k > len(nums):
            return None
        # (1) Kth Largest Element 为了方便编写代码，这里将第k大转成第 [len(A) - k]小问题。  比如 1,3,4,2 第1大就是第index=3th smallest小的数字(从0开始算)
        return self.partition(nums, 0, len(nums) - 1, len(nums) - k)
        # (2) 如果是 Kth Smallest Numbers  比如 1,2,3,4,5  k = 3 就是返回大小该在index=2的数字
        # return self.partition(A, 0, len(A) - 1, k-1)
    def partition(self, nums, start, end, k):
        if start == end: # 也可以写 start >= end
            # 说明找到了
            return nums[k] # 也可以写成 nums[start] 或 nums[end]

        left, right = start, end
        pivot = nums[(start + end) // 2]
        while left <= right:
            while left <= right and nums[left] < pivot:
                left += 1
            while left <= right and nums[right] > pivot:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left, right = left + 1, right - 1

        # 情况1
        if k <= right:
            # start - right 区间都小于等于 pivot
            return self.partition(nums, start, right, k)
        # 情况2
        if k >= left:
            # left - end 区间都大于等于 pivot
            return self.partition(nums, left, end, k)

        # 情况3: 有可能 right 和 left 中间隔了一个数: left < k < right 这个数就刚好是我们要找的数
        return nums[k]

    # 力扣ME 199. Binary Tree Right Side View
    def rightSideView(self, root: 'Optional[TreeNode]') -> 'List[int]':
        if not root:
            return []
        ans = []

        from collections import deque
        queue = deque([root])
        while queue:
            size = len(queue)
            for index in range(size):
                cur = queue.popleft()
                # 当是 从右边看第一个元素时
                if index == size - 1:
                    ans.append(cur.val)

                if cur.left:
                    queue.append(cur.left)
                if cur.right:
                    queue.append(cur.right)

        return ans

    # 力扣MH 227. Basic Calculator II 本题不包含括号和非负数
    def calculate(self, s: str) -> int:
        stack = []
        curr_num = 0
        last_operator = '+'  # 初始值

        for i, char in enumerate(s):
            # case1：c是数字，处理下
            if char.isdigit():
                curr_num = 10 * curr_num + int(char)

            # case2：如果遇到的，不是数字也不是空格(那就是那些 加减乘除符号了)
            #        或 i 已经扫到 最后一位，哪怕char是digit，也要处理
            #        这时候开始处理
            if (not char.isdigit() and not char.isspace()) or i == len(s) - 1:
                # case2.1 遇到 + - 时，连上 operator proceeding it 再入栈
                if last_operator == '+':
                    stack.append(+ curr_num)
                elif last_operator == '-':
                    stack.append(- curr_num)

                # case2.2 遇到 * / 时 弹栈顶数字，与currNum计算后(因为*/优先级高)，把结果入栈
                elif last_operator == '*':
                    stack.append(stack.pop() * curr_num)
                elif last_operator == '/':
                    # (-3//4) 等于 -1, 所以要用 int(-3/4) 才能等于0
                    stack.append(int(stack.pop() / curr_num))

                # 把 lastOperator 更新成本轮遇到测char
                last_operator = char
                # 处理完这一轮了，currNum要清零
                curr_num = 0

        return sum(stack)

    # 力扣EH 415. Add Strings 字符串int处理转换
    def addStrings(self, num1: str, num2: str) -> str:
        res = []

        i1 = len(num1) - 1
        i2 = len(num2) - 1
        flag = 0
        while i1 >= 0 or i2 >= 0:
            a = ord(num1[i1]) - ord('0') if i1 >= 0 else 0
            i1 = i1 - 1

            b = ord(num2[i2]) - ord('0') if i2 >= 0 else 0
            i2 = i2 - 1

            Sum = a + b + flag
            flag = Sum // 10  # 进位
            res.append(str(Sum % 10))

        if flag != 0:
            res.append(str(flag))

        return ''.join(reversed(res))
    """
    【二进制和十进制互换】
    to_binary = bin(5)             # 出来是个string: '0b101'
    to_decimal = int(to_binary, 2) # 出来时个十进制的int: 5
    """

# 力扣ME 1570 Dot Product of Two Sparse Vectors 这个是不太efficent的做法
class SparseVector1:
    def __init__(self, nums: 'List[int]'):
        self.array = nums

    def dotProduct(self, vec:'SparseVector1'):
        result = 0
        for num1, num2 in zip(self.array, vec.array):  # 这个zip是return an iterator of tuples
            result += num1 * num2
        return result

# 力扣ME 1570 Dot Product of Two Sparse Vectors 去除了不是0的计算
class SparseVector2:
    """
    这个方法比方法1提升的点是，去除了不是0的计算
    Time complexity: O(n)O(n) for creating the Hash Map; O(L)O(L) for calculating the dot product.

    Space complexity: O(L)O(L) for creating the Hash Map,
    as we only store elements that are non-zero. O(1)O(1) for calculating the dot product.

    """
    def __init__(self, nums: 'List[int]'):
        self.index_nonZero = {}
        for i, n in enumerate(nums):
            if n != 0:
                self.index_nonZero[i] = n

    def dotProduct(self, vec: 'SparseVector2') -> int:
        result = 0
        # iterate through each non-zero element in this sparse vector
        # update the dot product if the corresponding index has a non-zero value in the other vector
        for i, n in self.index_nonZero.items():
            if i in vec.index_nonZero:
                result += n * vec.index_nonZero[i]
        return result

# Definition for a binary tree node.
class TreeNode:  # used by 力扣938
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Node:  # used by 力扣1650
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None

if __name__ == '__main__':

    dictionary = { 1 : 1.1, 2 : 2.5, 0 : 0.5}
    List = sorted(dictionary) # = [0,1,2]
    sol = Solution()
    res = sol.addStrings('3', '4')

    print(res)