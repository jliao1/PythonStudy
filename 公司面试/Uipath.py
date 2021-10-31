def how_to_use_nonlocal():
    def inner():
        nonlocal counter
        while counter:
            print(counter)
            counter -=1
    counter = 5
    inner()

gloable_variable = 0
def how_to_use_global_variable():
    global gloable_variable
    gloable_variable += 1
    print(gloable_variable)

# 力扣E 349 Intersection of Two Arrays
def set_intersection(set1, set2):
    return [x for x in set1 if x in set2]
def intersection(nums1, nums2):
    """
    测试案例1：
    Input: nums1 = [1,2,2,1], nums2 = [2,2]
    Output: [2]

    测试案例2：
    Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
    Output: [9,4]

    时间空间复杂度是 O(m+n) m和n是两个list的长度
    """
    set1 = set(nums1)
    set2 = set(nums2)

    if len(set1) < len(set2):
        return set_intersection(set1, set2)
    else:
        return set_intersection(set2, set1)

# Leetcode Easy 20. Valid Parentheses
def isValid(s):
    """
    测试一下
    Input: s = "()[]{}"
    Output: true

    Input: s = "([)]"
    Output: false
    """
    if not s:
        return True

    stack = []
    for c in s:
        # 压栈
        if c == '{' or c == '[' or c == '(':
            stack.append(c)
        else:
            # 栈需非空，不然后面stack[-1]会报 run time error
            if not stack:
                return False

            if c == ']' and stack[-1] != '[' \
                    or c == ')' and stack[-1] != '(' \
                    or c == '}' and stack[-1] != '{':

                return False
            # 弹栈
            stack.pop()

    return not stack  # 最后stack空了就return True，不然就return False

# lintcode M 57 3Sum for降纬1次 + 双指针做法  数组未sort，返回所有满足条件的数字组合，因为solution里有多组(要考虑对duplicate number的特殊处理 by 排序从小到达处理+遇到已处理过的相同元素就skip)
res = [] # 这是个 global 变量
def threeSum(nums):
    """
    测试数据：
    numbers = [-1,0,1,2,-1,-4]
    output = [[-1, 0, 1],[-1, -1, 2]]

    【降纬思想】
    要找 a + b + c = 0
    实际上是找 a + b = -c
    把其中一个因子for循环一下，剩下的就是2个纬度了
    因此看到4数之和也可以先降成3维

    其实2sums用哈希表做就是这种思路，
    在限定下找另一个值
    比如找 a+b = target，我们就是
    for a in hash:
        找(target-a)是否在哈希表里

    可以用哈希表来做, 总时间复杂度 O(n^2)，空间复杂度O(n)
    但由于我们在进入two_sum_equal_to() 这个函数时，数组已经sort好了，那就用双指针来做的更快，总时间复杂度 O(n^2)，空间复杂度O(1)
    """
    global res
    # 先排序，要 return 的3个数字是要找 a ≤ b ≤ c的，对a进行for循环，后面找two sums也是要基于对于排序数组的操作，方便去重
    # ！！！这种先排序可以去（1）去重相同答案（2）加快速度
    nums.sort()   # 去重技巧1：排序从小到大依次处理，避免重复处理
    # 开始降维了，由于a + b + c = 0, 下面这个 for 是对 a = nums[i] 来for的, 这样就把题目降成二维了
    for i in range(len(nums)):
        # 下标有效检测
        # 去重技巧2：若这个元素已等于前一个，不再重复处理，不然会被加到result里
        if (i-1) >= 0 and nums[i] == nums[i-1]:  # 要养成习惯，当对
            continue
        # 然后开始找，-a = b + c
        target = - nums[i]
        two_sum_equal_to(target, i, nums)
    return res
def two_sum_equal_to(target, i, nums):
    # 进到这里，nums已经是sort好了的，就用双指针来做空间消耗最小(不用哈希表了)
    # 由于two sum 也就是 b + c 的和是 target = -a = -nums[i]
    # 是在 i+1 ~ len(nums)-1 范围内找的（就避免重复查找，并且找出来的3个数字index肯定不同）
    global res
    left = i + 1
    right = len(nums) - 1
    while left < right:
        two_sum = nums[left] + nums[right]
        if two_sum < target:
            left += 1
        elif two_sum > target:
            right -= 1
        else:  # two_sum == target:
            res.append([nums[i], nums[left], nums[right]])
            # 找到满足条件的，并不马上退出，因为还要继续往中间找直到 left=right

            '''
            去重：这个是 left 移1步后，发现跟前一个相等，就为了避免重复，移到直到nums[left]跟前一数不等的时候
                  比如会有这种情况 -47 1 1 2 45 46 46
                                    l            r  
            left 和 right 移动一位后:   l        r  又找到一组1+46
                                                  为了避免重复加一个while循环 去掉这层重复
            '''
            left += 1
            # 去重点3：相同元素skip，不再重复处理，不然会被加到result里
            while (left < right) and nums[left] == nums[left-1]: # 易错点：如果这个数和前一个相等，skip，直到跟前一个数不等
                left += 1

# 测试的 tree
def buildTree1():
    """
        5
       / \
      2   6
     / \   \
    4   5   4    # 5 5 6
    """
    node1 = TreeNode(5)
    node2 = TreeNode(2)
    node3 = TreeNode(6)
    node4 = TreeNode(4)
    node5 = TreeNode(5)
    node6 = TreeNode(4)

    node1.left = node2
    node1.right = node3
    node3.right = node6
    node2.left = node4
    node2.right = node5

    return node1
# 数 tree 的 good nodes 个数, 是 leetcode 1448
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
def goodNodes( root: TreeNode) -> int:

    def dfs(node, current_max):
        nonlocal number_of_good_nodes

        if current_max <= node.val:
            number_of_good_nodes += 1
        if node.left:
                dfs(node.left, max(node.val, current_max))
        if node.right:
            dfs(node.right, max(node.val, current_max))

    number_of_good_nodes = 0
    if not root:
        return number_of_good_nodes

    dfs(root, float("-inf"))

    return number_of_good_nodes

# lintcode Medium 71 · Binary Tree Zigzag Level Order Traversal 正常BFS做，需要逆序的时候翻转一下列表再append，思路最简单吧
def zigzagLevelOrder(root):
    """
    题目主要思路： BFS + Queue + 翻转
    """
    if not root:
        return []

    # bfs 开始了
    result = []
    from collections import deque
    q = deque()
    q.append(root)
    is_left_to_right = True

    while q:
        this_level_nodes = []

        size = len(q) # 这一层的长度

        for i in range(size):
            cur = q.popleft()
            this_level_nodes.append(cur.val)

            if cur.left:
                q.append(cur.left)
            if cur.right:
                q.append(cur.right)

        if not is_left_to_right:
            # 如果需要逆序，那就把level_nodes去reverse 一下再 append
            this_level_nodes.reverse()

        result.append(this_level_nodes)

        # 翻转读取顺序
        is_left_to_right = not is_left_to_right

    return result

# 建立一个linked list methods: add to front， remove from end， print
# 网页答案 https://www.alphacodingskills.com/ds/notes/linked-list-delete-the-last-node.php
class ListNode:
    def __init__(self, val, next=None):
        self.val = val
        self.next = next
class Linkedlist:

    def __init__(self):
        self.head = None

    def add_to_front(self, val): # val means value
        new_node = ListNode(val)

        # 如果这个 Linkedlist 是空的
        if self.head == None:
            self.head = new_node
            return

        # 如果这个 Linkedlist 非空
        temp = self.head
        new_node.next = temp
        self.head = new_node


    def remove_from_end(self):
        # 如果已经空了，就直接结束，啥都不返回
        if self.head == None:
            return

        # 如果只剩1个元素了, 直接把 self.head 变成 None
        if self.head.next == None:
            self.head = None
            return

        # 如果还剩2个或者2个以上元素
        temp = self.head
        while temp.next.next != None:
            temp = temp.next
        temp.next = None


    def print(self):
        # 打印当前 Linkedlist 里所有的元素？
        temp = self.head
        if (temp != None):
            print("The list contains:", end=" ")
            while (temp != None):
                print(temp.val, end="->")
                temp = temp.next
            print()
        else:
            print("The list is empty.")


if __name__ == '__main__':
    # 测试 good
    # test 1(edge case): root 是 None 的情况
    # test 2(正常建的一个树) 来数 good nodes
    root1 = buildTree1() # 建树
    result = goodNodes(root1)
    print(result)


    # 测试 Linked List
    sol = Linkedlist()
    sol.add_to_front(1)
    sol.add_to_front(2)
    sol.add_to_front(3)
    sol.add_to_front(4)
    sol.print()
    sol.remove_from_end()
    sol.remove_from_end()
    sol.remove_from_end()
    sol.print()
    sol.remove_from_end()
    sol.remove_from_end()
    sol.print()


    # 测试 binary tree 的 zigzag traversal
    answer = zigzagLevelOrder(root1)
    print(answer)


    # 测试 3sum
    result2 = threeSum( [-1,0,1,2,-1,-4])
    print(result2)    #  output = [[-1, 0, 1],[-1, -1, 2]]


    # 测试
    result3 = intersection( [4,9,5], [9,4,9,8,4])
    print(result3)