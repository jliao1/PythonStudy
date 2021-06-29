# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class Solution:

    def get_len(self, head):
        cnt = 0
        while head:
            cnt += 1
            head = head.next
        return cnt

    # LeetCode 1721. Swapping Nodes in a Linked List
    def swapNodes1(self, head: ListNode, k: int) -> ListNode:
        """
        :param k: 正数第K个和倒数第K个交换

        由于需要先求链表长度，时间复杂度固定是在O(2n)
        空间复杂度是常数 O(1)
        思路：
        找到需要交换的 第一个 第二个 node后，分情况讨论
        情况1: 第一个node和第二个node相同
        情况2: nodes相邻，还要分谁在前后  用 swap_adj(self, fst_pre, sec_pre):
        情况3: nodes不相邻，不用分谁在前后  用 swap_remote(self, fst_pre, sec_pre):
        """
        if not head:
            return head

        # dummy is used for record the first node prev is the k=1
        dummy = ListNode(0)
        dummy.next = head

        # get len of this linked list
        len = self.get_len(head)

        # Initializing first prev and second prev
        fst_pre = dummy
        sec_pre = dummy

        # find first node prev
        for i in range(k - 1):
            fst_pre = fst_pre.next

        # find second node prev
        for i in range(len - k):
            sec_pre = sec_pre.next

        # case 1: two nodes are the same
        if fst_pre == sec_pre:
            return dummy.next

        # case 2: two nodes are adjacent
        if fst_pre.next == sec_pre:
            self.__swapAdj(fst_pre, sec_pre)
            return dummy.next
        if sec_pre.next == fst_pre:
            self.__swapAdj(sec_pre, fst_pre)
            return dummy.next

        # case 3: two nodes are neither adjacent nor same
        self.__swapRemote(fst_pre, sec_pre)

        return dummy.next

    # LeetCode 1721. Swapping Nodes in a Linked List
    def swapNodes2(self, head: ListNode, k: int) -> ListNode:
        """
        空间复杂度依旧是O(1)
        优化1：用前/后指针做，O(n~2n)，best case是O(n)，worst case是O(2n)
        优化2：不用分不同交换nodes的位置情况，直接用 swap(self, preV1, preV2) 包含了所有情况
        """
        if not head:
            return head

        # dummy is used for record the first node prev is the k=1
        dummy = ListNode(0)
        dummy.next = head

        # Initializing ahead and behind pointer
        ahead = dummy
        behind = dummy

        # use the ahead pointer to find the 1st node prev
        for i in range(k - 1):
            ahead = ahead.next
        # record the 1st node prev
        fst_pre = ahead

        # use the behind pointer to find the 2nd node prev
        while ahead.next.next:
            ahead = ahead.next
            behind = behind.next
        # record the 2nd prev
        sec_pre = behind

        self.__swap(fst_pre, sec_pre)
        # swap
        return dummy.next

    # LintCode 511 · Swap Two Nodes in Linked List
    def swapTwoNodes(self, head, v1, v2):
        """
        :param v1: 第一个交换数字的值
        :param v2: 第二个交换数字的值
        Time complexity O(n)
        Space complexity O(1)
        """
        # find preV1 and preV2
        dummy = ListNode(0, head)
        preV1 = dummy
        preV2 = dummy
        while preV1.next and preV1.next.val != v1:
            preV1 = preV1.next
        if not preV1.next:
            return dummy.next
        while preV2.next and preV2.next.val != v2:
            preV2 = preV2.next
        if not preV2.next:
            return dummy.next

        # 不用管2个nodes是否相邻，直接swap
        self.__swap(preV1, preV2)

        return dummy.next

    # Leetcode 92. Reverse Linked List II
    def reverseBetween(self, head, m, n):
        """
        Time complexity O(n)
        Space complexity O(1)
        """
        if not head or not head.next or m == n:
            return head

        sentinel = ListNode(0, head)
        curr = sentinel

        # Find preM and preN
        cnt = 0
        preM = sentinel
        preN = sentinel
        while curr:
            if cnt == m - 1:
                preM = curr
            if cnt == n - 1:
                preN = curr
                break

            cnt += 1
            curr = curr.next

        # mark start and end from which the inner linked list is going to reverse
        reverseStart = preM.next
        reverseEnd = preN.next
        tempEndNext = reverseEnd.next
        reverseEnd.next = None

        # reverse
        preM.next, reverseEnd = self.reverse1(reverseStart)
        # connect reverse to original linked list
        reverseEnd.next = tempEndNext

        return sentinel.next

    # Leetcode 143. Reorder List
    def reorderList(self, head):
        """
        Time complexity O(n)
        Space complexity O(1)
        """
        if not head or not head.next:
            return head

        # find middle
        M = self.findMiddle(head)

        # reverse the second half in-place
        second = self.reverse2(M.next)

        first = head
        M.next = None

        # merge two
        while second and first:  # 其实只要写 while second 就够了，因为second只会比first更先走完
            temp = second.next
            second.next = first.next
            first.next = second
            first = second.next
            second = temp
        '''
        # 其实以上merge的这6句更短可以写成这样
        while second.next:
            first.next, first = second, first.next
            second.next, second = first, second.next        
        '''

        return head




# class 内的功能函数______________________________________________________________________________________________

    # 交换LinkedList里的2个nodes, 不管相邻不相邻，相不相同。代码来自 lint code 的511题的solution中，名叫 RickSJCA 用户提供的答案
    def __swap(self, pre1st, pre2nd):
        """
        :param pre1st: 1st是第一个数字的位置，pre1st是第一个数字的前一位指针
        :param pre2nd: 2nd是第二个数字的位置，pre2是第二个数字的前一位指针
        这两个参数不分前后 pre1st 和 pre2nd 可以互换。
        :return: Nothing, 但就是把 1st 和 2nd 要交换的位置的数字，交换了一下
        """
        # 先交换入口
        pre1st.next, pre2nd.next = pre2nd.next, pre1st.next
        # 再交换出口
        pre1st.next.next, pre2nd.next.next = pre2nd.next.next, pre1st.next.next

    # 交换相邻的两个 nodes
    def __swapAdj(self, pre1st, pre2nd):
        """
        :param pre1st: 1st是第一个数字的位置，pre1st是第一个数字的前一位指针
        :param pre2nd: 2nd是第二个数字的位置，pre2是第二个数字的前一位指针
        这两个参数，要分先后
        :return:  Nothing, 但就是把 1st 和 2nd 要交换的位置的数字，交换了一下
        """
        pre1st.next = pre2nd.next
        pre2nd.next = pre1st.next.next
        pre1st.next.next = pre2nd

    # 交换不相邻的两个 nodes
    def __swapRemote(self, pre1st, pre2nd):
        """
        :param pre1st: 1st是第一个数字的位置，pre1st是第一个数字的前一位指针
        :param pre2nd: 2nd是第二个数字的位置，pre2是第二个数字的前一位指针
        这两个参数不分前后 pre1st 和 pre2nd 可以互换
        :return: Nothing, 但就是把 1st 和 2nd 要交换的位置的数字，交换了一下
        """
        fst = pre1st.next
        sec = pre2nd.next
        sec_nxt = sec.next

        pre1st.next = sec
        sec.next = fst.next

        pre2nd.next = fst
        fst.next = sec_nxt

    # 返回 翻转后的 新头，新尾
    def reverse1(self, head):
        """
        :param head:
        :return: 翻转后的linked list的 新head 和 新tail
        """
        # curt表示前继节点
        newHead = None
        newTail = head
        while head:
            # temp指针存 head的下一个节点
            temp = head.next
            # head的下一个 指到 curt位置上
            head.next = newHead
            # 再把 curt 指针移到 当前head位置上
            newHead = head
            # 再把 head 指针移到 当前temp位置上
            head = temp
        return newHead, newTail

    # 只返回 翻转后的 新头, 这个写法更简短
    def reverse2(self, head):
        prev, curr = None, head
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        return prev

    # 返回中间node，比如eg: 1 2 3 4 return 3
    def findMiddle(self, head):
        """
        :param head:
        :return: the middle of the linked list  eg：1 2 3 return 2, eg: 1 2 3 4 return 3
        """
        fast = head
        slow = head

        while fast and fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next

        return slow

# class 外的测试函数______________________________________________________________________________________________

# 测试用的打印Linked List
def printList(head):
    result = []
    while head:
        result.append(head.val)
        head = head.next
    print(result)


if __name__ == '__main__':

    node1 = ListNode(2)
    node2 = ListNode(-1)
    node3 = ListNode(0)
    node4 = ListNode(4)
    node5 = ListNode(5)
    node6 = ListNode(6)
    node7 = ListNode(7)
    node8 = ListNode(8)
    node9 = ListNode(9)
    node10 = ListNode(10)

    node1.next = node2
    node2.next = node3
    # node3.next = node4
    # node4.next = node5
    # node5.next = node6
    # node6.next = node7
    # node7.next = node8
    # node8.next = node9
    # node9.next = node10

    sol = Solution()

    printList(node1)

    res = sol.reorderList(node1)

    printList(res)
