'''
【关于指针的两个基本操作】
currNode指针后移：              curr = curr.next
再把 curt 指针移到 当前head位置上: curt = head (把 head 地址 赋值给 curt)
改变currNode的next指向another：  curr.next = another


【链表一定要考虑 头 中 尾】
为了统一化处理，用一个 dummy node

【while里什么情况下是False】
在while 0
  while None
  while False
  while 空的字符串,tuple,list

【LinkedList 和 普通 List 比较】
查找 List 用 index 查找 O(1)，而 LinkedList 需要 O(n) 遍历
插入如果已知要插入哪里，LinkedList 快用 O(1)，List要把其他后移用O(n)

【写链表的时候要注意】
要保证 while 里的 内容不能是 None
所以比如最好 while curr and curr.next： 如果循环里会用到 curr.next 的话


'''



class ListNode(object):

    def __init__(self, val, next=None):
        self.val = val
        self.next = next


class Solution:
    cnt = 0
    '''
    Insert a node in a sorted linked list.
    空间 O(N)
    时间 O(1)
    '''
    def insertNode(self, head, val):
        # write your code here
        dummy = ListNode(float('-inf'))
        dummy.next = head

        curr = dummy
        new = ListNode(val)

        # while curr.next的意思 = while curr.next is not None
        while curr.next and curr.next.val < new.val:
            curr = curr.next

        new.next = curr.next
        curr.next = new

        return dummy.next


    '''
    Remove all elements from a linked list of integers that have value val.
    空间 O(N)
    时间 O(1)
    '''
    def removeElements(self, head, val):
        # write your code here
        dummy = ListNode(float('-inf'))
        dummy.next = head
        curr = dummy

        while curr.next:
            if curr.next.val == val:
                curr.next = curr.next.next
            else:
                curr = curr.next

        return dummy.next

    """
    Find the middle node of a linked list and return it.
    有3种思路
    (1)遍历第1遍，找到长度n
       再遍历第2遍到n/2的位置找到 middle
    (2)只遍历1遍，用 快/慢指针 找到 middle

    实际上(1)虽然遍历2遍，但(1)与(2)的操作次数是一样的，都是O(1.5N) 
    不是代码看起来只遍历一遍就更省时间，实际上是比basic operation的操作次数的
    从工程上来说，其实（1）会更 readable

    (3)用空间换时间
       在遍历第1遍的时候把每个node位置存到 list 里
       直接返回list的中间的下标
       总的复杂度是O(N)

    这里我们用(2) 快/慢 指针来做
    """
    def middleNode(self, head):
        if not head:
            return head

        slow, fast = head, head
        # 为啥不能直接写 while fast.next.next？
        # 因为要先确保 fast.next 存在，才能继续探讨 fast.next.next 存在
        # 否则会报错 fast.next不一定还有next了
        while fast.next and fast.next.next:
            # 慢指针走1步时，快指针就走2步
            # 这适合 n 是奇数/偶数 的所有情况
            fast = fast.next.next
            slow = slow.next

        return slow

    '''
    Given a list, rotate the list to the right by k places, where k is non-negative.
    这是我自己写的版本
    空间 O(N)
    时间 O(1)
    '''
    def rotateRight1(self, head, k):
        # write your code here
        # 若head是null，或者head只是一个节点
        if not head or head.next is None or k == 0:
            return head

        # 遍历求长度
        curr = head
        cnt = 1
        tail = None
        while curr.next:
            cnt += 1
            curr = curr.next
            if not curr.next:
                tail = curr

        if k >= cnt:
            k = k % cnt

        curr = head
        for i in range(1, cnt - k):
            curr = curr.next

        tail.next = head
        head = curr.next
        curr.next = None

        return head

    '''
    这是九章老师讲课的版本，用 前/后 指针来看
    要点是：能把一些功能模块拆成函数，就拆成函数写，先写核心模块，最后有时间再去完善功能函数，这样及时最后写不完功能函数但也有可能是过的
    '''

    def rotateRight2(self, head, k):
        if head is None:
            return head

        # dummy 在这里的作用是 一会儿 head 在代码中会后移
        # dummy.next 永远保留了最初对 head 节点的记录
        dummy = ListNode(0)
        dummy.next = head

        length = self.get_len(head)

        # 用链表长度取模
        k = k % length

        # 前/后 指针 登场
        ahead, behind = dummy, dummy  # 前/后 指针，初始放在dummy

        # 前指针先移动k步
        for i in range(k):
            ahead = ahead.next

        # 然后开始同步移动 前/后 指针
        while ahead.next:  # 只要ahead没到最后一位，继续移动, 直到 ahead 移到末尾停下
            behind = behind.next
            ahead = ahead.next

        ahead.next = dummy.next
        dummy.next = behind.next
        behind.next = None

        return dummy.next

    def get_len(self, head):
        l = 0
        while head:
            l += 1
            self.cnt += 1
            head = head.next
        return l

    '''
    Given a linked list, remove the nth node from the end of list and return its head.
    自己做的
    '''
    def removeNthFromEnd1(self, head, n):
        if not head:
            return head

        len = self.get_len(head)

        # if n > len or n == 0:
        #     return head

        k = len - n
        dummy = ListNode(0)
        dummy.next = head
        curr = dummy  # 地址相等,

        '''
        dummy -> node1 -> node2 -> node3 -> node4
          ↑
         curr
        在经过 curr.next = curr.next.next 操作后】
        就变成
        dummy -> node2 -> node3 -> node4 
          ↑
         curr
        '''

        k = len - n

        for i in range(k):
            curr = curr.next

        curr.next = curr.next.next

        return dummy.next


    '''
    九章的答案：快慢指针做法
    题意：删除链表中倒数第n个结点，尽量只扫描一遍。
    使用两个指针扫描，当第一个指针扫描到第N个结点后，
    第二个指针从表头与第一个指针同时向后移动，
    当第一个指针指向空节点时，另一个指针就指向倒数第n个结点了       
    '''
    def removeNthFromEnd2(self, head, n):
        res = ListNode(0)
        res.next = head
        tmp = res
        for i in range(0, n):
            head = head.next
        while head != None:
            head = head.next
            tmp = tmp.next
        tmp.next = tmp.next.next
        return res.next

    '''
    翻转链表
    我自己写的版本0
    '''
    def reverse0(self, head):
        pre = ListNode(0)
        pre.next = head

        temp = head.next
        head.next = None
        head = temp

        while head:
            temp = head.next
            head.next = pre.next
            pre.next = head
            head = temp

        return pre.next

    '''
    翻转链表
    九章答案：需要画图仔细体会
    '''
    def reverse1(self, head):
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

    '''
    翻转链表
    九章答案：递归写法：根本看不懂
    '''
    def reverse2(self, head):
        if not head or not head.next:
            return head

        tail = self.reverse2(head.next)

        head.next.next = head
        head.next = None
        return tail

    '''
    Description
    Merge two sorted (ascending) linked lists and return it as a new sorted list. The new sorted list should be made by splicing together the nodes of the two lists and sorted in ascending order.
    
    这是我自己写的版本，看看2版九章的版本比较简洁
    '''
    def mergeTwoLists1(self, l1, l2):
        # write your code here
        if l1 is None and l2 is None:
            return l1

        if not l1 and l2:
            return l2

        if l1 and not l2:
            return l1

        dummy = ListNode(0)
        tail = dummy

        while l1 and l2:
            t1, t2 = None, None
            if l1 is not None:
                t1 = l1.val
            if l2 is not None:
                t2 = l2.val

            if t1 and t2:
                if t1 >= t2:
                    tail.next = ListNode(t2)
                    l2 = l2.next
                else:
                    tail.next = ListNode(t1)
                    l1 = l1.next

                tail = tail.next

            if not l2 and l1:
                tail.next = l1

            if not l1 and l2:
                tail.next = l2

        return dummy.next

    #九章的版本
    def mergeTwoLists2(self, l1, l2):
        dummy = ListNode(0)
        tmp = dummy
        while l1 != None and l2 != None:
            if l1.val < l2.val:
                tmp.next = l1
                l1 = l1.next
            else:
                tmp.next = l2
                l2 = l2.next
            tmp = tmp.next
        if l1 != None:
            tmp.next = l1
        else:
            tmp.next = l2
        return dummy.next

    '''
    Write code to remove duplicates from an unsorted linked list.
    自己写的，用 dict 做的
            也可以用 set 做，用到前后指针，看版本2
    '''
    def removeDuplicates1(self, head):
        # write your code here
        dic = {}

        dummy = ListNode(0)
        dummy.next = head
        curr = dummy

        while curr.next:
            if curr.next.val in dic:
                curr.next = curr.next.next
            else:
                dic[curr.next.val] = True
                curr = curr.next

        return dummy.next

    '''
    用set做
    用前/后指针做
    '''
    def removeDuplicates2(self, head):
        visited = set()
        prev = None
        curr = head

        while curr:
            if curr.val in visited:
                # prev的下一个 = curr的下一个，就删除了curr
                # 然后curr移动到 prev的下一个(也就是原curr的下一个)
                prev.next = curr.next
                curr = curr.next  # 这句话也可以写成 curr = prev.next

            else:
                visited.add(curr.val)
                # prev 始终在 curr 前面一位
                prev = curr
                curr = curr.next

        return head

    '''
    Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
    这是我自己写的 recursion 版本
    Time Complexity: O(N) where NN is the size of the linked list.
    Space Complexity: O(N) stack space utilized for recursion.
    '''
    def swapPairs1(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        # 若代码走到此，说明至少 head 有2个 nodes

        curr = head.next
        head.next = curr.next
        curr.next = head
        head = curr

        # 注意这个地方要连上的
        head.next.next = self.swapPairs(head.next.next)

        return head

    '''
    iterative版本
    Time Complexity : O(N) where N is the size of the linked list.
    Space Complexity : O(1)
    '''
    def swapPairs2(self, head):
        if not head or not head.next:
            return head

        dummy = ListNode(0)
        dummy.next = head
        curr = dummy

        while head and head.next:
            # swap
            curr.next = head.next
            curr = curr.next
            head.next = curr.next
            curr.next = head

            # Reinitializing the head and curr for next swap
            curr = head
            head = head.next

        return dummy.next

    '''
     Remove Duplicates from Sorted List
     这是我自己写的
     Time Complexity : O(N) where N is the size of the linked list.
     Space Complexity : O(1)
    '''
    def deleteDuplicates(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        fast = head.next
        slow = head

        while fast:
            if fast.val != slow.val:
                fast = fast.next
                slow = slow.next
            else:
                fast = fast.next
                slow.next = fast

        return head

    # 打印Linked List
    def printList(self, head):
        res = []
        while head:
            res.append(head.val)
            head = head.next
        print(res)


    '''
    Palindrome Linked List 
    做到了挑战的 时间O(n)  空间O(1)
    '''
    def isPalindrome(self, head):
        # write your code here
        if not head or not head.next:
            return True

        dummy = ListNode(0)
        dummy.next = head

        # find the middle node (if even, choose the latter one)
        middle = self.get_M(dummy)

        reverse = self.reverse1(middle)

        return self.compare(head,reverse)

    # compare if two lists are the same
    def compare(self, head, reverse):

        while head and reverse:
            if head.val != reverse.val:
                return False
            head = head.next
            reverse = reverse.next
        return True

    # find the middle node (if even, choose the latter one)
    def get_M(self, head):
        fast = head
        slow = head

        while fast and fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next

        return slow.next

    '''
    LeetCode 82. Remove Duplicates from Sorted List II
    自己写的答案，用了前/后指针
    '''
    def deleteDuplicates(self, head):
        # write your code here
        if not head or not head.next:
            return head

        dummy = ListNode(0)
        dummy.next = head

        curr = head
        pre = dummy

        while curr and curr.next:
            if pre.next.val != curr.next.val:
                pre = pre.next
                curr = curr.next
            else:
                while curr.next and curr.next.val == curr.val:
                    curr = curr.next

                pre.next = curr.next
                curr = curr.next
        return dummy.next


if __name__ == '__main__':

    node1 = ListNode(1)
    node2 = ListNode(2)
    node3 = ListNode(3)
    node4 = ListNode(4)
    node5 = ListNode(5)
    node6 = ListNode(6)
    node7 = ListNode(7)
    node8 = ListNode(8)
    node9 = ListNode(9)
    node10 = ListNode(10)

    node1.next = node2
    node3.next = node4
    node3.next = node4
    node4.next = node5
    node5.next = node6
    node6.next = node7
    node7.next = node8
    node8.next = node9
    node9.next = node10

    l2 = ListNode(2)

    sol = Solution()

    res = sol.swapNodes3(node1,3)
    pass







