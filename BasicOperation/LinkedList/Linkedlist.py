# coding=utf-8


class ListNode:

    def __init__(self, val):
        self.val = val
        self.next = None

    def traverse(self):
        cur = self
        while cur is not None:
            print(cur.val, end=' ')
            cur = cur.next
        print()

class Linkedlist:

    def __init__(self):
        self.head = None


    # location 是从 0 位开始算的
    def get(self, location):
        cur = self.head
        for i in range(location):
            cur = cur.next
        return cur.val


    def add(self, location, val):
        if location > 0:
            pre = self.head
            for i in range(location - 1):
                pre = pre.next
            new_node = ListNode(val)
            new_node.next = pre.next
            pre.next = new_node
        elif location == 0:
            new_node = ListNode(val)
            new_node.next = self.head
            self.head = new_node


    def addToEnd(self, val):
        new_tail_node = ListNode(val)
        if self.head is None:
            self.head = new_tail_node
        else:
            cur = self.head
            while cur.next is not None:
                cur = cur.next
            cur.next = new_tail_node


    def set(self, location, val):
        # location 是从 0 位开始算的
        cur = self.head
        for i in range(location):
            cur = cur.next
        cur.val = val


    def remove(self, location):
        if location > 0:
            pre = self.head
            for i in range(location - 1):
                pre = pre.next

            pre.next = pre.next.next

        elif location == 0:
            self.head = self.head.next

    def traverse(self):
        cur = self.head
        while cur is not None:
            print(cur.val, end=' ')
            cur = cur.next
        print()


    def is_empty(self):
        return self.head is None


    '''
    lintCode174. Remove Nth Node From End of List
    Example 1:
	Input: list = 1->2->3->4->5->null， n = 2
	Output: 1->2->3->5->null
    '''
    #方法一，先算 linked list 长度
    # def removeNthFromEnd(self, n):
        # cur = self.head
        # len = 0
        #
        # while cur is not None:
        #     len = len + 1
        #     cur = cur.next
        # location = len - n      # 如果n是2，那么location就是3，实际上是正数第4个元素
        #
        # if location > 0:
        #     pre = self.head
        #     for i in range(location - 1): # 那么就要遍历到第3个元素，就是 0 1 2
        #         pre = pre.next
        #
        #     pre.next = pre.next.next
        #
        # elif location == 0:
        #     self.head = self.head.next
    # 方法二，快慢指针，争取traverse一遍
    # 假设链表是 1 2 3 4 5
    def removeNthFromEnd(self, n):
        fast = slow = self.head
        slowList = ListNode(0)
        slowList.next = slow
        # i is not used in the loop, can be any letter
        for i in range(n):    # 因为 n = 2，所以 i 只会走到 0，1 共2次
            # fast指向链表第n个元素
            fast = fast.next  # 如果链表是 1 2 3 4 5 的话，最后这个for循环结束，fast从1走到了3

            # 直至fast = Null，此时slow指向倒数第n个元素
        while fast.next is not None:
            fast = fast.next
            slow = slow.next
            # 这个while循环结束的时候，fast走到5，slow走到了3

        # skip the nth node （然后把第4个node，倒数第二个 skip掉）
        slow.next = slow.next.next
        return slowList

    '''
    翻转列表
    '''
    # 方法一：iteration approach 从头提取出来加在新链表尾部 视频讲解2:42 https://www.youtube.com/watch?v=bOOdi7S5Ar4
    # def reverse(self, head):  # 其实可以把 head 去掉，函数内的 head 写成 self.head
    # if not head: return head  # edge case：如果head是none，直接return head
    #     #pre表示前继节点
    #     pre = None
    #     while head : (while head != None)
    #         # nxt 记录下一个节点，head是当前节点
    #         nxt = head.next
    #         head.next = pre   #这句的意思是：head的下一个指向 pre：  head -> pre
    #         pre = head        #然后把pre移到 原来head位置
    #         head = nxt        #然后把 新head 移到 原 nxt 位置
    #     return pre     # 返回的是新链表的头节点
    # 方法二：递归 视频讲解2:42 https://www.youtube.com/watch?v=bOOdi7S5Ar4
    def reverse(self, head):   # head只是传了个头地址过来

        # If head is empty or has reached the list end
        # keep recursing head down next to the end
        if head is None or head.next is None: # 为什么head.next是需要的？
            return head  # 这里return的是最后1个，previous recursive call是倒数第二个

        # Reverse the newhead list
        newhead = self.reverse(head.next) # 第一次返回的newHead是 这个linkedList的最后1个，而previous recursive call里的head 是倒数第二个

        # Put first element at the end
        nxt = head.next     # head 的下一个指向 nxt
        nxt.next = head     # 让nxt 的下一个指向 head
        head.next = None    # To avoid infinite loop

        # Fix the header pointer
        return newhead    # 返回的是新链表的头节点


    # Merge Two Sorted Lists
    def mergeTwoLists(self, l2):
        return -1


