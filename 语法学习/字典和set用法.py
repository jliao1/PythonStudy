"""

老师您好！
对于217题，请问为什么我用 dict 就可以submit成功，而用 set 在submit的时候就会出现wrong answer？？

我用dict的代码长这样：
    def removeDuplicates(self, head):
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



我用set的代码长这样：
    def removeDuplicates(self, head):
        # write your code here
        s = set()

        dummy = ListNode(0)
        dummy.next = head
        curr = dummy

        while curr.next:
            if str(curr.next.val) in s:
                curr.next = curr.next.next
            else:
                s.update(str(curr.next.val))   #  错误点在于set 用 add，别用 update，update是用来更新合并set的
                curr = curr.next

        return dummy.next


"""