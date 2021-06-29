class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random

class Solution:

    """
    Leetcode: 138 Copy List with Random Pointer
    Time complexity: O(n)
    Space complexity: O(n)   版本2的Space complexity 优化到 O(1)
    """
    def copyRandomList1(self, head: 'Node') -> 'Node':
        oldToNew = {None: None}  # hashmap will take O(n) memory

        # step1: creat a new copy of old nodes without linking them yet
        #        and map every old node to the new node
        oldNode = head
        while oldNode:
            newNode = Node(oldNode.val)
            oldToNew[oldNode] = newNode

            oldNode = oldNode.next

        # step2: do the new node connecting
        curr = head
        while curr:
            newNode = oldToNew[curr]
            newNode.next = oldToNew[curr.next]
                        # edge case: 如果curr.next是None，那它要map谁？
                        # None要map到None, 所以 第一行写 oldToNew = {None: None}
            newNode.random = oldToNew[curr.random]
            curr = curr.next

        return oldToNew[head]

    '''
    Leetcode: 138 Copy List with Random Pointer
    Time complexity: O(n)
    Space complexity: O(1) 比版本1低
    '''
    def copyRandomList2(self, head: 'Node') -> 'Node':

        if not head:  # 这里不能加 or not head.next，会报错，因为里面多一个attribute啦
            return head

        # part (1)
        curr = head
        # Generate a new weaved list of original and cloned nodes
        # Before: A -> B -> C -> None
        # After: A -> A' -> B -> B' -> C -> C' -> None
        while curr:
            cloned = Node(curr.val)
            cloned.next = curr.next
            curr.next = cloned
            curr = cloned.next  # 漏掉了：要移动curr

        # part (2)
        curr = head
        # Now link the random pointers of the cloned nodes
        # Iterate the newly generated list and use the original nodes random pointers, to assign references to random pointers for cloned nodes.
        while curr and curr.next:  # 不需要curr.next.next否则就curr就不会走到最后了
            curr.next.random = curr.random.next if curr.random else None
            curr = curr.next.next

        # part (3)
        # Delete original nodes, only cloned nodes left
        # Before: A -> A' -> B -> B' -> C -> C' -> None
        # Before: A' -> B' -> C' -> None
        sentinel = Node(0)
        sentinel.next = head
        curr = sentinel
        while curr and curr.next and curr.next.next:
            curr.next = curr.next.next
            curr = curr.next

        return sentinel.next

if __name__ == '__main__':
    pass