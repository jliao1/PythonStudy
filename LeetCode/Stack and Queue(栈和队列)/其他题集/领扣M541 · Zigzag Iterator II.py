import queue
from collections import deque

class ZigzagIterator1:
    """
    Your ZigzagIterator2 object will be instantiated and called as such:
    solution, result = ZigzagIterator2(vecs), []
    while solution.hasNext(): result.append(solution.next())

    实现一个iterator通常要实现2个方法 next 和 hasNext

    版本1的解体思路就是当成二纬矩阵来做，普通遍历的方法。但整个过程比较复杂。
    当题目有一种"轮换"的感觉的时候，就可以用 queue的数据结构，版本2用queue做的
    """
    def __init__(self, vec):
        """
        @param: vecs: a list of 1d vectors
        空间复杂度O(k) 因为all the input vectors in the variable self.vectors. As a result, we would need O(K) space for K vectors
        """
        self.rowIdx = 0  # 行指针
        self.colIdx = 0  # 列指针
        self.vec = vec

        # the real length of vec 非空行计数器
        self.nonEmptyVecCnt = len(self.vec)

        for i in range(len(self.vec)):
            # If 有 row 是 empty
            if len(self.vec[i]) == 0:
                # nonEmptyVec 减一
                self.nonEmptyVecCnt -= 1

    def _next(self):
        """
        @return: An integer
        时间复杂度好像 O(k), k就是有几行的意思, it will take us K iterations to find a valid element output. Hence, its time complexity is O(K).
        """
        # 【要保证列指针有效】
        # 如果 列指针 已经 走过了 本行最后一个元素
        while self.colIdx >= len(self.vec[self.rowIdx]):
            # 那说明 列指针指了个寂寞，就更新下 行指针, 指到同列的下一行去
            self.rowIdx += 1
            # 但若指到同列的下一行后，已经超出 最后一行，又指了个寂寞
            # 那就行回到0，列加1 (就是指到首行的下一列去)
            if self.rowIdx == len(self.vec):
                self.rowIdx = 0
                self.colIdx += 1

        # 此时 行列指针都 valid 了
        # 想要拿出来的value (也就是行列指针相交的位置）不过返回之前还要)
        val = self.vec[self.rowIdx][self.colIdx]

        # 【一旦拿出 val 后的状态维护】 (maintain nonEmptyVecCnt)
        # 如果 列指针 已在 当前行的最后一个元素了(=当前行的总长度减-1）
        #      就代表，一旦拿出 val，这行就被掏空了
        #      就需要更新 self.nonEmptyVecCnt，这很重要因为这个数据决定我们还有没有下一个元素 hasNext
        if self.colIdx == len(self.vec[self.rowIdx]) - 1:
            self.nonEmptyVecCnt -= 1

        # 【为下一次调用做准备: 行指针要valid】
        # 行 先加 1
        self.rowIdx += 1
        # 若行加1后，已经超出末行，行回到0，列加1
        if self.rowIdx == len(self.vec):
            self.rowIdx = 0
            self.colIdx += 1

        return val

    def hasNext(self):  # 判断还有没有下一个
        """
        时复O(1)
        如果所有 row 都已经遍历了，返回 False
        @return: True if has next
        """
        return self.nonEmptyVecCnt > 0

# LintCode Medium 541 · Zigzag Iterator II "轮换"的感觉，用单队列queue来做. 但是运行结果很慢,因为python的Queue()底层实现就很慢
from queue import Queue
class ZigzagIterator2:
    # 空间复杂度O(二维数组里的元素总个数)
    def __init__(self, vectors):
        self.Q = Queue()
        for vector in vectors:
            if vector:
                q = Queue()
                '''
                vectors里面装的元素是 vector，我这里把vector都开个Queue()空间转成queue存着
                坏处是：
                1. 增加了不必要的空间复杂度
                2. 针对元素vector，其实总体的需求就是 scan 一遍它
                                 用 python 内置的 iterator 就可以满足
                                 没必要再多用种数据结构
                3. 注意遍历完后，数据不能变空喔，不能改变数据的
                '''
                [q.put(i) for i in vector]
                self.Q.put(q)

    # 时间复杂度O(1)
    def next(self):
        vector = self.Q.get()
        val = vector.get()
        if vector.qsize() != 0:
            self.Q.put(vector)
        return val

    # 时间复杂度O(1)
    def hasNext(self):
        return self.Q.qsize() != 0
        # return self.que.qsize() > 0

# LintCode Medium 541 · Zigzag Iterator II "轮换"的感觉，用collections里的双端队列deque()和python自带的iter()来做. deque比Queue()底层实现快,尽管他们时间复杂度一样
import collections
class ZigzagIterator3:
    # 空间复杂度O(k) k是vectors里元素总个数
    def __init__(self, vectors):
        self.queue = collections.deque()
        for vector in vectors:
            # 如果 vector 为空，就不放进 deque
            if len(vector) > 0:
                            # iter(vector) 是一个遍历器
                self.queue.append([iter(vector), len(vector)])

    # 时间复杂度O(1)
    def _next(self):
        v_iter, v_len = self.queue.popleft()
        value = next(v_iter)
        v_len -= 1
        if v_len > 0:
            self.queue.append([v_iter, v_len])
        return value

    # 时间复杂度O(1)
    def hasNext(self):
        return len(self.queue) > 0

# LintCode Medium 541 · Zigzag Iterator II 依然用collections里的双端队列deque来做，但没用iterator了，换了种思路是逆序处理
class ZigzagIterator4:
    # 空间复杂度O(k) k是vectors里元素总个数

    def __init__(self, vectors):
        # 声明一个双端队列
        self.queue = collections.deque()
        # 遍历这k个一维向量，如果vec不为空，就放进 双端队列里
        for vec in vectors:
            if vec:
                self.queue.append( vec[::-1] )
                """
                为啥这里要 vec[::-1] 呢？ 这是与 ZigzagIterator3 不太一样的地方
                由于 vec 是一个 list，一会儿我们要 pop(0)时间复杂度是O(n)
                所以我们先翻转，一会儿 pop() 最后一位时间复杂度就是O(1)了
                当然翻转也要O(n)啦，所以还是目测还是 ZigzagIterator3 更efficient一些
                但在写 next() 时还是这种方法感觉逻辑要清晰一些
                """

    # 时间复杂度O(1)
    def _next(self):
        # 出队列
        vec = self.queue.popleft()
        # vec 是个list，pop出它最后一个元素（就是翻转前的第一个元素）
        value = vec.pop()
        # 如果 vec 在pop上面的元素后，还非空，就重新入队 queue
        if vec:
            self.queue.append(vec)

        return value

    # 时间复杂度O(1)
    def hasNext(self):
        return len(self.queue) > 0
