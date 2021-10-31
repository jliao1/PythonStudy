# 力扣 E 155. Min Stack 用pair,pair里第一个元素是value，第二个记录current_min
class MinStack:
    # 全部时间O(1)
    def __init__(self):
        self.stack = []

    def push(self, val: int) -> None:
        if not self.stack:
            # val,current_min
            pair = [val, val]
            self.stack.append(pair)
        else:
            last_pair = self.stack[-1]
            Min = last_pair[1]

            if val <= Min:
                Min = val

            new_pair = [val, Min]
            self.stack.append(new_pair)

    def pop(self) -> None:
        if not self.stack:
            return
        self.stack.pop()

    def top(self) -> int:
        if not self.stack:
            return
        return self.stack[-1][0]

    # 时间复杂度O(1)
    def getMin(self) -> int:
        if not self.stack:
            return
        return self.stack[-1][1]

if __name__ == '__main__':
    obj = MinStack()
    obj.push(-2)
    obj.push(0)
    obj.push(-3)
    print(  obj.getMin()  )
    obj.pop()
    print(  obj.top()  )
    param_4 = obj.getMin()