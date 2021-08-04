
# 力扣 Medium 1381. Design a Stack With Increment Operation
class CustomStack:

    def __init__(self, maxSize: int):
        self.stack = []  # Use list to implement the custom stack
        self.MAX_SIZE = maxSize

    def push(self, x: int) -> None:
        if len(self.stack) < self.MAX_SIZE:
            self.stack.append((x, 0))

    def pop(self) -> int:
        if not self.stack:  # Return -1 if stack is empty
            return -1

        top = self.stack.pop()
        # 如果 pop 了之后 stack 还不为空的话
        if self.stack:
            self.__add_val_to_ith_tuple(-1, top[1])
        return top[0] + top[1]

    # O(1)
    # Uses lazy increment, it doesn't increment bottom k elements but temporary store val
    # Pop function add stored vals to the element when it pops
    def increment(self, k: int, val: int) -> None:
        if not self.stack:  # Return if stack is empty
            return

        if k >= len(self.stack):
            self.__add_val_to_ith_tuple(-1, val)
        else:
            self.__add_val_to_ith_tuple(k - 1, val)

    def __add_val_to_ith_tuple(self, i: int, val: int) -> None:
        self.stack[i] = (self.stack[i][0], self.stack[i][1] + val)

if __name__ == '__main__':

    list = [1,2,3,4,5]


    obj = CustomStack(5)
    obj.push(1)
    obj.push(2)
    obj.push(3)
    obj.push(4)
    obj.push(5)
    obj.push(6)
    obj.push(7)
    param_5 = obj.pop()
    print(param_5)

    obj.increment(3, 100)

    param_4 = obj.pop()
    print(param_4)

    param_103 = obj.pop()
    print(param_103)

    param_102 = obj.pop()
    print(param_102)

    param_101 = obj.pop()
    print(param_101)

    param_null = obj.pop()
    print(param_null)


    pass