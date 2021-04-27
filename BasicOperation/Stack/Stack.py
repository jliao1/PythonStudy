# Stack在 python里没有现成的，自己用list实现
# 用 append 压栈，pop弹栈，[-1]检查栈顶是什么
# list其实还要厉害点，可以在任何地方访问，stack只能访问顶部

class Stack:
    """使用List模拟栈"""
    def __init__(self):
        self.items = []

    def is_empty(self):
        # 写法1：直接not这个list，如果是空的，not它就返回true；不然就是false
             return not self.items  # if not self.items 翻译成 如果 self.items 是空的
        # 写法2：判断长度是不是0，是0就是空的
            # return len(self.items) == 0
        # 写法3：直接if这个list，如果是空的，if它结果是false；不空就是true
        # if self.items:  # 翻译为：如果非空
        #     return False  # list非空 执行这条
        # else:
        #     return True  # list空 执行这条

    def push(self, item):
        self.items.append(item)

    # 扔出栈顶的值，并返回这个值
    def pop(self):
        return self.items.pop()  # list的pop里不输入subscript就默认删除最后一个位置

    def peek(self):
        if not self.is_empty():  # 先判断空不空
            return self.items[-1]  # 不空再返回list里最后一个值（下标是-1）

    def size(self):
        return len(self.items)