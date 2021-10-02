# 力扣 M 146. LRU Cache 用 OrderedDict实现 可以当stack或队列使用
class LRUCache:
    """
    对于 OrderedDict()有两个方法要熟悉

    popitem(last=True)
    The popitem() method for ordered dictionaries returns and removes a (key, value) pair.
    The pairs are returned in LIFO 栈 order if last is true , or FIFO 队列 order if false.

    move_to_end(key, last=True)
    Move an existing key to either end of an ordered dictionary.
    The item is moved to the right end if last is true (the default) or to the beginning if last is false.
    Raises KeyError if the key does not exist:
    """
    def __init__(self, capacity: int):
        self.cap = capacity
        import collections
        self.dic = collections.OrderedDict()

    def get(self, key: int) -> int:
        if key in self.dic:
            # set key as the newest one
            self.dic.move_to_end(key)
            return self.dic[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.dic:
            self.dic[key] = value
            self.dic.move_to_end(key)
        else:
            self.dic[key] = value

            if len(self.dic) == (self.cap + 1):
                self.dic.popitem(last=False)


if __name__ == '__main__':
    sol = LRUCache(2)
    ans = sol.put(1,1)

