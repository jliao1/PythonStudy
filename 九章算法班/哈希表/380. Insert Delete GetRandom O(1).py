# 领扣M 380 Insert Delete GetRandom O(1)
class RandomizedSet:
    # 思路：用hash记录下标， 删除只需要和末尾元素调换即可

    def __init__(self):

        self.l = []
        self.d = {}

    def insert(self, val: int) -> bool:
        if val in self.d:
            return False
        self.d[val] = len(self.l)
        self.l.append(val)
        return True

    def remove(self, val: int) -> bool:
        if val not in self.d:
            return False

        remove_index = self.d[val]
        last_element = self.l[-1]
        last_index = len(self.l) - 1

        self.d[last_element] = remove_index
        self.d.pop(val)

        self.l[remove_index] = last_element
        self.l.pop()

        return True

    def getRandom(self) -> int:
        index = random.randint(0, len(self.l) - 1)
        return self.l[index]
