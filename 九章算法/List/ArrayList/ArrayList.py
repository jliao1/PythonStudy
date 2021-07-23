# Description
# 用 ArrayList 实现一些操作：
#
# create(n). 创建一个大小为n的ArrayList，包含n个整数，依次为[0, 1, 2, ... n-1]
# clone(list). 克隆一个list。使得克隆后的list与原来的list是完全独立无关的。
# get(list, index). 查询list中index这个位置的数。
# set(list, index, val). 将list中index这个位置的数改为val。
# remove(list, index). 移除list中index这个位置的数。
# indexOf(list, val). 在list中查找值为val的数，返回它的index。如果没有返回-1。


'''
Array List 就是 List

'''


class ArrayListManager:
    '''
     * @param n: You should generate an array list of n elements.
     * @return: The array list your just created.
    '''
    def create(self, n):
        # Write your code here
        list1 = []
        for i in range(n):
            list1.append(i)
        return list1

    '''
     * @param list: The list you need to clone
     * @return: A deep copyed array list from the given list
    '''
    def clone(self, list):
        # Write your code here
        dist = []
        for a in list:
            dist.append(a)
        return dist

    '''
     * @param list: The array list to find the kth element
     * @param k: Find the kth element
     * @return: The kth element
    '''
    def get(self, list, k):
        # Write your code here
        return list[k]

    '''
     * @param list: The array list
     * @param k: Find the kth element, set it to val
     * @param val: Find the kth element, set it to val
    '''
    def set(self, list, k, val):
        # write your code here
        list[k] = val

    '''
     * @param list: The array list to remove the kth element
     * @param k: Remove the kth element
    '''
    def remove(self, list, k):
        # write tour code here
        list.remove_helper(k)

    '''
     * @param list: The array list.
     * @param val: Get the index of the first element that equals to val
     * @return: Return the index of that element
    '''
    def indexOf(self, list, val):
        # Write your code here
        if list is None:
            return -1
        try:
            ans = list.index(val)
        except ValueError:
            ans = -1
        return ans