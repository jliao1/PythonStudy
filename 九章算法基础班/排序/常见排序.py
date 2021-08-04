
"""这几种解法都要背！！！！"""
# lintcode Easy 463 · Sort Integers
def selection_sort(A):
    # 时间复杂度 O(n^2)，空间复杂度O(1)，是 in-place的
    size = len(A)
    for i in range(size):
        min_index = i
        for j in range(i+1, size):
            # 这轮for loop 结束后，在 i+1 ～ n 找到了最小元素
            if A[j] < A[min_index]:
                min_index = j
        # 找到后 swwap
        A[i], A[min_index] = A[min_index], A[i]

def insertion_sort(arr):
    # worsr case下时间复杂度 O(n^2)，best case下是O(1)
    # 空间复杂度O(1)
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):

        key = arr[i]

        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key

def bubble_sort1(arr):
    """
    这个冒泡法的意思是，最大的数字，会一轮一轮地跑到后部分
    unsorted ｜ sorted
    """
    # 时间复杂度 O(n^2)，空间复杂度O(1)
    size = len(arr)

    # i 的范围是 0 ～ size-2
    for i in range(size - 1): # range(n) also work but outer loop will repeat one time more than needed.

        # j 的范围是 0 ～ size-2-i
        for j in range(0, size - i - 1):
            '''
            也可以写成 for j in range(0, size - 1):
            那么 j 的范围就是 0 ～ size-2
            但还是写成 j in range(0, size - i - 1)好，
            为什么？
            因为 后面 side of the array is sorted
                So don’t need to go to the end of it every time
                所以每一次其实都在reducing inner loop的范围呢
            '''
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

def bubble_sort2(nums):
    """
    这个冒泡法的意思是，最小的数字，会一轮一轮地跑到前部分
    sorted ｜ unsorted
    """
    for i in range(len(nums)):
        for j in range(i + 1, len(nums)):
            if nums[j] < nums[i]:
                nums[i], nums[j] = nums[j], nums[i]

"""
分治法的思考方式：
- 先假设小的任务已经完成（实际上未完成）
- 在此基础上完成大的任务，此时原来小的任务也就一并完成了
"""
def merge_sort1(array):
    """
    时间复杂度：每层耗时N，一共有logN层(因为平衡)，所以就是NlogN
    空间复杂度：堆空间tmp+栈空间
              栈空间消耗就看递归有多少层：logN层最多
              堆空间 tmp 在每一层都是几分之N
              所以就是NlogN
              但空间是 NlogN的话有点大，这是因为递归的时候，每一次都在merge里开辟tmp空间
              其实完成可以之在外面开一份儿大的长度为N的array，每层递归都共用这一份儿就好，每次都传参来使用
              这样就把空间复杂度降低到O(n)了，所以看版本2

    """
    merge_sort_helper1(array, 0, len(array) - 1)
def merge_sort_helper1(array, left, right): # [left, right]
    # 递归的终点，这种情况就不需要排序了
    if left >= right:
        return

    mid = (left + right) // 2     # [left, mid]   [mid +1, right]
    merge_sort_helper1(array, left, mid)     # 左边有序了
    merge_sort_helper1(array, mid + 1, right)  # 右边有序了
    merge1(array, left, right)               # 再合并
def merge1(array, left, right):
    size = right - left + 1
    temp = [0 for _ in range(size)]

    mid = (left + right) // 2
    # 左半边（要么长度=右半边，要么长度比右半边多1）
    # left_part_i范围: left - mid
    left_part_i = left
    # 右半边
    # right_part_i范围: mid+1 - right
    right_part_i = mid + 1

    # 这个直接把左/右部分都分完了
    for k in range(size):
        # 当右半边有数可取       并且      右半边没数可取     或             左半边数 <= 右半边
        if left_part_i <= mid and (right_part_i > right or array[left_part_i] <= array[right_part_i]):
            temp[k] = array[left_part_i]
            left_part_i += 1
        else:
            # 其他情况
            temp[k] = array[right_part_i]
            right_part_i += 1

    # 这个把 左半边 右半边 sort 好的，copy到原来array里
    for k in range(size):
        array[left + k] = temp[k]


# merge sort 终极版 写法
def merge_sort2(array):
    """
    时间复杂度：每层耗时N，一共有logN层(因为平衡)，所以就是NlogN
    空间复杂度：堆空间tmp+栈空间
              栈空间消耗就看递归有多少层：logN层最多
              堆空间 tmp 在每一层都是几分之N
              所以就是NlogN
              但空间是 NlogN的话有点大，这是因为递归的时候，每一次都在merge里开辟tmp空间
              其实完成可以之在外面开一份儿大的长度为N的array，每层递归都共用这一份儿就好，每次都传参来使用
              这样就把空间复杂度降低到O(n)了，所以看版本2

    """
    tmp = [0 for _ in range(len(array))]
    merge_sort_helper2(array, 0, len(array) - 1, tmp)
def merge_sort_helper2(array, left, right, tmp): # [left, right]
    # 递归的终点，这种情况就不需要排序了
    if left >= right:
        return

    mid = (left + right) // 2     # [left, mid]   [mid +1, right]
    merge_sort_helper2(array, left, mid, tmp)     # 左边有序了
    merge_sort_helper2(array, mid + 1, right, tmp)  # 右边有序了
    merge2(array, left, right, tmp)               # 再合并
def merge2(array, left, right, temp):
    size = right - left + 1

    mid = (left + right) // 2
    # 左半边（要么长度=右半边，要么长度比右半边多1）
    # left_part_i范围: left - mid
    left_part_i = left
    # 右半边
    # right_part_i范围: mid+1 - right
    right_part_i = mid + 1

    # 这个直接把左/右部分都merge了
    for k in range(size):
        # 当右半边有数可取       并且      右半边没数可取     或             左半边数 <= 右半边
        if left_part_i <= mid and (right_part_i > right or array[left_part_i] <= array[right_part_i]):
            temp[k] = array[left_part_i]
            left_part_i += 1
        else:
            # 其他情况
            temp[k] = array[right_part_i]
            right_part_i += 1

    # 这个把 左半边 右半边 sort 好的，copy到原来array里
    for k in range(size):
        array[left + k] = temp[k]


'''
基于比较的排序最快不会超过 NlogN 了，mergesort已经满足了
但 merge sort再怎么样空间复杂度也较高是O(n)
所以来个quick sort是in-place的，空间复杂度 logN

快速排序思想： 
-把数组分为两边，使得数组左边小于等于数组的右边（但左右元素个数，不一定相等）
-对左右两部分数组分别排序（递归）

做法是：
-选取基准数（这会影响时间复杂度，一般选中间一个/随机选，避免选到最大/最小worst case是O(n^2)情况）
-将数组分割为两部分，长度不一定相等（partition）但时间复杂度是O(n)的，因为会走1遍
-递归处理子问题

时间复杂度：
worst case下是O(n^2)，一般是 NlogN

空间复杂度：
栈空间 O(logN)  没有堆空间  因此比merge sort好
'''
# lintcode Easy 464 · Sort Integers II
def quick_sort(array): # 这个代码要背. 思路是先整体有序，再局部有序。分治法
    if not array:
        return
    quick_sort_helper(array, 0, len(array)-1)
def quick_sort_helper(A, start, end):
    # 递归出口
    if start >= end:
        return

    pivot = A[(start + end) // 2]  # 或者: pivot = random.randint(start, end)

    left, right = start, end


    # 接下来开始 partition 过程
    # 总要做 left <= right 的检查防止越界
    while left <= right:
        # i 指向的值 < pivot 那就i右移
        while left <= right and A[left] < pivot:
            left += 1
        # j 指向的值 > pivot 那就j左移
        while left <= right and pivot < A[right]:
            right -= 1

        '''
        为什么当 i 和 j 的值都指向 pivot了还要交换呢？
        1。保证子问题规模一定小于原问题 要decrease递归才会结束  不然在一种情况下，全部元素都相等，这种情况规模不会decrease
        2。使子问题规模尽量相等，降低时间复杂度，因为i/j往中间聚拢，子问题规模就会越来越平均
        '''
        if left <= right:  # 否则交换值
            A[left], A[right] = A[right], A[left]
            # 后面要分割，left 和 right 不能相交的，当 left = right时，还需各进一步
            left += 1
            right -= 1
    '''
    这样弄完一轮之后，左边 start～right 的元素都是 ≤ pivot，
                   右边 left～ end 的元素都是 ≥ pivot的了
                   中间如果有 丢弃的数，那也一定是 pivot
    '''

    # 递归：  子问题的左右边界一定是 [left, j] 和 [i, left]
    quick_sort_helper(A, start, right)
    quick_sort_helper(A, left, end)

    # lintcode Medium 143 · Sort Colors II 彩虹排序
    def sortColorsTwo3(self, colors, k):
        """
        【彩虹排序】解决sort颜色的问题
        这个就是所谓的彩虹排序rainbow Sort 。
        Rainbow Sort其实更像是quicksort的变种，
        我们找到需要排列的颜色中间的那一个作为基准值，
        然后进行类似quicksort的partition

        时间复杂度 O(NlogK)n是数的个数， k 是颜色数目。这是基于比较的算法的最优时间复杂度
        call stack 空间复杂度 O(logK)
        因为  每次是对KK分成左右进行递归，因此有logK 层，每层递归遍历到整个序列，长度为N

        不基于比较的话，可以用计数排序（Counting Sort）
        """
        if not colors:
            return
        # rainbow sort, k 取中间颜色
        self.raibow_sort(colors, 0, len(colors) - 1, 1, k)

    def raibow_sort(self, colors, start, end, color_from, color_to):
        if start >= end or color_from == color_to:
            return

        # 每次选定一个中间的颜色，这个中间的颜色用给出的k来决定，
        # 将小于等于中间的颜色的就放到左边，
        # 大于中间颜色的就放到右边，然后分别再递归左右两半
        color_mid = (color_from + color_to) // 2

        left = start
        right = end

        while left <= right:
            # 移动left指针
            while left <= right and colors[left] <= color_mid:
                left += 1

            while left <= right and colors[right] > color_mid:
                right -= 1

            if left <= right:  # and colors[right] <= colors[left]:
                colors[left], colors[right] = colors[right], colors[left]
                #  下面两行是可以省略的
                left += 1
                right -= 1

        self.raibow_sort(colors, start, right, color_from, color_mid)  # 好像 color_mid-1 写成 color_mid 也行
        self.raibow_sort(colors, left, end, color_mid + 1, color_to)


if __name__ == '__main__':
    array = [2, 8, 5, 2, 4, 6, 3]

    for num in array:
        print(num, end = ' ')

    print()
