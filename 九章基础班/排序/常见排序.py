
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

# 如果是在已排好序的数组里插入新的，耗时O(n)
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

    是 stable 的，比如有2个3，原来的第一个3排序完后依然出现在第二个3的前面
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
    空间复杂度：外面开一份儿大的长度为N的array，每层递归都共用这一份儿就好，每次都传参来使用
              这样就把空间复杂度降低到O(n)了
              但空间的开辟和释放，很耗时间，这让它整体表现比 quick_sort 差

    是 stable 的，比如有2个3，原来的第一个3排序完后依然出现在第二个3的前面
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
worst case下是O(n^2)，比如输入的数据已经排序好了，每次却选第一个数作partition
平均是 NlogN

空间复杂度：
栈空间 O(logN)  没有堆空间  因此比merge sort好

quicksort 是不stable的 
'''
# lintcode Easy 464 · Sort Integers II
def quick_sort(array): # 这个代码要背. 思路是先整体有序，再局部有序。分治法
    if not array:
        return
    quick_sort_helper(array, 0, len(array)-1)
def quick_sort_helper(A, start, end):
    '''
    注意点5：
    双重循环的时候，算时间复杂度，是要考虑的是最内层的循环主题执行的次数
    比如这题主要是考虑
    while left <= right:
        while left <= right and A[left] < pivot:
            left += 1   # 考虑这条语句执行多少次就可以了， left最多+n次
    '''
    # 递归出口
    if start >= end:
        # 越界了，或只剩1个数了，就不用排了
        return
    '''
    注意点1: pivot 最好选取接近 median value 的值
    或者: pivot = random.randint(start, end)
    '''
    pivot = A[(start + end) // 2]

    left, right = start, end

    # 接下来开始 模版partition 过程(是不stable的)
    '''
    注意点2：为什么接下来4个while里为什么是left <= right 而不是 left < right？
            因为如果这样的话，比如 [3,2,1,4,5]  最后剩下[1,2]再怎么做也还是会被分成[]和[1,2]，就无限循环了，就stack overflow了
            因此如果要写 left < right，那么while循环结束时在 left = right，
                                     此时需要多一个 if 判断 nums[left] 是<pivot在左侧怎么样，还是≥pivot在右侧怎么样
                                     如果是 left <= right 那在退出这个 while 循环的时候，left一定指向右半边的第一个位置
    '''
    while left <= right: # 总要做 left <= right 的检查防止越界
        # i 指向的值 < pivot 那就i右移
        while left <= right and A[left] < pivot:
            left += 1
        # j 指向的值 > pivot 那就j左移
        while left <= right and pivot < A[right]:
            right -= 1
        '''
        注意点3：
        以上 A[left] < pivot 和 pivot < A[right] 为什么没有等于呢？
        因为要partition成的结果是：半边元素都是 ≤ pivot，右半边 ≥ pivot，最后 左半边整体 ≤ 右半边 就行
                            不能分成左半边<pivot,右半边≥pivot 或左半边≤pivot, 右半边>pivot嘛？
                        （1）不行。因为比如遇到这样的情况[1,1,1,1] 会分成 []和[1,1,1,1], 问题规模没有减小，会stack overflow
        我们期望的partition是一个均匀的分配(这样可以降低时间复杂度)
        （2）这样做还可以保证分配尽量均匀
        比如[2,2,1,0]  pivot=1  原本，大于pivot的数比较多，小于pivot的数比较少
        最后 partition 变成[0,1,2,2]  pivot就补到左半边比较少那部分去了
        再比如[2,1,0,0]  pivot=1, partition后就是[0,0,1,2] pivot就补到右半边较少的部分去了
        pivot起到中间一个润滑剂作用，左边比较少补到左边，右边比较少就补到右边
        这样就导致 partition 得比较均匀     
                  
        注意点4：
        以下为什么当 left 和 right 的值都指向 pivot了还要交换呢？
        1。保证子问题规模一定小于原问题 要decrease递归才会结束  不然在一种情况下，全部元素都相等，这种情况规模不会decrease
        2。使子问题规模尽量相等，降低时间复杂度，因为left/right往中间聚拢，子问题规模就会越来越平均。就是说 partition 最好在中间位置        
'''
        if left <= right:  # 否则交换值
            A[left], A[right] = A[right], A[left]
            # 后面要分割，left 和 right 不能相交的，当 left = right时，还需各进一步
            left += 1
            right -= 1
    '''
    这样弄完一轮之后，左边 start～right 的元素都是 ≤ pivot，
                   右边 left～ end 的元素都是 ≥ pivot的了
                   中间如果有 孤立的数，那它的值也一定是 pivot
                   为什么一定要左边 ≤ pivot, 右边 ≥ pivot，为什么必须有等号呢？
                （1）说明pivot是可左可 右的。让两边分的结果更平均平衡一些。降低run time。只要 左边整体 ≤ 右边就行
                （2）主要是处理一个数组全都是相同元素时的edge case，依然可以有效partition，而不要stack overflow
    '''

    # 递归：  子问题的左右边界一定是 [left, j] 和 [i, left]
    quick_sort_helper(A, start, right)
    quick_sort_helper(A, left, end)

# lintcode Medium 5 · Kth Largest Element
def quickSelect(self, A, k):
    """
    最容易想到的是直接排序，返回第k大的值。时间复杂度是O(nlogn)

    而这是 O(n) 的做法
    这题其实是快速排序算法的变体，
    通过快速排序算法的partition步骤，
    可以将小于等于pivot的值划分到pivot左边，
    大于等于pivot的值划分到pivot右边
    从而缩小范围继续找第k大的值(k要么在pivot左边或右边)

    平均 时间复杂度O(n) 因为 T(n) = T(n / 2) + O(n) 算出来是O(n)
    空间复杂度 O(1)
    """
    if not A or k < 1 or k > len(A):
        return None
    # 为了方便编写代码，这里将第 k 大转换成第 [len(A) - k] 小问题。
    return self.partition(A, 0, len(A) - 1, len(A) - k)
def partition(self, nums, start, end, k):
    """
    During the process, it's guaranteed start <= k <= end
    """
    if start == end:
        # 说明找到了
        return nums[k]

    left, right = start, end
    pivot = nums[(start + end) // 2]
    while left <= right:
        while left <= right and nums[left] < pivot:
            left += 1
        while left <= right and nums[right] > pivot:
            right -= 1
        if left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left, right = left + 1, right - 1

    # 情况1
    if k <= right:
        # pivot 左区间都小于等于 pivot
        return self.partition(nums, start, right, k)
    # 情况2
    if k >= left:
        # pivot 右区间都大于等于 pivot
        return self.partition(nums, left, end, k)

    # 情况3: right 和 left 中间隔了一个数，这个数就刚好是我们要找的数
    return nums[k]

# lintcode(力扣75) Medium 148 · Sort Colors 方法是 counting sort
def sortColors1(self, nums):
    """
    这种是 counting sort
    这种空间复杂度是O(range), range最大是O(n) 比如万一遇到每个元素都不一样
    时间复杂度O(n)

    空间复杂度O(k), k = 2
    worst case情况下用来存颜色的 count_color 是O(n)，因为有可能一个颜色只出现1次
    如果想降低空间复杂度：
    array + 固定两三种元素 + O(N)时间 + O(1) 的空间 =》其实做两三次quick sort就好，看方法2

    counting sort是基于值的排序，时间复杂度可以做到O(n)，但为什么系统内的sort不用这个，因为range有可能会很大
                              然后不可数，所以for不了。整数可数，但实数不可数，字符串也不可数
                              很多O(n)的算法有局限性
                              而quick和merge sort是基于比较的排序，基于比较就可以return结果
    """
    # 其实可以用 dict 来 count，但能用简单的数据结构就用list啦
    count_color = [0] * 3      # 这个需要额外的空间了
    for each in nums:
        count_color[each] += 1

    # sorted
    index = 0
    for i in range(len(count_color)):
        while count_color[i] > 0:
            nums[index] = i
            index += 1
            count_color[i] -= 1

    return nums

# lintcode Medium 143 · Sort Colors II 彩虹排序 经典算法
def sortColorsTwo3( colors, k):
    """
    瞎猜的话这题肯定不是O(n*k)和O(n^k)，因为这两个时间复杂度都比快排NlogN大
    一般肯定要比NlogN快，不然做题就没啥意义啦
    那么是NlogK还是KlogN呢？可以举特殊例子，
    如果k=1就不需要排序，就是O(1)
    如果k=2时就两种颜色分开，就partition一次就好是O(n)
    所以乍一猜，是O(NlogK)，然后推它算法，一般有两种说法
    （1）n * logK
        降纬 n 次 logK的操作，涉及到log级的是 heap，红黑树，二分法…… 好像不太像
    （2）logK次 的 n 次操作
         归并排序：按树的结构拆解，计算每一层的时间复杂度是O(N)，一共O(logN)层
         由此知道我们希望这题有logK层，希望 K/2  K/4 …… 1 ，每层是O(N)
                                 那怎么去把K一分为二，直到分到1呢
                                 把颜色范围一分为二（顺带把数组一分为二 ）

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
    raibow_sort(colors, 0, len(colors) - 1, 1, k)
def raibow_sort(colors, start, end, color_from, color_to):
    #  只有一个数，或颜色只有一种
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
            left += 1  # 注意这个等号不能写在下面那个条件，因为求color_mid的整除操作是偏小的
            # 若等号写在下面这行，会使得partition不均匀
        while left <= right and colors[right] > color_mid:
            right -= 1

        if left <= right:  # and colors[right] <= colors[left]:
            colors[left], colors[right] = colors[right], colors[left]
            #  下面两行是可以省略的
            left += 1
            right -= 1

    raibow_sort(colors, start, right, color_from, color_mid)  # 好像 color_mid-1 写成 color_mid 也行
    raibow_sort(colors, left, end, color_mid + 1, color_to)

if __name__ == '__main__':
    array = [3,2,1,4,5]
    quick_sort(array)

    for num in array:
        print(num, end = ' ')

    print()
