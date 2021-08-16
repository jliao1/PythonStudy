"""
双指针是典型的不使用 extra memory的算法
跟基础班的排序那章主题类似
"""

class Solution:

    # lintcode Medium 415 · Valid Palindrome 令狐老师的版本
    def isPalindrome(self, s):
        """
        相同的逻辑的代码，提成一个函数 is_valid_char() 避免代码耦合
        如果 and 和 or 在一个判断语句里太多太长了，也可以提成函数，增加可读性
        一般一个条件，有一个 and 或 or 最好，太多了降低 readability
        """
        if not s:
            return True

        left = 0
        right = len(s) - 1

        while left <= right:
            while left <= right and not self.is_valid_char(s[left]):
                left += 1

            while left <= right and not self.is_valid_char(s[right]):
                right -= 1
            #                          就算 s[left] 是数字也可以 s[left].lower() 不会报错啦
            if left <= right and s[left].lower() != s[right].lower():
                return False
            else:
                left += 1
                right -= 1

        return True
    def is_valid_char(self, char):
        return char.isdigit() or char.isalpha()

    # lintcode Medium 891 · Valid Palindrome II 功能相同的代码，提炼成子函数
    def validPalindrome(self, s):
        if not s:
            return True

        left, right = self.find_difference(s, 0, len(s) - 1)

        if left < right:
            return self.is_palindrome(s, left + 1, right) | self.is_palindrome(s, left, right - 1)
        else:
            return True
    def is_palindrome(self, s, left, right):
        l, r = self.find_difference(s, left, right)
        if l < r:
            return False
        else:
            return True
    # 相同的功能的代码，最好提炼成子函数
    def find_difference(self, s, left, right):
        while left <= right and s[left] == s[right]:
            left += 1
            right -= 1

        return left, right

    # lintcode easy 373 · Partition Array by Odd and Even 比较一般的那种
    def partitionArray(self, nums):
        start, end = 0, len(nums) - 1
        while start <= end:
            while start <= end and nums[start] % 2 == 1:
                start += 1
            while start <= end and nums[end] % 2 == 0:
                end -= 1
            if start <= end:
                nums[start], nums[end] = nums[end], nums[start]
                start += 1
                end -= 1

    # lintcode Medium 144 · Interleaving Positive and Negative Numbers
    def rerange(self, A):
        num_of_negative = 0
        num_of_positive = 0
        # 统计正负数的个数
        for each in A:
            if each < 0:
                num_of_negative += 1
            if each > 0:
                num_of_positive += 1
        # 如果负数多，负数放前面；如果正树多，正数放前面
        self.partition(A, 0, 0, len(A) - 1, num_of_negative >= num_of_positive)
        # 交换（这题基本上只有one pass解法，没有2 pass）
        self.interleave(A, num_of_negative == num_of_positive)
    def partition(self, A, key, start, end, start_negative):
        left = start
        right = end

        flag = 1 if start_negative else -1

        while left <= right:
            while left <= right and A[left] * flag < key:
                left += 1
            while left <= right and key < A[right] * flag:
                right -= 1

            if left <= right:
                A[left], A[right] = A[right], A[left]
                left += 1
                right -= 1
    def interleave(self, A, is_same_length):
        right = len(A) - 1
        left = 0 if is_same_length else 1

        while left < right:
            A[left], A[right] = A[right], A[left]
            left = left + 2
            right = right - 2

    # lintcode Easy 539 · Move Zeroes 同向双指针
    def moveZeroes1(self, nums):
        slow = 0
        fast = 0
        while fast < len(nums):
            if nums[fast] != 0:
                nums[fast], nums[slow] = nums[slow], nums[fast]
                slow += 1
            # 这个条件去掉才对 if nums[right] == 0:
            fast += 1

    # lintcode Easy 539 · Move Zeroes 同向双指针
    def moveZeroes2(self, nums):
        """这是如果stable, 最有算法：要修改次数最少"""
        left = 0
        right = 0

        while right < len(nums):
            if nums[right] != 0:
                # 覆盖，但不换过来了(如果换回来也有可能被覆盖掉)
                nums[left] = nums[right]
                left += 1
            right += 1
        # 末尾的数再用0覆盖掉就好
        while left < len(nums):
            nums[left] = 0
            left += 1

    # lintcode Easy 539 · Move Zeroes 同向双指针
    def moveZeroes3(self, nums):
        """
        如果不需stable，最有算法是？
        相当于是非0到左边，0到右边的一个partition
        """
        pass



if __name__ == '__main__':
    pass