'''
Compare two version numbers version1 and version2.
If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.

You may assume that the version strings are non-empty and contain only digits and the . character.
The . character does not represent a decimal point and is used to separate number sequences.
For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.

'''

class Solution(object):
    # 将字符串转化成整数，再比较大小
    def compareVersion(self, version1, version2):
        v1_list = version1.split('.')  # 出来是个list
        v2_list = version2.split('.')

        # 这个逻辑，漂亮了       取长度更大的
        for i in range(0, max(len(v1_list), len(v2_list))):
                                   # 如果 out of bound 赋值0
            v1 = int(v1_list[i]) if len(v1_list) > i else 0
            v2 = int(v2_list[i]) if len(v2_list) > i else 0
            if v1 > v2:
                return 1
            elif v1 < v2:
                return -1
        return 0

if __name__ == '__main__':
    sol = Solution()
    res = sol.compareVersion('1', '01')
    print(res)