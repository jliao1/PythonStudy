"""
【常见排序都要背！】

【if 分情况讨论】
在一个helper function里如果有多种case需要讨论
- 一般 if 先讨论简单 case，再依次讨论 复杂der 的case
- 还有一种是 if 范围先严格，再 if 依次扩大范围，这样就可以有筛选的感觉，条件窄的先执行
- 如果 case 都是 remove，但是有几种情况的remove，可以都集中到一个 remove 功能函数里写，不用分开成好几个吧


【algorithm paradigm 和 algorithm】
algorithm paradigm 算法思想：比如 Divide and conquer，Brute-force search，Dynamic programming，greedy，Backtracking
algorithm 是算法执行的具体步骤
算法实现方式： recursion iteration

【非常非常非常大的一个Tree到底是啥意思】
要考虑到
非常非常非常大的一个Tree到底是啥意思。
不要递归
不要遍历
有比较操作的话用小的去比大的（和sql里小表join大表一样）
找其他能用到的条件（但是这个题反复看也就是二叉树，找不到优化的方案）


"""