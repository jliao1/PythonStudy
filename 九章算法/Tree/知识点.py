"""
BFS可以干什么？
1）查找你想要的目标点（一层一层查找）
2）到达目标点的最小距离（经历了几层第一次达到目标点）

时间空间复杂度
1.Time complexity is the sameO(N) both for DFS and BFS since one has to visit all nodes.
2.Space complexity is O(H) for DFS and O(D) for BFS,
  where H is a tree height, and D is a tree diameter.
  They both result in O(N) space in the worst-case scenarios
  该怎么选择？
  so choose DFS for skewed tree and BFS for complete tree.

BST作用：
（1）对BST进行中序遍历 left-root-right 得到得是一个递增(非降)序列
（2）有插入操作（普通的二叉树binary tree没有(因为普通的结构太松散混乱了)，普通的只有深度/宽度优先遍历）
（3）查找高校（尤其是平衡BST应用是数据库，搜索引擎）
（4）常用操作: 插入，查找，删除（用得少）

需要背诵的
1.求二叉树高度
2.判断2个树是否相等
3.前序中序DFS的iterative写法

"""