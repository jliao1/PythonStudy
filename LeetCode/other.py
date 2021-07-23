class Solution:

    # Medium 1041. Robot Bounded In Circle 元组的使用
    def isRobotBounded(self, instructions: str) -> bool:
        dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        dirIndex = 0
        position, transitionDir = (0, 0), dirs[dirIndex]
        for char in instructions:
            if char == 'G':
                # 元组相加
                position = tuple(map(lambda i, j: i + j, position, transitionDir))
            elif char == 'L':
                dirIndex -= 1
                transitionDir = dirs[dirIndex % 4]
            else:
                dirIndex += 1
                transitionDir = dirs[dirIndex % 4]
        return transitionDir != (0, 1) or position == (0, 0)