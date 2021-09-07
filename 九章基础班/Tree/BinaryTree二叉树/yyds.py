
class Solution:

    def shortestDistance(self, maze, start, destination) :
        from collections import deque
        #               U       R      D      L
        DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        # make "start" and "destination" as tuple cuz they are list at the beigining
        start = (start[0], start[1], None)
        destination = (destination[0], destination[1], None)

        # initialize
        queue = deque([start])
        visited = {start: 0}

        while queue:
            # current node info
            curr_node = queue.popleft()
            curr_x = curr_node[0]
            curr_y = curr_node[1]
            curr_direction = curr_node[2]

            # stopping case
            if curr_node == destination:
                return visited[curr_node]

            # case 1: next node should re-choose a direction
            if curr_direction == None:
                # choose next possible direction
                for next_direction in DIRECTIONS:
                    # calculate next node info
                    next_x = curr_x + next_direction[0]
                    next_y = curr_y + next_direction[1]
                    next_node = (next_x, next_y, next_direction)

                    if self.is_vaild3(maze, next_node):
                        self.process_next_node(next_node, curr_node, visited, queue)

            # case 2: next node will go along its current direction
            if curr_direction != None:
                # calculate next node info with current direction
                next_direction = curr_direction
                next_x = curr_x + next_direction[0]
                next_y = curr_y + next_direction[1]
                next_node = (next_x, next_y, next_direction)

                # case 2.1: if next_node is valid
                if self.is_vaild3(maze, next_node):
                    self.process_next_node(next_node, curr_node, visited, queue)
                else:
                    # case 2.2: if next_node is NOT valid, that means next node will meet a barrier or boundary
                    #           so we should go back the current node's x and y, but direction should set to None
                    next_node = (curr_x, curr_y, None)
                    self.process_next_node(next_node, curr_node, visited, queue)

        return -1

    def process_next_node(self, next_node, curr_node, visited, queue):
        if next_node in visited:
            return
        queue.append(next_node)
        visited[next_node] = visited[curr_node] + 1

    def is_vaild3(self, maze, node):
        rows, columns = len(maze), len(maze[0])
        x = node[0]
        y = node[1]

        # 没有撞墙也没有走出地图就是 valid
        if not (0 <= x < rows and 0 <= y < columns):
            return False

        if maze[x][y] == 1:
            return False

        return True

if __name__ == '__main__':

    Input = [[0,0,1,0,0],[0,0,0,0,0],[0,0,0,1,0],[1,1,0,1,1],[0,0,0,0,0]]
    sol = Solution()
    l = sol.shortestDistance([0,4],[4,4],Input)
    print(l)
