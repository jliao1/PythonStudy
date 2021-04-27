from Queue import Queue  # 这个是用linked list写得所以要import
from queue import Queue  # 这个是python自带的，做lintCode题可以直接使用

if __name__ == '__main__':
    # my_que = Queue()
    # for i in range(50):
    #     my_que.enqueue(i)
    #
    # while not my_que.is_empty():
    #     print(my_que.dequeue(), end=' ')
    # print()


    # 下面这段用的是python库里的 queue module
    que = Queue()
    for i in range(50):
        que.put(i)  # 进队列

    while not que.empty():
        print(que.get(), end=' ')   # get是出队列，并且返回值
    print()
    print(que.qsize())    # 知道队列当前有多少元素