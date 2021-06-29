
if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]
    y = ['a', 'b', 'c', 'd']
    xy = zip(x, y)

    for a, b in zip(x, y):
        print(a)
        print(b)

    x = [1,2,3,4,5]
    xx = zip(x)

    for a in zip(x):
        print (a)
    for a in zip(x):
        print ( a[0] )