
'''

x << y
将一个运算对象的各二进制位全部左移若干位（左边的二进制位丢弃，右边补0）
Returns x with the bits shifted to the left by y places
(and new bits on the right-hand-side are zeros). This is
the same as multiplying x by 2的y 次方.

x >> y
将一个数的各二进制位全部右移若干位，正数左补0，负数左补1，右边丢弃。

x & y
运算规则：0&0=0;  0&1=0;   1&0=0;    1&1=1;

x | y
运算规则：0|0=0；  0|1=1；  1|0=1；   1|1=1；

~ x
Returns the complement of x - the number you get by switching
each 1 for a 0 and each 0 for a 1. This is the same as -x - 1.

x ^ y
运算规则：0^0=0；  0^1=1；  1^0=1；   1^1=0；

'''