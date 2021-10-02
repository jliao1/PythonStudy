def changeAds(base10):
    # 直接 integer 转换成 string
    to_binary = bin(base10)  # convert base10 to binary
    to_string = str(to_binary)
    # find the first index of '1'
    first_1_index = to_string.find('1')

    if first_1_index == -1: # means no '1' found
        return 0

    valid_part = to_string[first_1_index:]

    # begin negating
    List = []
    for char in valid_part:
        if char == '0':
            List.append('1')
        elif char == '1':
            List.append('0')

    negated_binary_String = ''.join(List)

    # 直接string二进制转换成十进制 convert binary to decimal
    to_decimal = int(negated_binary_String,2)

    return to_decimal





if __name__ == '__main__':
    input = 100


    ans = changeAds(input)
    print(ans)