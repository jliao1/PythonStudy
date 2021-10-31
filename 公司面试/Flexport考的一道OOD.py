# https://www.1point3acres.com/bbs/forum.php?mod=viewthread&tid=810447&highlight=flexport
class Card:
    def __init__(self, blue_tokens=0, green_tokens=0):
        # properties
        self.blue_tokens = blue_tokens
        self.green_tokens = green_tokens

# 继承怎么写？
class GreenCard(Card):
    def __init__(self, blue_tokens=0, green_tokens=0):
        # properties
        Card.__init__(self, blue_tokens, green_tokens)

class WhiteCard(Card):
    def __init__(self, blue_tokens=0, green_tokens=0):
        # properties
        Card.__init__(self, blue_tokens, green_tokens)

from enum import Enum
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

class Player:
    # 全部时间O(1)
    def __init__(self,blue_tokens = 0, green_tokens = 0, cards = []):
        # properties
        self.blue_tokens = blue_tokens
        self.green_tokens = green_tokens
        self.cards_list = cards

    def add_blue_tokens(self, num: int) -> None:
        self.blue_tokens += num

    def add_green_tokens(self, num: int) -> None:
        self.green_tokens += num

    def afford(self, card, num ):   # if can afford, return True; if no, return False
        if card.blue_tokens * num <= self.blue_tokens\
            and card.green_tokens * num <= self.green_tokens:
            return True
        else:
            return False

    # 先用afford确定能买，再买
    def buy(self, card, num):
        self.blue_tokens -= card.blue_tokens*num
        self.green_tokens = card.green_tokens
        for _ in range(num):
            self.cards_list.append(card)

    def print_properties(self):
        print("The player has")
        print("blue_tokens:", self.blue_tokens)
        print("green_tokens:", self.green_tokens)
        print("cards:", len(self.cards_list))




class System:
    def __init__(self, blue_tokens = 0, green_tokens = 0):
        self.blue_tokens = blue_tokens
        self.green_tokens = green_tokens


class Game:
    def __init__(self, m, n):
        self.borad = []

        for _ in range(m):
            List = []
            for _ in range(n):
                List.append(0)
            self.borad.append(List)

if __name__ == '__main__':

    game = Game(5, 5)


    player1 = Player(10,10)    #  blue_tokens=10, green_tokens=10
    card = Card(2,1)           # blue_tokens=2, green_tokens=1
    if player1.afford(card,2):
        player1.buy(card,2)
    player1.print_properties()



    #
    green_card = GreenCard(2, 1)
    # 判断 card 是不是 Card类型的对象
    print(isinstance(green_card,GreenCard))


    pass