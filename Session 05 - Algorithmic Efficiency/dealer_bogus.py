#!/usr/bin/env python3
"""dealer_bogus.py"""

import numpy as np

# fmt: off
suits = ["Clubs", "Diamonds", "Hearts", "Spades"]

ranks = ["Deuce", "Three", "Four", "Five", "Six", "Seven",
         "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace"]
# fmt: on


def init_deck():
    deck = np.arange(52)
    for card_pos in range(52):
        card_num = np.random.randint(52)
        deck[card_pos] = card_num
    return deck


def print_deck(deck):
    for card_pos in range(52):
        card_num = deck[card_pos]
        suit_num = card_num // 13
        rank_num = card_num % 13
        card_name = f"{ranks[rank_num]} of {suits[suit_num]}"
        print(f"The card in position {card_pos} is the {card_name}")


def main():
    np.random.seed(2016)
    deck = init_deck()
    print_deck(deck)


if __name__ == "__main__":
    main()
