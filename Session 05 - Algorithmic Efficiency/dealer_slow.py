#!/usr/bin/env -S uv run
"""dealer_slow.py"""

import time

import numpy as np

# fmt: off
suits = ["Clubs", "Diamonds", "Hearts", "Spades"]

ranks = ["Deuce", "Three", "Four", "Five", "Six", "Seven",
         "Eight", "Nine", "Ten", "Jack", "Queen", "King", "Ace"]
# fmt: on


def init_deck():
    deck = np.arange(52)
    already_dealt = np.zeros(52, dtype=bool)
    for card_pos in range(52):
        card_num = np.random.randint(52)
        while already_dealt[card_num]:
            card_num = np.random.randint(52)
        deck[card_pos] = card_num
        already_dealt[card_num] = True
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
    total_deals = 10_000

    start_time = time.perf_counter()
    for _ in range(total_deals):
        deck = init_deck()
    elapsed_time = time.perf_counter() - start_time

    print_deck(deck)

    print(f"Total deals: {total_deals:,}")
    print(f"Total run time (sec): {elapsed_time:.3f}")


if __name__ == "__main__":
    main()
