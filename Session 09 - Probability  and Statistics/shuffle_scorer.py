#!/usr/bin/env -S uv run
"""shuffle_scorer.py"""

import random

import numpy as np


def deal_cards(deck):
    hands = np.zeros((4, 4, 13))  # 4 hands, 4 suits per hand, 13 cards per suit
    for card_pos, card_num in enumerate(deck):
        hand_num = card_pos % 4
        suit_num = card_num // 13
        rank_num = card_num % 13
        hands[hand_num][suit_num][rank_num] = 1
    return hands


def score_deal(hands):
    score = 0.0
    for player in range(4):
        # Ideally each player should have 3.25 cards of each suit
        for suit in range(4):
            score += (np.sum(hands[player][suit]) - 3.25) ** 2
        # Ideally each player should have one card of each rank
        for rank in range(13):
            score += (hands[player][0][rank] - 1) ** 2
            score += (hands[player][1][rank] - 1) ** 2
            score += (hands[player][2][rank] - 1) ** 2
            score += (hands[player][3][rank] - 1) ** 2
    return score


def wash_shuffle(deck):
    for card_pos, card_num in enumerate(deck):
        new_card_pos = random.randint(0, 51)
        deck[card_pos] = deck[new_card_pos]
        deck[new_card_pos] = card_num
    return deck


def riffle_shuffle(deck):
    # Cut the deck into two equal piles
    left_pile, right_pile = deck[:26], deck[26:]
    # Prepare a new empty deck to hold the riffled halves
    new_deck = np.zeros(0, dtype=int)
    while len(new_deck) < 52:
        # Riffle in a set of cards from the LEFT pile
        chunk = random.randint(1, 4)  # Take from 1 to 4 cards inclusive
        new_deck = np.append(new_deck, left_pile[:chunk])
        left_pile = left_pile[chunk:]  # Remove this chunk form the LEFT pile
        # Riffle in a set of cards from the RIGHT pile
        chunk = random.randint(1, 4)  # Take from 1 to 4 cards inclusive
        new_deck = np.append(new_deck, right_pile[:chunk])
        right_pile = right_pile[chunk:]  # Remove this chunk form the RIGHT pile
    return new_deck


def score_shuffle(shuffle_func, num_deals=10_000):
    """
    Takes the name of a shuffle function and the number of deals to simulate\n
    Returns the average deal score across those total number of deals
    """
    total_score = 0.0
    deck = np.arange(52)
    for _ in range(num_deals):
        deck = shuffle_func(deck)
        total_score += score_deal(deal_cards(deck))
    return total_score / num_deals


def main():
    random.seed(2016)
    print(f"Wash Shuffle Avg. Score   = {score_shuffle(wash_shuffle)}")
    print(f"Riffle Shuffle Avg. Score = {score_shuffle(riffle_shuffle)}")


if __name__ == "__main__":
    main()
