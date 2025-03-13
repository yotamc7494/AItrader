# data_manager/data_preprocessing.py
"""
Functions for data balancing, normalizing, etc.
"""

import random
from config import min_thresh

import random


def BalanceTrades(trades_list):
    """
    Balances the dataset by ensuring equal distribution of BUY, SELL, and NEUTRAL trades.

    Parameters:
        trades_list (list): List of trade dictionaries containing 'results'.
        min_thresh (list): Thresholds for categorizing buy/sell movements.

    Returns:
        list: Balanced list of trades.
    """
    label = "3"  # We balance based on the 1-minute movement
    buy_threshold = min_thresh[0]  # Use first threshold
    sell_threshold = -min_thresh[0]  # Negative for sell

    # Categorize trades into BUY, SELL, and NEUTRAL
    buy_trades = [trade for trade in trades_list if
                  trade['results'][label] is not None and trade['results'][label] > buy_threshold]
    sell_trades = [trade for trade in trades_list if
                   trade['results'][label] is not None and trade['results'][label] < sell_threshold]
    neutral_trades = [trade for trade in trades_list if
                      trade['results'][label] is not None and sell_threshold <= trade['results'][
                          label] <= buy_threshold]

    # Find the minimum class size to balance the dataset
    min_class_size = min(len(buy_trades), len(sell_trades), len(neutral_trades))

    # Randomly sample to make all classes equal in size
    buy_trades = random.sample(buy_trades, min_class_size) if len(buy_trades) > min_class_size else buy_trades
    sell_trades = random.sample(sell_trades, min_class_size) if len(sell_trades) > min_class_size else sell_trades
    neutral_trades = random.sample(neutral_trades, min_class_size) if len(
        neutral_trades) > min_class_size else neutral_trades

    # Merge balanced trades
    balanced_trades = buy_trades + sell_trades + neutral_trades

    print(f"✅ Balanced Trades - BUY: {len(buy_trades)}, SELL: {len(sell_trades)}, NEUTRAL: {len(neutral_trades)}")

    return balanced_trades

