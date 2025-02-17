# data_manager/data_preprocessing.py
"""
Functions for data balancing, normalizing, etc.
"""

import random
from config import min_thresh

def BalanceTrades(trades_list):
    """
    Moved from your old code. Balances trades into buy/sell/neutral per duration.
    """
    for i in range(5):  # 5 durations
        label = str(i + 1)
        buy_threshold = min_thresh[i]
        sell_threshold = -min_thresh[i]
        buy_trades = [idx for idx, trade in enumerate(trades_list)
                      if trade['results'][label] is not None and trade['results'][label] > buy_threshold]
        sell_trades = [idx for idx, trade in enumerate(trades_list)
                       if trade['results'][label] is not None and trade['results'][label] < sell_threshold]
        neutral_trades = [idx for idx, trade in enumerate(trades_list)
                          if trade['results'][label] is not None
                          and sell_threshold <= trade['results'][label] <= buy_threshold]

        min_class_size = min(len(buy_trades), len(sell_trades), len(neutral_trades))
        buy_trades = random.sample(buy_trades, min_class_size) if len(buy_trades) > min_class_size else buy_trades
        sell_trades = random.sample(sell_trades, min_class_size) if len(sell_trades) > min_class_size else sell_trades
        neutral_trades = random.sample(neutral_trades, min_class_size * 2) if len(neutral_trades) * 2 > min_class_size else neutral_trades

        valid_indices = set(buy_trades + sell_trades + neutral_trades)
        for idx, trade in enumerate(trades_list):
            if idx not in valid_indices:
                trade['results'][label] = None

        print(f"For duration {label}:")
        print(f"BUY trades: {len(buy_trades)}")
        print(f"SELL trades: {len(sell_trades)}")
        print(f"NEUTRAL trades: {len(neutral_trades)}")
        print(f"Total balanced trades: {3 * min_class_size}")

    # remove trades that ended up with no durations
    idx = 0
    while idx < len(trades_list):
        remove = True
        for i in range(5):
            label = str(i + 1)
            if trades_list[idx]['results'][label] is not None:
                remove = False
                break
        if remove:
            trades_list.pop(idx)
        else:
            idx += 1

    return trades_list
