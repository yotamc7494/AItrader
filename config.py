# config.py
"""
Central place for global settings, constants, and references.
"""

import os

# Where we store trade data
TRADE_PKL = "retro_trades.pkl"

# If you want a separate folder for saved models
SAVED_MODELS_DIR = "./saved_models"

# Default settings dict, as you had in your old code
settings = {
    'collect trades': True,
    'suggest trades': False,
    'auto execute': False
}

# The main symbols list you used
symbols = [
    {"symbol": "EURJPY", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "GBPUSD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "GBPCAD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "AUDJPY", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "EURCAD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "CADJPY", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "GBPAUD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "EURGBP", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "USDJPY", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "AUDUSD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "EURUSD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "AUDCAD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "AUDCHF", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "CADCHF", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "CHFJPY", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "EURCHF", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "GBPCHF", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "GBPJPY", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "USDCAD", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "USDCHF", "exchange": "FX_IDC", "screener": "forex"},
    {"symbol": "EURAUD", "exchange": "FX_IDC", "screener": "forex"}
]

# For balancing trades, min thresholds for durations
min_thresh = [0.0007, 0.0008, 0.0009, 0.0010, 0.0011]
