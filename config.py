# config.py
import socket

hostname = socket.gethostname()
device = "Laptop" if hostname == "Yotam-Laptop" else "PC"

# Where we store trade data
TRADE_PKL = "retro_trades.pkl"

# If you want a separate folder for saved models
SAVED_MODELS_DIR = "./saved_models"

# The main symbols list you used
symbols = [
    {"symbol": "EURJPY", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "GBPUSD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "GBPCAD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "AUDJPY", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "EURCAD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "CADJPY", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "GBPAUD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "EURGBP", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "USDJPY", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "AUDUSD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "EURUSD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "AUDCAD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "AUDCHF", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "CADCHF", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "CHFJPY", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "EURCHF", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "GBPCHF", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "GBPJPY", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "USDCAD", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "USDCHF", "exchange": "OANDA", "screener": "forex"},
    {"symbol": "EURAUD", "exchange": "OANDA", "screener": "forex"}
]

# For balancing trades, min thresholds for durations
min_thresh = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009] # 10, 12, 15, 18
