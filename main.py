# main.py
"""
Entry point with a console menu.
Keeps your old "T", "C", "S", "E" structure.
"""

from config import settings, min_thresh
from ml_pipeline.train_all_brains import train_all_durations_for_multitask
from execution.trade_manager import AITradeManager
from data_manager.trade_storage import SaveTrades
# from data_manager.trade_storage import LoadTrades, SaveTrades  # if needed
# from execution.real_trading import EnterLiveTrade, etc.


def main():
    while True:
        print("\nSelect Option: ")
        print("T - Train Model")
        print("C - Collect Trades")
        print("S - Edit Settings")
        print("E - Exit")
        choice = input("Choice: ").strip().upper()

        if choice == "T":
            train_all_durations_for_multitask(threshold_for_label=min_thresh)
        elif choice == "C":
            tm = AITradeManager()
            try:
                while True:
                    tm.get_trades()
            except KeyboardInterrupt:
                print("\nSaving Trades")
                SaveTrades(tm.retro_trades)
        elif choice == "S":
            edit_settings()
        elif choice == "E":
            print("Exiting...")
            break
        else:
            print("Not a valid input.")


def edit_settings():
    # example of toggling your 'settings' dict
    keys = list(settings.keys())
    while True:
        for i, k in enumerate(keys):
            print(f"{i+1}. {k} = {settings[k]}")
        print("E to exit")

        ans = input("Select setting or E: ")
        if ans.strip().upper() == "E":
            break
        if ans.isdigit():
            idx = int(ans) - 1
            if 0 <= idx < len(keys):
                key = keys[idx]
                settings[key] = not settings[key]
                print(f"{key} changed to {settings[key]}")
            else:
                print("Invalid index.")
        else:
            print("Invalid input.")


if __name__ == "__main__":
    main()
