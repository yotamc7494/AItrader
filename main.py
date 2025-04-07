# main.py
from execution.trade_manager import AITradeManager
from rf_pipline.rf_logic import train_models
from data_manager.trade_storage import LoadTrades, SaveTrades
from rf_pipline.hyperparameter_tuning import run_dynamic_tuning
from data_manager.trade_constructor import construct_trades


def main():
    print("Select Option: ")
    print("T - Train Model")
    print("C - Collect Trades")
    print("L - Live Trading")
    print("M - Multi-parameter Tuning")
    print("CT - Construct 1W trades")
    choice = input("Choice: ").strip().upper()

    if choice == "T":
        train_models()
    elif choice == "M":
        run_dynamic_tuning()
    elif choice == "CT":
        trades = construct_trades()
        choice = input("Overwrite Current trades? Y/n: ")
        if choice != "Y":
            trades += LoadTrades("C:\\Users\\Yotam\\Desktop\\fetched trades")
        SaveTrades(trades, "C:\\Users\\Yotam\\Desktop\\fetched trades")
    elif choice == "L":
        tm = AITradeManager()
        while True:
            tm.get_trades()
    elif choice == "C":
        tm = AITradeManager()
        try:
            while True:
                tm.get_trades(True)
        except KeyboardInterrupt:
            if tm.added_trades > 0:
                SaveTrades(tm.retro_trades)
    else:
        print("Not a valid input.")
        main()


if __name__ == "__main__":
    main()

