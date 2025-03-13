# main.py
from execution.trade_manager import AITradeManager
from rf_pipline.rf_logic import train_rf_for_durations, train_trade_direction, get_detected_trades
from data_manager.trade_storage import LoadTrades
from rf_pipline.load_and_save import load_rf_models, save_rf_models


def main():
    while True:
        print("\nSelect Option: ")
        print("T - Train Model")
        print("C - Collect Trades")
        print("E - Exit")
        choice = input("Choice: ").strip().upper()

        if choice == "T":
            trades = LoadTrades()
            choice = input("Retrain Mag RF? Y/n: ").strip().upper()
            if choice == 'Y':
                models = train_rf_for_durations(trades)
                save_rf_models(models, 'mag')
            else:
                models = load_rf_models('mag')
            choice = input("Retrain Dir RF? Y/n: ").strip().upper()
            if choice == 'Y':
                training_trades = get_detected_trades(trades, models)
                final_models = train_trade_direction(training_trades)
                save_rf_models(final_models, 'dir')
            else:
                models = load_rf_models('dir')
        elif choice == "C":
            tm = AITradeManager()
            while True:
                tm.get_trades()
        elif choice == "E":
            print("Exiting...")
            break
        else:
            print("Not a valid input.")


if __name__ == "__main__":
    main()
