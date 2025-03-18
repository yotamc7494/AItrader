# main.py
import numpy
from config import symbols
from execution.trade_manager import AITradeManager
from rf_pipline.rf_logic import train_rf_for_durations, train_trade_direction, get_detected_trades, test_trade_direction_model
from data_manager.trade_storage import LoadTrades, SaveTrades
from rf_pipline.load_and_save import load_rf_models, save_rf_models
from rf_pipline.hyperparameter_tuning import run_dynamic_tuning
import random


def main():
    print("Select Option: ")
    print("T - Train Model")
    print("C - Collect Trades")
    print("L - Live Trading")
    print("M - Multi-parameter Tuning")
    print("E - Exit")
    choice = input("Choice: ").strip().upper()

    if choice == "T":
        trades = LoadTrades()
        random.shuffle(trades)
        train_length = int(len(trades)*0.8)
        training_trades = trades[:train_length]
        test_trades = trades[train_length:]
        choice = input("Retrain Mag RF? Y/n: ").strip().upper()
        if choice == 'Y':
            mag_models = train_rf_for_durations(training_trades)
            save_rf_models(mag_models, 'mag')
        else:
            mag_models = load_rf_models('mag')
        choice = input("Retrain Dir RF? Y/n: ").strip().upper()
        if choice == 'Y':
            training_trades = get_detected_trades(training_trades, mag_models)
            dir_models = train_trade_direction(training_trades)
            save_rf_models(dir_models, 'dir')
        else:
            dir_models = load_rf_models('dir')
        filtered_test_trades = get_detected_trades(test_trades, mag_models)
        total_trades_taken, accuracy = test_trade_direction_model(filtered_test_trades, dir_models)
        symbols_amount = len(symbols)
        trades_per_hour = symbols_amount * 60
        total_hours = len(trades)/trades_per_hour
        trades_taken_per_hour = total_trades_taken/total_hours
        payout = 0.8
        hourly_return_percent = trades_taken_per_hour * (accuracy * payout - (1 - accuracy))
        print(f"{int(trades_taken_per_hour)} Trades Per Hour")
        print(f"{accuracy*100:.4f}% Avg Accuracy")
        print(f"{hourly_return_percent:.4f}% return per hour")
    elif choice == "M":
        run_dynamic_tuning()
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
