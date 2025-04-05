from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score
import numpy as np
from xgboost import XGBClassifier
from config import min_thresh, symbols
from rf_pipline.load_and_save import load_rf_models, save_rf_models
import random
from data_manager.trade_storage import LoadTrades
from collections import defaultdict
DIR_CONFIDENT = 0.97
MAG_CONFIDENT = 0.7


def train_rf_for_durations(trades):
    models = {}

    for i in range(1, 5):  # Train for each duration (1min to 5min)
        duration = str(i + 1)
        threshold = min_thresh[i]

        print(f"\n🔹 Training Trade Detector for {duration}-minute predictions with threshold {threshold:.6f}...")

        # Prepare dataset
        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        y = np.array([1 if abs(trade['results'][duration]) > threshold else 0 for trade in trades])

        # Balance the dataset (force equal number of NO_TRADE and TRADE)
        trade_indices = np.where(y == 1)[0]
        no_trade_indices = np.where(y == 0)[0]
        ratio = len(no_trade_indices)/len(trade_indices)
        min_class_size = min(len(trade_indices), len(no_trade_indices))

        balanced_indices = np.concatenate([
            np.random.choice(trade_indices, min_class_size, replace=False),
            np.random.choice(no_trade_indices, min_class_size, replace=False)
        ])

        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42,
                                                            stratify=y_balanced)

        # Train XGBoost with class weighting to prioritize trade detection
        model = XGBClassifier(
            n_estimators=450,
            max_depth=3,
            learning_rate=0.02,
            gamma=0.9,
            colsample_bytree=0.6,
            subsample=0.6,
            scale_pos_weight=0.4,
            min_child_weight=6,
            objective="binary:logistic",
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred_proba = model.predict_proba(X_test)  # Get probability for both BUY (1) and SELL (0)
        y_pred = np.where(y_pred_proba[:, 1] > MAG_CONFIDENT, 1, 0)
        accuracy = precision_score(y_test, y_pred, pos_label=1)
        print(f"✅ Trade Detector Accuracy ({duration} min): {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=["NO_TRADE", "TRADE"]))

        # Save model
        models[duration] = model

    return models


def get_detected_trades(trades, trade_detector_models):
    detected_trades = {}
    for i in range(1, 5):  # Process for 2-5 minutes (skipping 1-minute)
        duration = str(i + 1)
        detected_trades[duration] = []
        model = trade_detector_models.get(duration)
        if not model:
            print(f"⚠️ No Trade Detector model found for {duration}-minute, skipping...")
            continue

        print(f"\n🔹 Filtering detected trades for {duration}-minute...")

        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        y_pred_proba = model.predict_proba(X)
        predictions = np.where(y_pred_proba[:, 1] > MAG_CONFIDENT, 1, np.where(y_pred_proba[:, 0] > MAG_CONFIDENT, 0, -1))
        #predictions = model.predict(X)
        # Keep only trades where the model predicted TRADE (1)\
        avg = 0
        for idx in range(len(trades)):
            if predictions[idx] == 1:
                trade = trades[idx].copy()
                trade['detected_for'] = duration  # Add duration for tracking
                detected_trades[duration].append(trade)
                avg += abs(trade['results'][duration])
        print(f"✅ Detected {len(detected_trades[duration])} trades for duration {duration} | avg: {avg/len(detected_trades[duration])}")
    return detected_trades


def filter_trades_by_thresh(trades):
    out = {}
    for i in range(1, 5):
        duration = str(i+1)
        duration_trades = [trade for trade in trades if abs(trade['results'][duration]) > min_thresh[i]]
        out[duration] = duration_trades
        print(f"✅ Detected {len(out[duration])} trades for duration {duration}")
    return out


def test_mag_models(trades, models):
    for i in range(1, 5):  # For durations 2 to 5 min
        duration = str(i + 1)
        threshold = min_thresh[i]

        print(f"\n🔹 Testing Trade Detector for {duration}-minute predictions with threshold {threshold:.6f}...")

        # Prepare dataset
        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        y = np.array([1 if abs(trade['results'][duration]) > threshold else 0 for trade in trades])

        model = models[duration]

        # Predict probabilities
        y_pred_proba = model.predict_proba(X)
        y_pred = (y_pred_proba[:, 1] > MAG_CONFIDENT).astype(int)

        # Filter only predicted TRADEs
        predicted_trades = y_pred == 1
        y_test = y[predicted_trades]
        y_pred = y_pred[predicted_trades]

        if len(y_test) == 0:
            print("⚠️ No trades were detected by the model.")
            continue

        # Evaluate precision
        precision = precision_score(y_test, y_pred, pos_label=1)
        print(f"✅ TRADE Precision ({duration} min): {precision*100:.2f}%")
        print(f"🧪 Support: {len(y_test)} | True Positives: {np.sum(y_test)} | False Positives: {len(y_test) - np.sum(y_test)}")


def train_trade_direction(duration_trades):
    models = {}

    for i in range(1, 5):  # Train for 2–5 min durations
        duration = str(i + 1)
        trades = duration_trades[duration]
        print(f"\n🔹 Training High-Confidence Trade Direction Classifier for {duration}-minute predictions...")

        # Prepare dataset
        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        y = np.array([1 if trade['results'][duration] > 0 else 0 for trade in trades])  # 1 = BUY, 0 = SELL

        # Train/Validation/Test Split (70/15/15)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

        # Train XGBoost Model
        model = XGBClassifier(
            n_estimators=1000,  # More trees to help low LR
            max_depth=6,  # More capacity to learn
            learning_rate=0.05,  # Stable LR
            gamma=0.3,  # Less regularization
            colsample_bytree=0.8,
            subsample=0.8,
            scale_pos_weight=1,
            objective="binary:logistic",
            eval_metric="logloss",
            verbosity=0
        )
        model.fit(X_train, y_train)

        # --- Evaluate on Test Set ---
        test_pred_proba = model.predict_proba(X_test)
        test_pred = np.where(test_pred_proba[:, 1] > DIR_CONFIDENT, 1, np.where(test_pred_proba[:, 0] > DIR_CONFIDENT, 0, -1))
        test_valid_indices = test_pred != -1
        test_pred = test_pred[test_valid_indices]
        y_test_filtered = y_test[test_valid_indices]

        test_accuracy = accuracy_score(y_test_filtered, test_pred)
        print(f"✅ Test Accuracy ({duration} min): {test_accuracy:.4f}")

        unique_classes = np.unique(y_test_filtered)
        label_map = {0: "SELL", 1: "BUY"}
        target_names = [label_map[c] for c in unique_classes]

        print(classification_report(y_test_filtered, test_pred, labels=unique_classes, target_names=target_names))

        # Save model
        models[duration] = model

    return models


def test_trade_direction_model(duration_trades, models):
    executed_trades = []  # Store trades with confident predictions
    avg_accuracy = 0

    for i in range(1, 5):  # Durations 2–5 min
        duration = str(i + 1)
        trades = duration_trades[duration]
        model = models.get(duration)
        if not model:
            print(f"⚠️ No model found for {duration} min, skipping...")
            continue

        print(f"\n🔹 Testing Trade Direction Classifier for {duration}-minute predictions...")

        # Prepare dataset
        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        y_true = np.array([1 if trade['results'][duration] > 0 else 0 for trade in trades])  # 1 = BUY, 0 = SELL

        # Predict with probability threshold
        y_pred_proba = model.predict_proba(X)
        y_pred = np.where(y_pred_proba[:, 1] > DIR_CONFIDENT, 1, np.where(y_pred_proba[:, 0] > DIR_CONFIDENT, 0, -1))

        # Evaluate only trades with confident predictions
        valid_indices = y_pred != -1
        confident_preds = y_pred[valid_indices]
        confident_truths = y_true[valid_indices]
        confident_trades = [trades[idx] for idx, valid in enumerate(valid_indices) if valid]

        # Per-trade accuracy
        per_trade_accuracy = (confident_preds == confident_truths).astype(int).tolist()
        for idx, trade in enumerate(confident_trades):
            trade['direction_correct'] = per_trade_accuracy[idx]
            trade['direction_pred'] = confident_preds[idx]
            trade['duration_tested'] = duration
            executed_trades.append(trade)

        # Stats
        total_trades_taken = len(per_trade_accuracy)
        overall_accuracy = accuracy_score(confident_truths, confident_preds) if total_trades_taken > 0 else 1
        print(f"{duration} min Accuracy: {overall_accuracy * 100:.4f}% | {total_trades_taken} Trades")
        avg_accuracy += overall_accuracy * total_trades_taken

    return executed_trades, avg_accuracy / len(executed_trades)


def analyze_trades_by_symbol(trades):
    symbol_counts = defaultdict(int)
    correct_counts = defaultdict(int)

    for trade in trades:
        symbol_str = trade['symbol']['symbol']
        symbol_counts[symbol_str] += 1
        correct_counts[symbol_str] += trade['direction_correct']

    total_trades = sum(symbol_counts.values())

    # Calculate appearance % and accuracy %
    appearance_percentages = {symbol: (count / total_trades) * 100 for symbol, count in symbol_counts.items()}
    accuracy_percentages = {}
    for symbol in symbol_counts:
        count = symbol_counts[symbol]
        correct = correct_counts[symbol]
        accuracy_percentages[symbol] = (correct / count) * 100 if count > 0 else 0

    # Compute std deviations
    appearance_std = np.std(list(appearance_percentages.values()))
    valid_accuracies = list(accuracy_percentages.values())
    accuracy_std = np.std(valid_accuracies) if len(valid_accuracies) > 1 else 0.0

    # Sort symbols by appearance and accuracy
    top_appearance = sorted(appearance_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
    bottom_appearance = sorted(appearance_percentages.items(), key=lambda x: x[1])[:3]
    top_accuracy = sorted(accuracy_percentages.items(), key=lambda x: x[1], reverse=True)[:3]
    bottom_accuracy = sorted(accuracy_percentages.items(), key=lambda x: x[1])[:3]

    print(f"📊 Trade Distribution: {total_trades} trades | Std Deviation (Appearance): {appearance_std:.2f}%")
    print("🔼 Top 3 Symbols by Appearance:")
    for symbol, percent in top_appearance:
        acc = accuracy_percentages[symbol]
        print(f"  {symbol}: {percent:.2f}% - {acc:.2f}% Accuracy")

    print("🔽 Bottom 3 Symbols by Appearance:")
    for symbol, percent in bottom_appearance:
        acc = accuracy_percentages[symbol]
        print(f"  {symbol}: {percent:.2f}% - {acc:.2f}% Accuracy")

    print(f"\n📊 Accuracy Std Deviation: {accuracy_std:.2f}%")
    print("✅ Top 3 Symbols by Accuracy:")
    for symbol, acc in top_accuracy:
        print(f"  {symbol}: {acc:.2f}%")

    print("❌ Bottom 3 Symbols by Accuracy:")
    for symbol, acc in bottom_accuracy:
        print(f"  {symbol}: {acc:.2f}%")


def train_models():
    trades = LoadTrades("C:\\Users\\Yotam\\Desktop\\fetched trades")
    random.shuffle(trades)
    train_length = int(len(trades) * 0.8)
    training_trades = trades[:train_length]
    test_trades = trades[train_length:]
    choice = input("Retrain Mag RF? Y/n: ").strip().upper()
    if choice == 'Y':
        mag_models = train_rf_for_durations(training_trades)
        save_rf_models(mag_models, 'mag')
    else:
        mag_models = load_rf_models('mag')
    test_mag_models(test_trades, mag_models)
    choice = input("Retrain Dir RF? Y/n: ").strip().upper()
    if choice == 'Y':
        training_trades = filter_trades_by_thresh(training_trades)
        dir_models = train_trade_direction(training_trades)
        save_rf_models(dir_models, 'dir')
    else:
        dir_models = load_rf_models('dir')
    filtered_test_trades = get_detected_trades(test_trades, mag_models)
    total_trades_taken, accuracy = test_trade_direction_model(filtered_test_trades, dir_models)
    symbols_amount = len(symbols)
    trades_per_hour = symbols_amount * 60 * len(mag_models.keys())
    total_hours = len(test_trades) / trades_per_hour
    trades_taken_per_hour = len(total_trades_taken) / total_hours
    payout = 0.8
    hourly_return_percent = trades_taken_per_hour * (accuracy * payout - (1 - accuracy))
    print(f"{int(trades_taken_per_hour)} Trades Per Hour")
    print(f"{accuracy * 100:.4f}% Avg Accuracy")
    print(f"{hourly_return_percent:.4f}% return per hour")
    choice = input("Show Symbols Distribution? Y/n: ").strip().upper()
    if choice == "Y":
        analyze_trades_by_symbol(total_trades_taken)
