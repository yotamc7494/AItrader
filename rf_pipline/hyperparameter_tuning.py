import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from data_manager.trade_storage import LoadTrades
from config import min_thresh, symbols


def tune_parameter(param_name, start_val, step, max_val, model_type, train_trades, val_trades, other_params, direction=False, mag_models=None):
    best_val = start_val
    best_score = -np.inf

    while start_val <= max_val:
        params = other_params.copy()
        params[param_name] = min(start_val, max_val)  # Ensure not exceeding max

        if model_type == 'mag':
            models = train_mag_model(train_trades, params)
            filtered_trades = get_detected_trades(val_trades, models)
            trades_taken = len(filtered_trades)
            correct_trades = sum(
                [1 for t in filtered_trades if (abs(t['results'][t['detected_for']]) > min_thresh[int(t['detected_for']) - 1])]
            )
            accuracy = correct_trades / trades_taken if trades_taken > 0 else 0
        else:
            if mag_models is None:
                raise ValueError("mag_models must be provided for direction model tuning")
            models = train_dir_model(train_trades, params)
            trades_taken, accuracy = test_trade_direction_model(val_trades, models)

        if accuracy > best_score:
            best_score = accuracy
            best_val = start_val
        start_val += step
    print(f"{param_name} Best val is: {best_val} | {best_score*100:.2f}% Accuracy")
    return best_val


def train_mag_model(trades, params):
    models = {}
    X = np.array([[x for xs in t['input'] for x in xs] for t in trades])

    for i in range(1, 5):
        duration = str(i + 1)
        y = np.array([1 if abs(t['results'][duration]) > min_thresh[i] else 0 for t in trades])
        trade_idx = np.where(y == 1)[0]
        no_trade_idx = np.where(y == 0)[0]
        min_size = min(len(trade_idx), len(no_trade_idx))

        idx = np.concatenate([
            np.random.choice(trade_idx, min_size, False),
            np.random.choice(no_trade_idx, min_size, False)
        ])

        Xb = X[idx]
        yb = y[idx]

        X_train, X_test, y_train, y_test = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)

        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            colsample_bytree=min(params['colsample_bytree'], 1.0),
            subsample=min(params['subsample'], 1.0),
            scale_pos_weight=params['scale_pos_weight'],
            min_child_weight=params['min_child_weight'],
            objective="binary:logistic",
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        models[duration] = model

    return models


def train_dir_model(trades, params):
    models = {}
    X = np.array([[x for xs in t['input'] for x in xs] for t in trades])

    for i in range(1, 5):
        duration = str(i + 1)
        y = np.array([1 if t['results'][duration] > 0 else 0 for t in trades])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        model = XGBClassifier(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            learning_rate=params['learning_rate'],
            gamma=params['gamma'],
            colsample_bytree=min(params['colsample_bytree'], 1.0),
            subsample=min(params['subsample'], 1.0),
            scale_pos_weight=params['scale_pos_weight'],
            min_child_weight=params['min_child_weight'],
            objective="binary:logistic",
            eval_metric="logloss"
        )
        model.fit(X_train, y_train)
        models[duration] = model

    return models


def run_dynamic_tuning():
    trades = LoadTrades()
    random.shuffle(trades)
    train_size = int(len(trades) * 0.7)
    val_size = int(len(trades) * 0.15)

    train_trades = trades[:train_size]
    val_trades = trades[train_size:train_size + val_size]
    test_trades = trades[train_size + val_size:]

    base_params = {
        'gamma': 0.5,
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.04,
        'colsample_bytree': 0.5,
        'subsample': 0.5,
        'scale_pos_weight': 0.1,
        'min_child_weight': 5,
    }

    param_steps = {
        'gamma': (0.1, 1.5),
        'n_estimators': (50, 500),
        'max_depth': (1, 6),
        'learning_rate': (0.01, 0.15),
        'colsample_bytree': (0.25, 1.0),
        'subsample': (0.25, 1.0),
        'scale_pos_weight': (0.1, 1.5),
        'min_child_weight': (1, 10),
    }

    print("🔹 Tuning Magnitude Model")
    tuned_mag_params = base_params.copy()
    for param, (step, max_val) in param_steps.items():
        tuned_mag_params[param] = tune_parameter(param, base_params[param], step, max_val, 'mag', train_trades, val_trades, tuned_mag_params)

    # Retrain on train + val for final test

    final_mag_models = train_mag_model(train_trades, tuned_mag_params)
    filtered_train_trades = get_detected_trades(val_trades, final_mag_models)
    print("\n🔹 Tuning Direction Model")
    tuned_dir_params = base_params.copy()
    for param, (step, max_val) in param_steps.items():
        tuned_dir_params[param] = tune_parameter(param, base_params[param], step, max_val, 'dir', train_trades, filtered_train_trades, tuned_dir_params, direction=True, mag_models=final_mag_models)

    filtered_train_trades = get_detected_trades(train_trades, final_mag_models)
    final_dir_models = train_dir_model(filtered_train_trades, tuned_dir_params)

    filtered_test_trades = get_detected_trades(test_trades, final_mag_models, print_out=True)
    total_trades_taken, accuracy = test_trade_direction_model(filtered_test_trades, final_dir_models, print_out=True)

    # Final metrics
    symbols_amount = len(symbols)
    trades_per_hour = symbols_amount * 60
    total_hours = len(test_trades) / trades_per_hour
    trades_taken_per_hour = total_trades_taken / total_hours
    payout = 0.8
    hourly_return_percent = trades_taken_per_hour * (accuracy * payout - (1 - accuracy))

    print(f"\n✅ {int(trades_taken_per_hour)} Trades Per Hour")
    print(f"✅ {accuracy * 100:.4f}% Avg Accuracy")
    print(f"✅ {hourly_return_percent:.4f}% return per hour")
    print("Final Trade Magnitude RF Params:", tuned_mag_params)
    print("Final Trade Direction RF Params:", tuned_dir_params)


def test_trade_direction_model(trades, models, print_out=False):
    total_trades = 0
    avg_accuracy = 0

    for i in range(1, 5):  # Durations 2–5 min
        duration = str(i + 1)
        model = models.get(duration)
        if not model:
            continue  # Skip if model missing

        # Prepare dataset
        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        y_true = np.array([1 if trade['results'][duration] > 0 else 0 for trade in trades])  # 1 = BUY, 0 = SELL

        # Predict with probability threshold
        y_pred_proba = model.predict_proba(X)
        y_pred = np.where(y_pred_proba[:, 1] > 0.8, 1, np.where(y_pred_proba[:, 0] > 0.8, 0, -1))

        valid_indices = y_pred != -1
        confident_preds = y_pred[valid_indices]
        confident_truths = y_true[valid_indices]

        total_trades_taken = len(confident_preds)
        overall_accuracy = accuracy_score(confident_truths, confident_preds) if total_trades_taken > 0 else 0

        total_trades += total_trades_taken
        avg_accuracy += overall_accuracy

        if print_out:
            print(f"{duration} min Accuracy: {overall_accuracy * 100:.4f}%")

    avg_accuracy = avg_accuracy / 4  # Average across durations
    return total_trades, avg_accuracy


def get_detected_trades(trades, trade_detector_models, print_out=False):
    detected_trades = []

    for i in range(1, 5):  # Process for 2-5 minutes (skipping 1-minute)
        duration = str(i + 1)
        model = trade_detector_models.get(duration)
        if not model:
            print(f"⚠️ No Trade Detector model found for {duration}-minute, skipping...")
            continue


        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        predictions = model.predict(X)  # Get predictions from the Trade Detector

        # Keep only trades where the model predicted TRADE (1)
        for idx in range(len(trades)):
            if predictions[idx] == 1:
                trade = trades[idx].copy()
                trade['detected_for'] = duration  # Add duration for tracking
                detected_trades.append(trade)
    if print_out:
        print(f"✅ Detected {len(detected_trades)} trades")
    return detected_trades