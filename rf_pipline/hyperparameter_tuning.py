import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from data_manager.trade_storage import LoadTrades
from config import min_thresh, symbols

# Define dynamic tuning for one parameter

def tune_parameter(param_name, start_val, step, max_val, model_type, trades, final_test_trades, other_params, direction=False, mag_models=None):
    best_val = start_val
    best_score = -np.inf
    no_improve_count = 0

    while no_improve_count < 2 and start_val <= max_val:
        params = other_params.copy()
        params[param_name] = min(start_val, max_val)

        if model_type == 'mag':
            models = train_mag_model(trades, params)
            filtered_trades = get_detected_trades(final_test_trades, models)
            trades_taken = len(filtered_trades)
            correct_trades = sum([1 for t in filtered_trades if (abs(t['results'][t['detected_for']]) > min_thresh[int(t['detected_for']) - 1])])
            accuracy = correct_trades / trades_taken if trades_taken > 0 else 0
        else:
            if mag_models is None:
                raise ValueError("mag_models must be provided for direction model tuning")
            models = train_dir_model(trades, params)
            filtered_trades = get_detected_trades(final_test_trades, mag_models)
            trades_taken, accuracy = test_trade_direction_model(filtered_trades, models)

        if accuracy > best_score:
            best_score = accuracy
            best_val = start_val
            no_improve_count = 0
        else:
            no_improve_count += 1

        start_val += step

    return best_val

# Base training functions for mag and direction

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

# Master tuning loop

def run_dynamic_tuning():
    trades = LoadTrades()
    train_len = int(len(trades) * 0.8)
    training_trades = trades[:train_len]
    final_test_trades = trades[train_len:]

    base_params = {
        'gamma': 0.3,
        'n_estimators': 100,
        'max_depth': 2,
        'learning_rate': 0.02,
        'colsample_bytree': 0.4,
        'subsample': 0.4,
        'scale_pos_weight': 0.05,
        'min_child_weight': 2,
    }

    param_steps = {
        'gamma': (0.1, 2.0),
        'n_estimators': (50, 500),
        'max_depth': (1, 10),
        'learning_rate': (0.01, 0.2),
        'colsample_bytree': (0.1, 1.0),
        'subsample': (0.1, 1.0),
        'scale_pos_weight': (0.05, 3.0),
        'min_child_weight': (1, 10),
    }
    no_improvement = 0
    filtered_trades = []
    best_mag_accuracy = 0
    best_mag_params = base_params.copy()
    print("Tuning Mag Model")
    while no_improvement < 2:
        tuned_mag_params = base_params.copy()
        for param, (step, max_val) in param_steps.items():
            tuned_mag_params[param] = tune_parameter(param, base_params[param], step, max_val, 'mag',
                                                     training_trades, final_test_trades,best_mag_params)
        mag_models = train_mag_model(training_trades, tuned_mag_params)
        filtered_trades = get_detected_trades(final_test_trades, mag_models)
        trades_taken = len(filtered_trades)
        correct_trades = sum(
            [1 for t in filtered_trades if (abs(t['results'][t['detected_for']]) > min_thresh[int(t['detected_for']) - 1])])
        accuracy = correct_trades / trades_taken if trades_taken > 0 else 0
        if best_mag_accuracy < accuracy:
            best_mag_accuracy = accuracy
            no_improvement = 0
            best_mag_params = tuned_mag_params
        else:
            no_improvement += 1
        print(f"Found accuracy {accuracy * 100:.4f}% | Best is {best_mag_accuracy * 100:.4f}%")

    print("Tuned Mag model, Now training Direction Model")

    no_improvement = 0
    best_dir_accuracy = 0
    best_dir_params = base_params.copy()
    while no_improvement < 2:
        tuned_dir_params = base_params.copy()
        for param, (step, max_val) in param_steps.items():
            tuned_dir_params[param] = tune_parameter(param, base_params[param], step, max_val, 'dir',
                                                     filtered_trades, final_test_trades, best_dir_params, direction=True, mag_models=mag_models)
        dir_models = train_dir_model(training_trades, tuned_dir_params)
        filtered_trades = get_detected_trades(final_test_trades, mag_models)
        trades_taken, accuracy = test_trade_direction_model(filtered_trades, dir_models)
        if trades_taken > 9 and accuracy > best_dir_accuracy:
            no_improvement = 0
            best_dir_params = tuned_dir_params
            best_dir_accuracy = accuracy
        else:
            no_improvement += 1
        print(f"Found accuracy {accuracy * 100:.4f}% | Best is {best_dir_accuracy * 100:.4f}%")

    print("\n✅ Final Tuned Parameters:")
    filtered_trades = get_detected_trades(final_test_trades, mag_models, print_out=True)
    total_trades_taken, accuracy = test_trade_direction_model(filtered_trades, dir_models, print_out=True)
    symbols_amount = len(symbols)
    trades_per_hour = symbols_amount * 60
    total_hours = len(trades) / trades_per_hour
    trades_taken_per_hour = total_trades_taken / total_hours
    payout = 0.8
    hourly_return_percent = trades_taken_per_hour * (accuracy * payout - (1 - accuracy))
    print(f"{int(trades_taken_per_hour)} Trades Per Hour")
    print(f"{accuracy * 100:.4f}% Avg Accuracy")
    print(f"{hourly_return_percent:.4f}% return per hour")
    print("Trade Magnitude RF Params:", best_mag_params)
    print("Trade Direction RF Params:", best_dir_params)


def test_trade_direction_model(trades, models, print_out=False):
    total_trades = 0
    avg_accuracy = 0
    for i in range(1, 5):  # Durations 2–5 min
        duration = str(i + 1)
        model = models.get(duration)
        if not model:
            print(f"⚠️ No model found for {duration} min, skipping...")
            continue


        # Prepare dataset
        X = np.array([[x for xs in trade['input'] for x in xs] for trade in trades])
        y_true = np.array([1 if trade['results'][duration] > 0 else 0 for trade in trades])  # 1 = BUY, 0 = SELL

        # Predict with probability threshold
        y_pred_proba = model.predict_proba(X)
        y_pred = np.where(y_pred_proba[:, 1] > 0.8, 1, np.where(y_pred_proba[:, 0] > 0.8, 0, -1))

        # Evaluate only trades with confident predictions
        valid_indices = y_pred != -1
        confident_preds = y_pred[valid_indices]
        confident_truths = y_true[valid_indices]

        # Per-trade accuracy: list of 1 (correct) / 0 (incorrect)
        per_trade_accuracy = (confident_preds == confident_truths).astype(int).tolist()

        # Stats
        total_trades_taken = len(per_trade_accuracy)
        overall_accuracy = accuracy_score(confident_truths, confident_preds) if total_trades_taken > 0 else 0
        avg_accuracy += overall_accuracy
        total_trades += total_trades_taken
        if print_out:
            print(f"{duration} min Accuracy: {overall_accuracy * 100:.4f}%")
    return total_trades, avg_accuracy/4


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