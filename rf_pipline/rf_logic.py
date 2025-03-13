from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from xgboost import XGBClassifier
from config import min_thresh


def train_rf_for_durations(trades):
    models = {}

    for i in range(1, 5):  # Train for each duration (1min to 5min)
        duration = str(i + 1)
        threshold = min_thresh[i]

        print(f"\n🔹 Training Trade Detector for {duration}-minute predictions with threshold {threshold:.6f}...")

        # Prepare dataset
        X = np.array([trade['input'].flatten() for trade in trades])
        y = np.array([1 if abs(trade['results'][duration]) > threshold else 0 for trade in trades])

        # Balance the dataset (force equal number of NO_TRADE and TRADE)
        trade_indices = np.where(y == 1)[0]
        no_trade_indices = np.where(y == 0)[0]
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
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            gamma=0.7,  # Reduce from 1.5 → allows slightly more TRADEs
            colsample_bytree=0.6,
            subsample=0.5,
            scale_pos_weight=0.2,  # Increase from 0.1 → allows more TRADEs
            min_child_weight=6,  # Lower from 10 → reduces over-filtering
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False
        )
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate Model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Trade Detector Accuracy ({duration} min): {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=["NO_TRADE", "TRADE"]))

        # Save model
        models[duration] = model

    return models


def get_detected_trades(trades, trade_detector_models):
    detected_trades = []

    for i in range(1, 5):  # Process for 2-5 minutes (skipping 1-minute)
        duration = str(i + 1)
        model = trade_detector_models.get(duration)
        if not model:
            print(f"⚠️ No Trade Detector model found for {duration}-minute, skipping...")
            continue

        print(f"\n🔹 Filtering detected trades for {duration}-minute...")

        X = np.array([trade['input'].flatten() for trade in trades])
        predictions = model.predict(X)  # Get predictions from the Trade Detector

        # Keep only trades where the model predicted TRADE (1)
        for idx in range(len(trades)):
            if predictions[idx] == 1:
                trade = trades[idx].copy()
                trade['detected_for'] = duration  # Add duration for tracking
                detected_trades.append(trade)

        print(f"✅ Detected {len(detected_trades)} trades for {duration}-minute")

    return detected_trades


def train_trade_direction(trades):
    models = {}

    for i in range(1, 5):  # Skip 1-minute model, train for 2-5 min
        duration = str(i + 1)

        print(f"\n🔹 Training High-Confidence Trade Direction Classifier for {duration}-minute predictions...")

        # Prepare dataset
        X = np.array([trade['input'].flatten() for trade in trades])
        y = np.array([1 if trade['results'][duration] > 0 else 0 for trade in trades])  # 1 = BUY, 0 = SELL

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train XGBoost Model
        model = XGBClassifier(
            n_estimators=250,
            max_depth=4,
            learning_rate=0.04,
            gamma=0.7,
            colsample_bytree=0.8,
            subsample=0.6,
            scale_pos_weight=1.0,  # Balanced training
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False
        )
        model.fit(X_train, y_train)

        # Make predictions with higher confidence threshold
        y_pred_proba = model.predict_proba(X_test)  # Get probability for both BUY (1) and SELL (0)
        y_pred = np.where(y_pred_proba[:, 1] > 0.7, 1, np.where(y_pred_proba[:, 0] > 0.7, 0, -1))
        # If probability(BUY) > 70%, predict BUY
        # If probability(SELL) > 70%, predict SELL
        # Else, reject the trade (-1 = NEUTRAL)

        # Remove rejected trades from evaluation
        valid_indices = y_pred != -1
        y_pred = y_pred[valid_indices]
        y_test = y_test[valid_indices]

        # Evaluate Model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ High-Confidence Trade Direction Accuracy ({duration} min): {accuracy:.4f}")
        print(classification_report(y_test, y_pred, target_names=["SELL", "BUY"]))

        # Save model
        models[duration] = model

    return models


