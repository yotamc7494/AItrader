# train_all_durations_for_multitask.py

import numpy as np
import random
import os
import tensorflow as tf

from data_manager.trade_storage import LoadTrades
from data_manager.data_preprocessing import BalanceTrades
from ml_pipeline.save_and_load import save_brains  # or your own saving function

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from ml_pipeline.multi_head_cnn_lstm import (
    build_cnn_lstm_multihead,
    build_multitask_dataset_for_duration
)

def train_all_durations_for_multitask(
    threshold_for_label=[0.0004, 0.0005, 0.0006, 0.0007, 0.0008],
    timesteps=5,
    learning_rate=0.0005,
    reg=0.0005,
    epochs=40,
    savedir="./saved_models"
):
    """
    1) Load & balance trades
    2) For each duration in [1..5], build dataset => (X, y_dir, y_mag)
    3) Train multi-head CNN–LSTM => direction_out, magnitude_out
    4) Evaluate on test set, storing 2 accuracies (dir & mag)
    5) Ask if you want to save final in-memory models
    """
    os.makedirs(savedir, exist_ok=True)

    all_trades = LoadTrades()  # e.g. 'retro_trades.pkl'
    random.shuffle(all_trades)
    balanced = BalanceTrades(all_trades)

    train_size = int(0.8 * len(balanced))
    train_trades = balanced[:train_size]
    test_trades  = balanced[train_size:]
    print(f"Train trades: {len(train_trades)}, Test trades: {len(test_trades)}")

    models = {}
    test_accuracies = {}  # we'll store a tuple (dir_acc, mag_acc) for each duration
    all_dir_preds = {}
    all_dir_truth = {}

    for d in range(1,6):
        dur_str = str(d)
        print(f"\n========== TRAINING MULTI-HEAD MODEL for DURATION {dur_str} ==========\n")
        dur_thresh = threshold_for_label[d-1]

        # Build dataset
        X_train, y_dir_train, y_mag_train = build_multitask_dataset_for_duration(
            train_trades, dur_str, dur_thresh
        )
        if len(X_train) == 0:
            print(f"No training samples for duration {dur_str}, skipping.")
            models[dur_str] = None
            continue

        # Build multi-head CNN–LSTM
        model = build_cnn_lstm_multihead(
            timesteps=timesteps,
            frame_shape=(15,6,6),
            learning_rate=learning_rate,
            reg=reg
        )

        model_path = os.path.join(savedir, f"entry_multitask_{dur_str}.keras")
        ckpt = ModelCheckpoint(
            model_path,
            monitor='val_dir_out_accuracy',  # we pick direction as primary or mag_out
            mode='max',
            save_best_only=True,
            verbose=1
        )
        callbacks = [
            EarlyStopping(patience=7, restore_best_weights=True, monitor='val_dir_out_accuracy', mode='max'),
            ReduceLROnPlateau(patience=3, factor=0.5, min_lr=1e-7, monitor='val_dir_out_accuracy', mode='max'),
            ckpt
        ]

        # Train
        hist = model.fit(
            X_train,
            {'dir_out': y_dir_train, 'mag_out': y_mag_train},
            validation_split=0.2,
            epochs=epochs,
            batch_size=16,  # or 8 if memory is tight
            callbacks=callbacks
        )

        # Evaluate
        X_test, y_dir_test, y_mag_test = build_multitask_dataset_for_duration(
            test_trades, dur_str, dur_thresh
        )
        if len(X_test)==0:
            print(f"No test samples for duration {dur_str}, skipping test.")
            models[dur_str] = model
            continue

        # Evaluate => returns { 'dir_out_loss':..., 'dir_out_accuracy':..., 'mag_out_loss':..., 'mag_out_accuracy':... }
        results = model.evaluate(X_test, {'dir_out': y_dir_test, 'mag_out': y_mag_test}, verbose=0, return_dict=True)
        dir_acc = results['dir_out_accuracy']
        mag_acc = results['mag_out_accuracy']
        print(f"Test Accuracy (dir) for Duration {dur_str}: {dir_acc:.2%}")
        print(f"Test Accuracy (mag) for Duration {dur_str}: {mag_acc:.2%}")

        test_accuracies[dur_str] = (dir_acc, mag_acc)
        models[dur_str] = model
        dir_out_pred, mag_out_pred = model.predict(X_test)
        # dir_out_pred => shape (N,2), mag_out_pred => shape(N,2)
        dir_classes = np.argmax(dir_out_pred, axis=1)  # 0 => SELL, 1 => BUY
        all_dir_preds[d] = dir_classes
        all_dir_truth[d] = y_dir_test

    # summary
    print("\n========== SUMMARY ACCURACIES ========== ")
    for d in range(1,6):
        dur_str = str(d)
        if dur_str in test_accuracies:
            (dir_acc, mag_acc) = test_accuracies[dur_str]
            print(f"Duration {dur_str}: dir_acc={dir_acc:.2%}, mag_acc={mag_acc:.2%}")
        else:
            print(f"Duration {dur_str}: no model or no test data")

    def agreement_accuracy(all_preds, all_truth, durations, k):
        """
        For each duration d, we only 'take the trade' if durations d..d+(k-1)
        all predict the same class for the same sample index.
        But if the arrays differ in length, we take the min length among them.

        Returns { d: (acc, coverage) } for each d.
          - acc: the fraction of those "agreed" samples that match d's ground truth
          - coverage: fraction of "agreed" samples out of that min length
        """
        result = {}

        for d in durations:
            needed = [d + offset for offset in range(k)]
            # check if d+(k-1) is in durations
            if not all(nd in durations for nd in needed):
                # can't do k-agreement for this d
                result[d] = (None, None)
                continue

            # gather predictions arrays for each needed duration
            preds_list = []
            lens = []
            for nd in needed:
                preds_list.append(all_preds[nd])
                lens.append(len(all_preds[nd]))
            # also get ground truth for duration d
            truth_d = all_truth[d]

            # find the min length across these k durations
            min_len = min(lens)
            # also ensure min_len <= len(truth_d)
            if len(truth_d) < min_len:
                min_len = len(truth_d)

            if min_len == 0:
                result[d] = (0.0, 0.0)
                continue

            correct_count = 0
            total_count = 0

            for i in range(min_len):
                # The class predicted by the first duration's preds
                first_class = preds_list[0][i]
                # check if all k durations match that
                agreed = True
                for j in range(1, k):
                    if preds_list[j][i] != first_class:
                        agreed = False
                        break

                if agreed:
                    total_count += 1
                    # check if that predicted class matches d's ground truth
                    if first_class == truth_d[i]:
                        correct_count += 1

            # coverage => fraction of min_len samples that had agreement
            if total_count == 0:
                acc = 0.0
                coverage = 0.0
            else:
                acc = correct_count / total_count
                coverage = total_count / min_len

            result[d] = (acc, coverage)

        return result

    print("\n========== K-AGREEMENT ACCURACIES (DIRECTION ONLY) ========== ")

    for k in [2, 3, 4, 5]:
        print(f"\n-- {k}-agreement (durations d..d+{k - 1} must match) --")
        agg = agreement_accuracy(all_dir_preds, all_dir_truth, [1, 2, 3, 4, 5], k)
        for d in [1, 2, 3, 4, 5]:
            acc, cov = agg[d]
            if acc is None:
                print(f"Duration {d}: can't do {k}-agreement.")
            else:
                print(f"Duration {d}: dir_acc={acc * 100:.2f}%, coverage={cov * 100:.2f}%")

    ans = input("\nDo you want to save the final in-memory models? (y/n): ").strip().lower()
    if ans=='y':
        save_brains(models, savedir)
    else:
        print("Skipping final model saving.")
    return models, test_accuracies



