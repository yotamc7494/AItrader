# multi_head_cnn_lstm.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Dropout,
    GlobalAveragePooling2D, Dense, TimeDistributed, LSTM
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def build_frame_cnn(reg=0.0005, dropout_base=0.3):
    """
    A small sub-CNN that processes a single frame (e.g., shape (15,30,30) or (15,10,10)).
    Returns a feature vector (64-dim) for each frame.
    """
    frame_input = Input(shape=(15,6,6))  # or (15,10,10) if that's your subwindow shape

    x = Conv2D(32, (3,3), padding='same', activation='relu', kernel_regularizer=l2(reg))(frame_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_base)(x)

    x = Conv2D(64, (3,3), padding='same', activation='relu', kernel_regularizer=l2(reg))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_base+0.1)(x)

    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(reg))(x)
    x = Dropout(dropout_base+0.1)(x)

    # final frame-level features => shape (batch, 64)
    frame_features = Dense(64, activation='relu', kernel_regularizer=l2(reg))(x)

    return Model(frame_input, frame_features, name="FrameCNN")


def build_cnn_lstm_multihead(
    timesteps=5,
    frame_shape=(15,6,6),  # shape of each subwindow
    learning_rate=0.0005,
    reg=0.0005
):
    """
    Multi-head CNN–LSTM:
      - A TimeDistributed sub-CNN to encode each frame
      - LSTM over the sequence of frames
      - Two output heads:
         1) direction (2-class => SELL vs. BUY)
         2) magnitude (2-class => SMALL vs. BIG)
    Returns compiled model.
    """
    from tensorflow.keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout

    # Build sub-CNN for a single frame
    frame_cnn = build_frame_cnn(reg=reg, dropout_base=0.3)

    # Model input => shape: (batch, timesteps, 15, 30, 30) or (batch, timesteps, 15,10,10)
    x_in = Input(shape=(timesteps, frame_shape[0], frame_shape[1], frame_shape[2]))

    # Encode each frame => shape (batch, timesteps, 64)
    td_features = TimeDistributed(frame_cnn)(x_in)

    # LSTM
    x = LSTM(64)(td_features)  # => shape (batch,64)

    # optional dense & dropout
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)

    # HEAD 1: direction => 2-class
    dir_out = Dense(2, activation='softmax', name='dir_out')(x)

    # HEAD 2: magnitude => 2-class
    mag_out = Dense(2, activation='softmax', name='mag_out')(x)

    model = Model(inputs=x_in, outputs=[dir_out, mag_out])

    opt = Adam(learning_rate=learning_rate)

    # Use multi-output losses/metrics
    model.compile(
        optimizer=opt,
        loss={
            'dir_out': 'sparse_categorical_crossentropy',
            'mag_out': 'sparse_categorical_crossentropy'
        },
        metrics={
            'dir_out': ['accuracy'],
            'mag_out': ['accuracy']
        }
    )
    model.summary()
    return model


def determine_dir_mag_for_duration(trade, duration_str, threshold=0.0005):
    """
    Returns (dir_label, mag_label) or None if no valid data.
    dir_label: 0 => sell, 1 => buy
    mag_label: 0 => small, 1 => big
    """
    val = trade['results'].get(duration_str)
    if val is None:
        return None

    # direction
    dir_label = 1 if val > 0 else 0  # val==0 => treat as 0 (sell) or skip? up to you

    # magnitude
    val_abs = abs(val)
    mag_label = 1 if val_abs > threshold else 0

    return (dir_label, mag_label)


def chunk_gaf_into_3_subwindows(gaf_3d):
    subwindows = []
    for i in [0, 6, 12, 18, 24]:
        # slice both dims [i:i+10], so we get shape => (15,10,10)
        chunk = gaf_3d[:, i:i + 6, i:i + 6]
        subwindows.append(chunk)

    # stack into shape (3,15,10,10)
    return np.array(subwindows, dtype=gaf_3d.dtype)


def build_multitask_dataset_for_duration(trades, duration_str, threshold):
    """
    For each trade, we read a 4D array => shape (timesteps, 15, 30, 30).
    We also compute (dir_label, mag_label).
    Returns X => (N, timesteps, 15, 30, 30),
            y_dir => (N,),
            y_mag => (N,)
    """
    X_list = []
    dir_list = []
    mag_list = []

    for trade in trades:
        labels = determine_dir_mag_for_duration(trade, duration_str, threshold)
        if labels is None:
            continue
        (dir_label, mag_label) = labels

        data_4d = chunk_gaf_into_3_subwindows(trade["input"])  # shape => (timesteps, 15, 30, 30)

        X_list.append(data_4d)
        dir_list.append(dir_label)
        mag_list.append(mag_label)

    X_arr = np.array(X_list, dtype=np.float32)
    y_dir_arr = np.array(dir_list, dtype=np.float32)  # used with sparse_categorical_crossentropy => shape (N,)
    y_mag_arr = np.array(mag_list, dtype=np.float32)
    return X_arr, y_dir_arr, y_mag_arr
