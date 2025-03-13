# data_manager/trade_storage.py
"""
Functions to load and save trades from pickle,
and store the global retro_trades list if you wish.
"""

import pickle
import os

from config import TRADE_PKL

# Global trades list, as in your old code
retro_trades = []

import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField

# Define global variables
TRADE_FOLDER = "./retro_trades"
CHUNK_SIZE = 5000


# Function to decode GAF images back to time-series data
def decode_gaf(gaf_image):
    """
    Approximates the original numerical time-series data from a GAF (15,30,30) image.

    Parameters:
        gaf_image (np.array): (15,30,30) GAF representation.

    Returns:
        np.array: (15,30) array approximating original indicators over time.
    """
    try:
        # Initialize output array
        num_indicators, time_steps, _ = gaf_image.shape
        recovered_series = np.zeros((num_indicators, time_steps))

        # Extract the diagonal values (strongest correlation to original sequence)
        for i in range(num_indicators):
            diagonal_values = np.diagonal(gaf_image[i])

            # Convert from angular field back to original normalized values
            recovered_series[i] = np.cos(np.arccos(np.clip(diagonal_values, -1, 1)))  # Clip to avoid NaNs

        # Fit MinMaxScaler to normalize extracted values
        scaler = MinMaxScaler(feature_range=(-1, 1))
        recovered_series = scaler.fit_transform(recovered_series.T).T  # Fit & transform

        return recovered_series
    except Exception as e:
        print(f"❌ Error decoding GAF: {e}")
        return None


# Function to transform and save trades
def transform_and_save_trades(trades):
    """
    Transforms GAF-based trades into numerical time-series indicators and saves them.

    Parameters:
        trades (list): List of trade dictionaries containing GAF images.
    """
    transformed_trades = []

    for trade in trades:
        if 'input' in trade and isinstance(trade['input'], np.ndarray) and trade['input'].shape == (15, 30, 30):
            # Decode GAF image to (15,30) numerical time-series
            numerical_features = decode_gaf(trade['input'])

            if numerical_features is not None:
                transformed_trades.append({
                    'symbol': trade['symbol'],
                    'input': numerical_features,
                    'results': trade['results']
                })
    SaveTrades(transformed_trades)


# Load function remains unchanged
def LoadTrades():
    """
    Load all trade files from TRADE_FOLDER in batches, concatenating them.

    Returns a single list of all trades combined.
    """
    if not os.path.exists(TRADE_FOLDER):
        print(f"⚠️ Folder {TRADE_FOLDER} not found.")
        return []

    all_files = sorted([f for f in os.listdir(TRADE_FOLDER) if f.endswith(".pkl")])
    all_trades = []
    total_loaded = 0

    for filename in all_files:
        full_path = os.path.join(TRADE_FOLDER, filename)
        try:
            with open(full_path, 'rb') as f:
                trades_chunk = pickle.load(f)
            all_trades.extend(trades_chunk)
            total_loaded += len(trades_chunk)
            print(f"✅ Loaded {len(trades_chunk)} trades from {full_path}.")
        except FileNotFoundError:
            print(f"⚠️ File {full_path} not found.")
        except Exception as e:
            print(f"❌ Error loading trades from {full_path}: {e}")

    print(f"✅ Finished loading trades. Total: {total_loaded} trades.")
    return all_trades


# Save function remains unchanged
def SaveTrades(trades):
    """
    Save trades in folder TRADE_FOLDER in batches of CHUNK_SIZE.
    Overwrites any existing .pkl files in that folder.
    """
    if not os.path.exists(TRADE_FOLDER):
        os.makedirs(TRADE_FOLDER, exist_ok=True)

    # Remove old pkl files in the folder
    old_files = [f for f in os.listdir(TRADE_FOLDER) if f.endswith(".pkl")]
    for oldf in old_files:
        os.remove(os.path.join(TRADE_FOLDER, oldf))

    total_trades = len(trades)
    print(f"Saving {total_trades} trades in batches of {CHUNK_SIZE} to folder: {TRADE_FOLDER}")

    start_idx = 0
    file_index = 0

    while start_idx < total_trades:
        end_idx = min(start_idx + CHUNK_SIZE, total_trades)
        chunk = trades[start_idx:end_idx]

        filename = f"retro_trades_{file_index:04d}.pkl"
        full_path = os.path.join(TRADE_FOLDER, filename)

        try:
            with open(full_path, 'wb') as f:
                pickle.dump(chunk, f)
            print(f"✅ Saved {len(chunk)} trades to {full_path}")
        except Exception as e:
            print(f"❌ Error saving trades to {full_path}: {e}")

        start_idx += CHUNK_SIZE
        file_index += 1

    print(f"✅ Done saving {total_trades} trades in {file_index} file(s).")

