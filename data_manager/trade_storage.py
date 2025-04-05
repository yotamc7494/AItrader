# data_manager/trade_storage.py
retro_trades = []
import os
import pickle

# Define global variables
CHUNK_SIZE = 5000

# Load function remains unchanged
def LoadTrades(TRADE_FOLDER = "C:\\Users\\Yotam\\Dropbox\\retro_trades"):
    if not os.path.exists(TRADE_FOLDER):
        print(f"⚠️ Folder {TRADE_FOLDER} not found.")
        return []

    all_files = sorted([f for f in os.listdir(TRADE_FOLDER) if f.endswith(".pkl")])
    all_trades = []
    total_loaded = 0
    bar_length = 30
    print(f"\n Loading Trades from {TRADE_FOLDER}")
    for idx, filename in enumerate(all_files):
        full_path = os.path.join(TRADE_FOLDER, filename)
        try:
            with open(full_path, 'rb') as f:
                trades_chunk = pickle.load(f)
            all_trades.extend(trades_chunk)
            total_loaded += len(trades_chunk)
            filled_length = int(((idx+1)/len(all_files))*bar_length)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\r[{bar}] {int(((idx+1)/len(all_files))*100)}% completed", end='', flush=True)
        except FileNotFoundError:
            print(f"\n⚠️ File {full_path} not found.")
        except Exception as e:
            print(f"\n❌ Error loading trades from {full_path}: {e}")

    print(f"\n✅ Finished loading trades. Total: {total_loaded} trades.")
    return all_trades


# Save function remains unchanged
def SaveTrades(trades, TRADE_FOLDER = "C:\\Users\\Yotam\\Dropbox\\retro_trades"):
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

