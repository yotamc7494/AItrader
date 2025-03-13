import pickle

def save_rf_models(models, filename="rf_models.pkl"):
    """
    Saves trained RF models to a file.

    Parameters:
        models (dict): Dictionary containing trained RF models.
        filename (str): File path for saving models.
    """
    with open(filename, "wb") as f:
        pickle.dump(models, f)
    print(f"✅ RF models saved to {filename}")

def load_rf_models(filename="rf_models.pkl"):
    """
    Loads trained RF models from a file.

    Parameters:
        filename (str): File path to the saved models.

    Returns:
        dict: Loaded RF models, or None if the file doesn't exist.
    """
    try:
        with open(filename, "rb") as f:
            models = pickle.load(f)
        print(f"✅ RF models loaded successfully from {filename}")
        return models
    except FileNotFoundError:
        print(f"❌ Model file {filename} not found. Train models first.")
        return None
