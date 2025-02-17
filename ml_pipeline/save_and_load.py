import os

def load_brains(savedir="./saved_models"):
    """
    Loads each of the 5 brain models from disk if present.
    Returns a dict: { '1': model1, '2': model2, ... }.
    If not found, entry is None.
    """
    import os
    import tensorflow as tf

    models = {}
    for d in range(1,6):
        model_path = os.path.join(savedir, f"entry_{d}_final.keras")
        if os.path.exists(model_path):
            print(f"Loading brain for duration {d} from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            models[str(d)] = model
        else:
            print(f"No saved model found for duration {d} at {model_path}.")
            models[str(d)] = None
    return models

def save_brains(models, savedir="./saved_models"):
    """
    Saves each trained model in 'models' dict to disk with name 'entry_{dur}_final.keras'.
    This is in addition to any checkpointing that happened during training.
    """
    os.makedirs(savedir, exist_ok=True)
    for dur_str, model in models.items():
        if model is not None:
            final_path = os.path.join(savedir, f"entry_{dur_str}_final.keras")
            model.save(final_path)
            print(f"Saved final model for duration {dur_str} -> {final_path}")
        else:
            print(f"No model for duration {dur_str}, skipping save.")