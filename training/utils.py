import os


def split_runs(root, val_ratio=0.2):
    runs = sorted(os.listdir(root))  # list all runs

    split = int(len(runs) * (1 - val_ratio))

    train = runs[:split]  # first chunk
    val = runs[split:]    # remaining

    return train, val