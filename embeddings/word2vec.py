# %%
import torch
import datasets

# %%
try:
    dataset = datasets.load_dataset("tweets_hate_speech_detection")
    print(f"dataset loaded successfully")
except Exception:
    print(f"problem loading the dataset")

# %%
print(f"{type(dataset)}")
