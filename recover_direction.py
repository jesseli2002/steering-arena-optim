# %%
import os
import sys
from pathlib import Path
from contextlib import ExitStack

import gc
import json
from dataclasses import dataclass

import einops
import numpy as np
import pandas as pd

import torch as t
from datasets import load_dataset
from dotenv import load_dotenv
from torch import Tensor
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# %%
device = t.device("cuda" if t.cuda.is_available() else "cpu")

if device != t.device("cuda"):
    print("\033[93mWarning: CUDA not available, using CPU. This will be slow.\033[0m")
dtype = t.bfloat16

t.cuda.set_device(0)  # ??? Possibly needed?

HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "Please set HF_TOKEN"

# %%
# Set to true when developing locally (to test pipeline); smaller models are used to fit on the GPU
LOCAL_DEV = True

if LOCAL_DEV:
    MODEL_NAME = "openai-community/gpt2-small"
    max_memory = None
else:
    MODEL_NAME = "allenai/Olmo-3-1125-32B"
    max_memory = None
    # max_memory = { # TODO - fill out once VastAI setup is chosen
    #     0: "5GiB",  # reserve room for experiments
    #     1: "10GiB",
    #     2: "10GiB",
    #     3: "9GiB",
    # }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
    max_memory=max_memory,
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

NUM_LAYERS = len(model.model.layers)
D_MODEL = model.config.hidden_size

print(f"Model: {MODEL_NAME}")
print(f"Layers: {NUM_LAYERS}, Hidden dim: {D_MODEL}")
