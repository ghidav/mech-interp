import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def standardize(x):
    return (x - np.mean(x)) / np.std(x)

def get_activations(prompts, model, comps, batch_size=8):
    n_layers = len(model.blocks)
    activations = {comp: {i: [] for i in range(n_layers)} for comp in comps}

    for i in tqdm(range(len(prompts) // batch_size)):
        toks = model.to_tokens(prompts.iloc[i*batch_size:(i+1)*batch_size].prompt.to_list())
        _, cache = model.run_with_cache(toks)

        for comp in comps:
            for j in range(n_layers):
                activations[comp][j].append(cache.cache_dict[f'blocks.{j}.{comp}'][:, -1, :].cpu())
        
        del cache
    for comp in comps:
        for j in range(n_layers):
            activations[comp][j] = torch.cat(activations[comp][j], dim=0)

    return activations

def list_to_str(x):
    s = ''
    for i in list(x):
        s += str(i) + ' '

    return s[:-1]