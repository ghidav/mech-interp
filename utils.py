import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from transformer_lens import HookedTransformer, ActivationCache


def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


def get_activations(prompts, model, comps, batch_size=8):
    n_layers = len(model.blocks)
    activations = {comp: {i: [] for i in range(n_layers)} for comp in comps}

    for i in tqdm(range(len(prompts) // batch_size)):
        toks = model.to_tokens(prompts.iloc[i * batch_size:(i + 1) * batch_size].prompt.to_list())
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


def probe_to_ablation_dict(probe_dataframe, topk, sort_by: str = "F1"):
    ablation_dict = {}
    if sort_by not in ["F1", "P"]:
        raise ValueError("Invalid sorting value")
    else:
        top_layer_neuron = probe_dataframe.sort_values(by=sort_by).head(topk)[["L", "N"]]
        for layer in top_layer_neuron['L'].unique():
            ablation_dict[int(layer)] = top_layer_neuron[top_layer_neuron["L"] == layer]["N"].astype(int).tolist()
        return ablation_dict


def load_model(hf_model_name, tl_model_name="", adapter_model="", device='cpu', n_devices=1, dtype=torch.float32):
    print("Loading model...")
    model = None
    if tl_model_name == "": tl_model_name = hf_model_name

    if adapter_model != "":
        hf_model = AutoModelForCausalLM.from_pretrained(
            hf_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        peft_model = PeftModel.from_pretrained(hf_model, adapter_model).merge_and_unload()
        del hf_model

        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        model = HookedTransformer.from_pretrained(tl_model_name, hf_model=peft_model, tokenizer=tokenizer,
                                                  device=device, n_devices=n_devices, dtype=dtype)
    else:
        try:
            model = HookedTransformer.from_pretrained(tl_model_name, device=device, n_devices=n_devices, dtype=dtype)
            print("Loaded model into HookedTransformer")
        except Exception as e:
            print(e)

        if not model:
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
                print("Loaded model from hf. Attempting to load it to HookedTransformer")
                model = HookedTransformer.from_pretrained(tl_model_name, hf_model=hf_model, tokenizer=tokenizer,
                                                          device=device, n_devices=n_devices, dtype=dtype)
                print("Loaded model into HookedTransformer")
                del hf_model

            except Exception as e:
                print(e)
    model.tokenizer.padding = 'left'  # TO CHECK!

    return model