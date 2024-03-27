import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from openai import OpenAI
import time

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

# Text generation on multi-device (should be updated with topk)
def generate_text(model, prompt, max_new_tokens=1):
    toks = model.to_tokens(prompt)

    for i in tqdm(range(max_new_tokens)):
        with torch.no_grad():
            new_tok = model(toks)[0, -1, :].argmax().to('cuda:0')
        
        toks = torch.concat([toks, new_tok[None, None]], -1)

    return model.to_string(toks)

# Get top-k tokens with proba
def get_top_tokens(model, prompt, k=5):
    with torch.no_grad():
        logits = model(model.to_tokens(prompt)).softmax(-1).type(torch.float32).cpu().numpy()

    top_logits = np.argsort(logits[0,-1, :])[::-1]
    for i in range(k):
        print(f"'{model.to_string(top_logits[i])}'-{top_logits[i]}\tP: {np.round(logits[0, -1, top_logits[i]], 3)}")


class ContentModeration:
    def __init__(self, token):
        self.client = OpenAI(api_key=token)

    def get_hate(self, message):
        """
        Run content moderation on a single message
        :param message:
        :return:
        """
        response = self.client.moderations.create(
            input=message,
        )

        if response.results:
            return response.results[0].category_scores
        else:
            return None

    def content_moderation(self, messages):
        """
        Run content moderation on a list of messages
        :param messages:
        :return:
        """
        collect = []
        for o in tqdm(messages, total=len(messages)):
            collect.append(self.get_hate(o))
            time.sleep(0.5)

        return collect