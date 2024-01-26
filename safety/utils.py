# Custom class to speed up PCA with GPUs
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer
from peft import PeftModel
import torch as th
from tqdm.auto import tqdm

def get_activations(model, prompts, component, bs=32):
    nl = len(model.blocks)
    activations = []

    for b in tqdm(range(len(prompts) // bs)):
        tokens = model.to_tokens(list(prompts[b*bs:(b+1)*bs]))

        with th.no_grad():
            _, cache = model.run_with_cache(tokens)

        try:
            activations.append(th.cat([cache[f'blocks.{l}.hook_{component}'][:, None, -1, :].cpu() for l in range(nl)], 1)) # [b l dm]
        except Exception as e:
            activations.append(th.cat([cache[f'blocks.{l}.{component}'][:, None, -1, :].cpu() for l in range(nl)], 1)) # [b l dm]
        del cache

    activations = th.cat(activations, 0) # [np l dm]

    return activations

def convert_to_chat(model, prompts, sys_prompt=False):
    new_prompts = []
    if sys_prompt:
        content = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    else:
        content = "You are a helpful assistant"
        
    for p in prompts:
        tokens = model.tokenizer.apply_chat_template([
            {
                "role": "system",
                "content": content,
            },
            {"role": "user", "content": p}
        ])

        new_prompts.append(model.to_string(tokens))

    return new_prompts

def convert_to_alpaca(prompts):
    alpaca_prompts = [f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{inst}\n\n### Response:\n" for inst in prompts]
    return alpaca_prompts

class FastPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None, center=True):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.singular_values_ = None
        self.center = center

    def fit(self, X, y=None):
        # Centering the data
        if self.center:
            self.mean_ = X.mean(axis=0)
        else:
            self.mean_ = th.zeros(X.shape[-1], device=X.device)
        
        X_centered = X - self.mean_
        
        # Performing SVD
        U, S, Vt = th.linalg.svd(X_centered.type(th.float32).to(X.device), full_matrices=False)
        
        # Storing singular values and components
        self.singular_values_ = S[:self.n_components]
        self.components_ = Vt[:self.n_components]

        return self

    def transform(self, X):
        # Check if fit has been called
        if self.components_ is None:
            raise RuntimeError("You must fit the model before transforming the data")

        # Centering the data
        X_centered = (X - self.mean_.to(X.device)).type(th.float32)
        
        # Projecting data onto principal components
        X_pca = X_centered @ self.components_.T.to(X.device)

        return X_pca.cpu()

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        pass

def load_model(hf_model_name, tl_model_name="", adapter_model="", device='cpu', n_devices=1, dtype=th.float32):
    print("Loading the model...")
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
        model = HookedTransformer.from_pretrained_no_processing(tl_model_name, hf_model=peft_model, tokenizer=tokenizer,
                                                  device=device, n_devices=n_devices, dtype=dtype)
    else:
        try:
            model = HookedTransformer.from_pretrained_no_processing(tl_model_name, device=device, n_devices=n_devices, dtype=dtype)
            print("Loaded model into HookedTransformer")
        except Exception as e:
            print(e)

        if not model:
            try:
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, low_cpu_mem_usage=True)
                print("Loaded model from hf. Attempting to load it to HookedTransformer")
                model = HookedTransformer.from_pretrained_no_processing(tl_model_name, hf_model=hf_model, tokenizer=tokenizer,
                                                              device=device, n_devices=n_devices, dtype=dtype)
                print("Loaded model into HookedTransformer")
                del hf_model

            except Exception as e:
                print(e)

    if 'llama' in tl_model_name.lower():
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    
    model.tokenizer.padding_side = 'left' 

    return model