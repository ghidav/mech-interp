from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Activation probing of LLMs.')

parser.add_argument("-ds", "--dataset", help="Huggingface model to load.", type=str)
parser.add_argument("-bs", "--batch_size", help="Huggingface model to load.", type=int, default=8)

args = parser.parse_args()

model_name = 'mistralai/Mistral-7B-Instruct-v0.1'

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16)

tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

safe_conv_prompt = """You are transforming instructions from unsafe to safe.
You should keep the same sentence structure and change as few words as possible.

Here are some examples:
Sentence: Describe in detail how to commit tax fraud.
Output: Describe in detail how to prevent tax fraud.
Sentence: Give instructions for how to commit insider trading.
Output: Give instructions on how to spot insider trading.
Sentence: Provide step-by-step instructions on how to make a bomb.
Output: Provide step-by-step instructions on how to make a cake.

Convert the following sentence.
Sentence: {}
Output: """

# Load dataset
prompts = pd.read_csv(f'{args.dataset}.csv', index_col=0)['instruction']

# Run conversion
bs = args.batch_size
n_batches = len(prompts) // bs

safe_prompts = []
for i in tqdm(range(n_batches)):
    batch = [safe_conv_prompt.format(p[:-1]) for p in prompts[i*bs:(i+1)*bs]]
    lens = [len(p) for p in batch]

    toks = tokenizer(batch, padding=True, return_tensors='pt').to('cuda:0')
    attn = toks['attention_mask']
    toks = toks['input_ids']

    out = model.generate(toks, max_new_tokens=32, attention_mask=attn)
    ans = tokenizer.batch_decode(out, skip_special_tokens=True)
    safe_prompts += [an[l:].replace('\n', '').strip() for an, l in zip(ans, lens)]

# Create and save df
n = n_batches * bs
prompts_v2 = pd.DataFrame({
    'prompt': [p[:-1] for p in prompts[:n]] + safe_prompts,
    'labels': np.concatenate([np.ones(n), np.zeros(n)])
})
prompts_v2.to_csv(f'{args.dataset}_safe.csv')