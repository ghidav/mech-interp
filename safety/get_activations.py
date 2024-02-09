import torch as th
import pandas as pd
import argparse
import os

from utils import convert_to_chat, get_activations, convert_to_alpaca, load_model, convert_to_sbi

parser = argparse.ArgumentParser(description="PCA experiment.")

parser.add_argument('-tl', '--tl_model', type=str, help='TL model to load')
parser.add_argument('-hf', '--hf_model', type=str, help='HF model to load')
parser.add_argument('-c', '--component', type=str, help="Component from which activations should be taken")
parser.add_argument('-ds', '--dataset', type=str, help="Dataset to load")

parser.add_argument('-adp', '--adapter', type=str, help='Adapter to load', default="")
parser.add_argument('-chat', '--chat', type=str, help="To add the chat prompt (one of 'none', 'base', 'safe')", default='none')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=32)

args = parser.parse_args()

# Creating dirs
model_folder = args.hf_model if args.adapter == "" else args.adapter
activ_path = os.path.join('activations', model_folder)
if not os.path.exists(activ_path):
    os.makedirs(activ_path)

# Data loading
df = pd.read_csv(f"data/{args.dataset}.csv", index_col=0)

# Model loading
model = load_model(args.hf_model, args.tl_model, args.adapter, device='cuda', n_devices=4, dtype=th.bfloat16)
model.eval()

if 'llama' in args.tl_model.lower():
    model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

if args.chat == 'none':
    pass
elif args.chat == 'base':
    prompts = convert_to_chat(model, df['prompt'].values, sys_prompt=False)
elif args.chat == 'safe':
    prompts = convert_to_chat(model, df['prompt'].values, sys_prompt=True)
elif args.chat == 'alpaca':
    try:
        prompts = df.apply(lambda row: convert_to_alpaca(row['instruction'], row['input']), axis=1).values # Da sistemare
    except:
        prompts = df.apply(lambda row: convert_to_alpaca(row['instruction']), axis=1).values
elif args.chat == 'safety-by-imitation':
    prompts = df.apply(lambda row: convert_to_sbi(row['instruction']), axis=1).values
else:
    raise NotImplementedError

print("Caching activations...")
activations = get_activations(model, prompts, args.component, bs=args.batch_size).cpu()
th.save(activations, os.path.join(activ_path, f"{args.dataset}_{args.chat}_{args.component}.pt"))
print("Activations cached.")