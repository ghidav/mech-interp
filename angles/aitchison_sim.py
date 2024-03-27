import argparse
import pandas as pd
import torch
import os
from tqdm import tqdm
from utils import load_model
from tuned_lens.nn.lenses import TunedLens, LogitLens, Unembed

from tuned_lens_utils import generate_aitchisons, extract_cb


os.environ['TOKENIZERS_PARALLELISM'] = "true"

#  Args and logging
parser = argparse.ArgumentParser(description='Compute Aitchison similarity from dataset.')

parser.add_argument("-m", "--hf_model", help="Huggingface model to load.", type=str)
parser.add_argument("-d", "--device", help="Device to load the model.", type=str, default='cuda')
parser.add_argument("-nd", "--n_devices", help="Number of devices to load the model.", type=int, default=1)
parser.add_argument("-bs", "--batch_size", help="Batch size of the prompts.", type=int, default=8)
parser.add_argument("-hp", "--half_precision", help="Whether to load the model in half precision 0/1.", type=int,
                    default=0)
parser.add_argument("-bm", "--transformerlens_model", help="Model as called in transformerlens, needs to be specified if different from hf_model.", type=str,
                    default="")
parser.add_argument("-ds", "--dataset", help="Dataset to run the probing (must be in the /data folder!).", type=str,
                    default="")
parser.add_argument("-adp", "--adapter_model", help="Huggingface adapter model.", type=str, default="")

args = parser.parse_args()

# Set output path
if args.adapter_model != "":
    folder = args.adapter_model.split('/')[-1]
else:
    folder = args.hf_model.split('/')[-1]
if not os.path.exists(f"images/aitchison/{folder}"):
    os.mkdir(f"images/aitchison/{folder}")

# Loading data 
prompts = pd.read_csv(f'data/{args.dataset}', index_col=0)['prompt']

# Loading the model
if bool(args.half_precision):
    dtype = torch.bfloat16
else:
    dtype = torch.float32

model = load_model(args.hf_model, args.transformerlens_model, args.adapter_model,
                   device=args.device, n_devices=args.n_devices, dtype=dtype)

tuned_lens = TunedLens.from_unembed_and_pretrained(
    unembed=Unembed(model),
    lens_resource_id="gpt2",
)
logit_lens = LogitLens.from_model(model)
tuned_lens.to('cuda:0')

# Compute similarities
logit_lens_cb, _ = extract_cb(model, prompts, logit_lens)
tuned_lens_cb, _ = extract_cb(model, prompts, tuned_lens)

simil_tuned_lens = generate_aitchisons(prompts, tuned_lens, tuned_lens_cb)
simil_logit_lens = generate_aitchisons(prompts, logit_lens, logit_lens_cb)

