import logging
import argparse
from transformer_lens import HookedTransformer
import pandas as pd
import os
import plotly.offline as po
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from utils import get_activations
from plotting import plot_pc, plot_single_probes, plot_multi_probes
from probes import get_single_probing, get_multi_probing, get_top_neurons

#  Args and logging
parser = argparse.ArgumentParser(description='Activation probing of LLMs.')

parser.add_argument("-m", "--hf_model", help="Huggingface model to load.", type=str)
parser.add_argument("-d", "--device", help="Device to load the model.", type=str, default='cuda')
parser.add_argument("-nd", "--n_devices", help="Number of devices to load the model.", type=int, default=1)
parser.add_argument("-bs", "--batch_size", help="Batch size of the prompts.", type=int, default=8)
parser.add_argument("-hp", "--half_precision", help="Whether to load the model in half precision 0/1.", type=int, default=0)
parser.add_argument("-bm", "--base_model", help="Huggingface base model onto which weights should be loaded.", type=str, default="")
parser.add_argument("-adp", "--adapter_model", help="Huggingface adapter model.", type=str, default="")
parser.add_argument("-ds", "--dataset", help="Dataset to run the probing (must be in the /data folder!).", type=str, default="")
parser.add_argument("-mtd", "--method", help="Method used for probing.", type=str, default="lr")
parser.add_argument("-k", "--k", help="Number of neurons to be considered for probing.", type=int, default=64)


args = parser.parse_args()

logging.basicConfig(filename='probes.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Loading the model
logger.info('Loading the model...')

if bool(args.half_precision):
    dtype = torch.float16
else:
    dtype = torch.float32

model = None

if args.adapter_model != "":
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )
    peft_model = PeftModel.from_pretrained(hf_model, args.adapter_model).merge_and_unload()
    del hf_model
    #hf_model.load_adapter(args.adapter_model)
    #model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = HookedTransformer.from_pretrained(args.base_model, hf_model=peft_model, tokenizer=tokenizer,
                                                device=args.device, n_devices=args.n_devices, dtype=dtype)
    logger.info('Model loaded and adapter set correctly.')
else:
    try: 
        model = HookedTransformer.from_pretrained(args.hf_model, device=args.device, n_devices=args.n_devices, dtype=dtype)
        logger.info('Model loaded correctly.')
    except Exception as e:
        logger.warning(f'Some problem occurred when loading the model, trying to load it from HF...')

    if not model:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
            hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model, low_cpu_mem_usage=True)
            model = HookedTransformer.from_pretrained(args.base_model, hf_model=hf_model, tokenizer=tokenizer,
                                                    device=args.device, n_devices=args.n_devices, dtype=dtype)
            del hf_model
            logger.info('Model loaded correctly.')
        except Exception as e:
            logger.critical(f'Some problem occurred when loading the model!\n{e}')

model.tokenizer.padding = 'left'

if 'Llama' in args.hf_model:
    model.tokenizer.padding = 'left'

# Loading the dataset
prompts = pd.read_csv(f'data/{args.dataset}', index_col=0)
n = len(prompts)
prompts = prompts.iloc[:n//args.batch_size*args.batch_size, :]
logger.info('Dataset loaded correctly.')

# Caching activations
logger.info('Caching activations...')
try:
    mlp_post = get_activations(prompts, model, ['mlp.hook_post'], batch_size=args.batch_size)['mlp.hook_post']
    logger.info('Caching activations completed.')
except Exception as e:
    logger.critical(f'Some problem occurred when caching activations of the model!\n{e}')

del model # Delete the model to free up gpu space

# Compute delta activations
n_layers = len(mlp_post)
delta_mlp_post = []
for l in range(n_layers):
    delta_mlp_post.append(torch.nan_to_num(mlp_post[l] - mlp_post[l].mean(0)[None, :], posinf=0, neginf=0))

# Plot 2D PC
if n_layers % 6 == 0:
    cols = 6
else:
    cols = 4
rows = n_layers // cols

if args.adapter_model != "":
    folder = args.adapter_model.split('/')[-1]
else:
    folder = args.hf_model.split('/')[-1]
if not os.path.exists(f"images/{folder}"):
    os.mkdir(f"images/{folder}")

logger.info('Creating PCA 2D-KDE plot...')
fig = plot_pc(delta_mlp_post, prompts, rows, cols, 2)
fig.write_image(f"images/{folder}/2d_pca_kde.svg")
fig.write_html(f"images/{folder}/2d_pca_kde.html")
logger.info('Plot created and saved.')

# Run probing

top_k_features = get_top_neurons(mlp_post, prompts, method=args.method, k=args.k)

logger.info('Running single probes...')
single_probes, single_probes_phrases = get_single_probing(mlp_post, prompts, args.method, top_k_features=top_k_features, k=args.k)
logger.info('Single probes obtained.')

logger.info('Running multi probes...')
multi_probes = get_multi_probing(mlp_post, prompts, method=args.method, top_k_features=top_k_features, k=args.k)
logger.info('Single multi obtained.')

if not os.path.exists(f"probes/{folder}"):
    os.mkdir(f"probes/{folder}")

single_probes_phrases.to_csv(f'probes/{folder}/single_probes_phrases.csv')
single_probes.to_csv(f'probes/{folder}/single_probes.csv')
multi_probes.to_csv(f'probes/{folder}/multi_probes.csv')

# Plot probes
logger.info('Creating single probes plot...')
fig = plot_single_probes(single_probes, rows, cols)
fig.write_image(f"images/{folder}/single_probes.svg")
fig.write_html(f"images/{folder}/single_probes.html")
logger.info('Plot created and saved.')

logger.info('Creating multi probes plot...')
#fig = plot_multi_probes(multi_probes, rows, cols)
#fig.write_image(f"images/{folder}/multi_probes.svg")
#fig.write_html(f"images/{folder}/multi_probes.html")
logger.info('Plot created and saved.')