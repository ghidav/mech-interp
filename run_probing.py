import logging
import argparse
import pandas as pd
import os
import torch

from utils import get_activations, load_model
from plotting import plot_pc, plot_single_probes, plot_multi_probes
from probes import get_single_probing, get_multi_probing, get_top_neurons

#  Args and logging
parser = argparse.ArgumentParser(description='Activation probing of LLMs.')

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
parser.add_argument("-mtd", "--method", help="Method used for probing.", type=str, default="lr_l2")
parser.add_argument("-single_k", "--single_k", help="Number of neurons to be considered for single probing.", type=int,
                    default=64)
parser.add_argument("-max_multi_k", "--max_multi_k", help="Max number of neurons to consider for multi probing.",
                    type=int, default=5)
parser.add_argument("-batch_k", "--batch_k", help="Number of batch neurons to be considered for multi probing.",
                    type=int, default=15)
parser.add_argument("-print_activations", "--print_activations", help="Whether to print activations for single probing.",
                    type=bool, default=True)
parser.add_argument("-adp", "--adapter_model", help="Huggingface adapter model.", type=str, default="")

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

model = load_model(args.hf_model, args.transformerlens_model, args.adapter_model,
                   device=args.device, n_devices=args.n_devices, dtype=dtype)

# Loading the dataset
prompts = pd.read_csv(f'data/{args.dataset}', index_col=0)
n = len(prompts)
prompts = prompts.iloc[:n // args.batch_size * args.batch_size, :]
logger.info('Dataset loaded correctly.')

# Caching activations
logger.info('Caching activations...')
try:
    mlp_post = get_activations(prompts, model, ['mlp.hook_post'], batch_size=args.batch_size)['mlp.hook_post']
    logger.info('Caching activations completed.')
except Exception as e:
    logger.critical(f'Some problem occurred when caching activations of the model!\n{e}')

del model

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
top_k_features = get_top_neurons(mlp_post, prompts, method=args.method, k=max(args.single_k, args.batch_k + args.max_multi_k - 1))

logger.info('Running single probes...')
single_probes, single_probes_phrases = get_single_probing(mlp_post, prompts, args.method, top_k_features=top_k_features, k=args.single_k, print_activations = args.print_activations)
logger.info('Single probes obtained.')

logger.info('Running multi probes...')
multi_probes = get_multi_probing(mlp_post, prompts, method=args.method, top_k_features=top_k_features, max_multi_k=args.max_multi_k, batch_k=args.batch_k)
logger.info('Single multi obtained.')

if not os.path.exists(f"probes/{folder}"):
    os.mkdir(f"probes/{folder}")

if single_probes_phrases is not None:
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
fig = plot_multi_probes(multi_probes, rows, cols)
fig.write_image(f"images/{folder}/multi_probes.svg")
fig.write_html(f"images/{folder}/multi_probes.html")
logger.info('Plot created and saved.')
