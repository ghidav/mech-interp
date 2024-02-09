import torch as th
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict

from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns

from utils import convert_to_chat, get_activations, convert_to_alpaca, load_model
from utils import FastPCA

parser = argparse.ArgumentParser(description="PCA experiment.")

parser.add_argument('-tl', '--tl_model', type=str, help='TL model to load')
parser.add_argument('-hf', '--hf_model', type=str, help='HF model to load')
parser.add_argument('-c', '--component', type=str, help="Component from which activations should be taken")

parser.add_argument('-adp', '--adapter', type=str, help='Adapter to load', default="")
parser.add_argument('-chat', '--chat', type=str, help="To add the chat prompt (one of 'none', 'base', 'safe')", default='none')
parser.add_argument('-bs', '--batch_size', type=int, help='Batch size', default=32)

args = parser.parse_args()

# Creating dirs
model_folder = args.tl_model if args.adapter == "" else args.adapter
activ_path = os.path.join('activations', model_folder)
if not os.path.exists(activ_path):
    os.makedirs(activ_path)

image_path = os.path.join('images', model_folder)
if not os.path.exists(image_path):
    os.makedirs(image_path)

score_path = os.path.join('scores', model_folder)
if not os.path.exists(score_path):
    os.makedirs(score_path)

# Data loading
xs = "xsafety"
full = "blocked"

xs_df = pd.read_csv(f"data/{xs}.csv", index_col=0)
full_df = pd.read_csv(f"data/{full}.csv", index_col=0).reset_index(drop=True)

xs_prompts = xs_df['prompt']
full_prompts = full_df['prompt']

bs = args.batch_size
xs_nb = len(xs_prompts) // bs
full_nb = len(full_prompts) // bs

# Loading activations
recompute = True
try:
    xs_activations = th.load(os.path.join(activ_path, f"{xs}_{args.chat}_{args.component}.pt"))
    full_activations = th.load(os.path.join(activ_path, f"{full}_{args.chat}_{args.component}.pt"))

    nl = xs_activations.shape[1]
    recompute = False
except Exception as e:
    print("Failed to load activations. They will be recomputed...")

if recompute:
    # Model loading
    model = load_model(args.hf_model, args.tl_model, args.adapter, device='cuda', n_devices=4, dtype=th.bfloat16)
    model.eval()

    if 'llama' in args.tl_model.lower():
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id

    nl = len(model.blocks)

    if args.chat == 'none':
        pass
    elif args.chat == 'base':
        xs_prompts = convert_to_chat(model, xs_prompts, sys_prompt=False)
        full_prompts = convert_to_chat(model, full_prompts, sys_prompt=False)
    elif args.chat == 'safe':
        xs_prompts = convert_to_chat(model, xs_prompts, sys_prompt=True)
        full_prompts = convert_to_chat(model, full_prompts, sys_prompt=True)
    elif args.chat == 'alpaca':
        xs_prompts = convert_to_alpaca(xs_prompts)
        full_prompts = convert_to_alpaca(full_prompts)
    else:
        raise NotImplementedError
    
    print("Caching activations...")
    xs_activations = get_activations(model, xs_prompts, args.component, bs=bs).cpu()
    th.save(xs_activations, os.path.join(activ_path, f"{xs}_{args.chat}_{args.component}.pt"))

    full_activations = get_activations(model, full_prompts, args.component, bs=bs).cpu()
    th.save(full_activations, os.path.join(activ_path, f"{full}_{args.chat}_{args.component}.pt"))
    print("Activations cached.")

y_xs = xs_df['label'][:len(xs_activations)].values.astype(int)
y_full = full_df['label'][:len(full_activations)].values.astype(int)

# Train test split
X_train_xs, X_test_xs, y_train_xs, y_test_xs = train_test_split(xs_activations, y_xs, random_state=42, stratify=y_xs)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(full_activations, y_full, random_state=42, stratify=y_full)

# Computing PCA
print("Computing PCA...")
pca = FastPCA(n_components=10)
X_train_pca_xs2xs = []
X_test_pca_xs2xs = []
X_train_pca_full2xs = []
X_test_pca_full2xs = []

X_train_pca_xs2full = []
X_test_pca_xs2full = []
X_train_pca_full2full = []
X_test_pca_full2full = []

xs_subspaces = []
full_subspaces = []

for l in tqdm(range(nl)):
    X_train_pca_xs2xs.append(pca.fit_transform(X_train_xs[:, l])[None, ...])
    X_test_pca_xs2xs.append(pca.transform(X_test_xs[:, l])[None, ...])
    X_train_pca_full2xs.append(pca.transform(X_train_full[:, l])[None, ...])
    X_test_pca_full2xs.append(pca.transform(X_test_full[:, l])[None, ...])
    xs_subspaces.append(pca.components_[None, ...])

    X_train_pca_full2full.append(pca.fit_transform(X_train_full[:, l])[None, ...])
    X_test_pca_full2full.append(pca.transform(X_test_full[:, l])[None, ...])
    X_train_pca_xs2full.append(pca.transform(X_train_xs[:, l])[None, ...])
    X_test_pca_xs2full.append(pca.transform(X_test_xs[:, l])[None, ...])
    full_subspaces.append(pca.components_[None, ...])

xs_subspaces = th.cat(xs_subspaces, dim=0)
full_subspaces = th.cat(full_subspaces, dim=0) # [l N dm]

X_train_pca_xs2xs = th.cat(X_train_pca_xs2xs, 0)
X_test_pca_xs2xs = th.cat(X_test_pca_xs2xs, 0)
X_train_pca_full2xs = th.cat(X_train_pca_full2xs, 0)
X_test_pca_full2xs = th.cat(X_test_pca_full2xs, 0)

X_train_pca_xs2full = th.cat(X_train_pca_xs2full, 0)
X_test_pca_xs2full = th.cat(X_test_pca_xs2full, 0)
X_train_pca_full2full = th.cat(X_train_pca_full2full, 0)
X_test_pca_full2full = th.cat(X_test_pca_full2full, 0) # [l N 10]
print("PCA computed.")

# Modeling
print("Modeling started...")
rocs = defaultdict(list)
accs = defaultdict(list)

lr = LogisticRegression(penalty=None, verbose=0)

for nc in [1, 2, 5, 10]:
    for l in tqdm(range(nl)):
        # FULL -> XS
        lr.fit(X_train_pca_xs2xs[l, :, :nc], y_train_xs)
        y_hat_xs2xs = lr.predict(X_test_pca_xs2xs[l, :, :nc])
        y_proba_xs2xs = lr.predict_proba(X_test_pca_xs2xs[l, :, :nc])[:, 1]

        lr.fit(X_train_pca_full2xs[l, :, :nc], y_train_full)
        y_hat_full2xs = lr.predict(X_test_pca_full2xs[l, :, :nc])
        y_proba_full2xs = lr.predict_proba(X_test_pca_full2xs[l, :, :nc])[:, 1]

        rocs[f'{nc}_xs2xs'].append(roc_auc_score(y_test_xs, y_proba_xs2xs))
        rocs[f'{nc}_full2xs'].append(roc_auc_score(y_test_full, y_proba_full2xs))

        accs[f'{nc}_xs2xs'].append(accuracy_score(y_test_xs, y_hat_xs2xs))
        accs[f'{nc}_full2xs'].append(accuracy_score(y_test_full, y_hat_full2xs))

        ### XS -> FULL
        lr.fit(X_train_pca_full2full[l, :, :nc], y_train_full)
        y_hat_full2full = lr.predict(X_test_pca_full2full[l, :, :nc])
        y_proba_full2full = lr.predict_proba(X_test_pca_full2full[l, :, :nc])[:, 1]

        lr.fit(X_train_pca_xs2full[l, :, :nc], y_train_xs)
        y_hat_xs2full = lr.predict(X_test_pca_xs2full[l, :, :nc])
        y_proba_xs2full = lr.predict_proba(X_test_pca_xs2full[l, :, :nc])[:, 1]
        
        rocs[f'{nc}_xs2full'].append(roc_auc_score(y_test_xs, y_proba_xs2full))
        rocs[f'{nc}_full2full'].append(roc_auc_score(y_test_full, y_proba_full2full))

        accs[f'{nc}_xs2full'].append(accuracy_score(y_test_xs, y_hat_xs2full))
        accs[f'{nc}_full2full'].append(accuracy_score(y_test_full, y_hat_full2full))

pd.DataFrame(rocs, index=range(nl)).to_csv(os.path.join(score_path, f"rocs_{args.chat}_{args.component}.csv"))
pd.DataFrame(accs, index=range(nl)).to_csv(os.path.join(score_path, f"accs_{args.chat}_{args.component}.csv"))
print("Modeling completed and results stored.")

#### VISUALIZATIONS ####
print("Plotting started...")
scaler = StandardScaler()

cols = 4
rows = nl // 4

fig, ax = plt.subplots(4, 1, layout='constrained', figsize=(6, 6))

### Subspace alignment
def principal_angles(U, V):
    # Compute the SVD of the product of Q1 transpose and Q2
    _, singular_values, _ = th.linalg.svd(th.matmul(U, V.transpose(-1, -2)))

    # Principal angles are the arccosine of the singular values
    angles = th.arccos(th.clip(singular_values, -1, 1))

    # Convert to degrees, if necessary
    angles_degrees = th.rad2deg(angles)

    return angles, angles_degrees

for i, nc in enumerate([1, 2, 5, 10]):
    _, alignment_score = principal_angles(xs_subspaces[:, :nc], full_subspaces[:, :nc])

    sns.barplot(alignment_score.T, ax=ax[i])
plt.savefig(os.path.join(image_path, f'alignment_{args.chat}_{args.component}.png'), dpi=150)


### Histograms
# xs -> full
fig, ax = plt.subplots(rows, cols, layout='constrained', figsize=(12, 3*rows))
palette = sns.color_palette("bright", n_colors=4)

for l in range(nl):
    sns.kdeplot(
        pd.DataFrame({
            'activ': np.concatenate([
                scaler.fit_transform(X_train_pca_full2full [l, :, 0, None]),
                scaler.transform(X_train_pca_xs2full[l, :, 0, None])
                ], 0)[..., 0],
            'label': np.concatenate([y_train_xs+2, y_train_full])
        }), x='activ', hue='label', alpha=0.5, ax=ax[l//4, l%4], common_norm=False, palette=palette)

plt.savefig(os.path.join(image_path, f'kde_{args.chat}_{args.component}_xs2full.png'))

# full -> xs
fig, ax = plt.subplots(rows, cols, layout='constrained', figsize=(12, 3*rows))

for l in range(nl):
    sns.kdeplot(
        pd.DataFrame({
            'activ': np.concatenate([
                scaler.fit_transform(X_train_pca_xs2xs [l, :, 0, None]),
                scaler.transform(X_train_pca_full2xs[l, :, 0, None])
                ], 0)[..., 0],
            'label': np.concatenate([y_train_xs+2, y_train_full])
        }), x='activ', hue='label', alpha=0.5, ax=ax[l//4, l%4], common_norm=False, palette=palette)

plt.savefig(os.path.join(image_path, f'kde_{args.chat}_{args.component}_full2xs.png'))

# 2D kde

# First set of plots
fig, ax = plt.subplots(rows, cols, figsize=(12, 3*rows))
for l in range(nl):
    df = pd.DataFrame({
        'x': np.concatenate([
            scaler.fit_transform(X_train_pca_xs2full[l, :, 0, None]),
            scaler.transform(X_train_pca_full2full[l, :, 0, None])
        ], 0).ravel(),
        'y': np.concatenate([
            scaler.fit_transform(X_train_pca_xs2full[l, :, 1, None]),
            scaler.transform(X_train_pca_full2full[l, :, 1, None])
        ], 0).ravel(),
        'label': np.concatenate([y_train_xs+2, y_train_full])
    })

    sns.kdeplot(
        data=df, x='x', y='y', hue='label', alpha=0.5, ax=ax[l // 4, l % 4], common_norm=False, palette=palette
    )

plt.savefig(os.path.join(image_path, f'kde_{args.chat}_{args.component}_xs2full_2d.png'), dpi=150)

# Second set of plots
fig, ax = plt.subplots(rows, cols, figsize=(12, 3*rows))
for l in range(nl):
    df = pd.DataFrame({
        'x': np.concatenate([
            scaler.fit_transform(X_train_pca_xs2xs[l, :, 0, None]),
            scaler.transform(X_train_pca_full2xs[l, :, 0, None])
        ], 0).ravel(),
        'y': np.concatenate([
            scaler.fit_transform(X_train_pca_xs2xs[l, :, 1, None]),
            scaler.transform(X_train_pca_full2xs[l, :, 1, None])
        ], 0).ravel(),
        'label': np.concatenate([y_train_xs+2, y_train_full])
    })

    sns.kdeplot(
        data=df, x='x', y='y', hue='label', alpha=0.5, ax=ax[l // 4, l % 4], common_norm=False, palette=palette
    )

plt.savefig(os.path.join(image_path, f'kde_{args.chat}_{args.component}_full2xs_2d.png'), dpi=150)
print("Plotting completed.")