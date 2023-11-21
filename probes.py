import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from scipy.stats import gaussian_kde
from collections import OrderedDict

from utils import list_to_str

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Compute PCA from activations
def get_pca(activations, k=10):
    pca = PCA()
    svds = {}

    for l in tqdm(range(len(activations))):
        pca.fit(activations[l])
        svds[l] = np.cumsum(pca.explained_variance_ratio_)

    return pd.DataFrame(svds).T.sort_values(0).head(k).T


# Get train test split based on layer and stratified for the response
def get_train_test(activations, prompts, layer):
    X = activations[layer].numpy()
    y = prompts.labels.to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    return X_train, X_test, y_train, y_test


# Get train test split based on layer and stratified for the response
def get_train_test_with_prompts(activations, prompts, layer):
    X = activations[layer].numpy()
    y = prompts.apply(lambda row: (row['prompt'], row['labels']), axis=1).to_numpy()
    classes = prompts.labels.to_numpy()
    X_train, X_test, res_y_train, res_y_test = train_test_split(X, y, stratify=classes, random_state=42)
    y_train = np.array([item[1] for item in res_y_train])
    y_test = np.array([item[1] for item in res_y_test])
    prompts_test = np.array([item[0] for item in res_y_test])

    return X_train, X_test, y_train, y_test, prompts_test


# Get the top-k neurons based on coef magnitude
@ignore_warnings(category=ConvergenceWarning)
def get_top_neurons(activations, prompts, method, k=64, **kwargs):
    top_k_features = []
    n = activations[0].shape[-1]

    if method == 'lr':
        model = LogisticRegression(n_jobs=-1, verbose=False, max_iter=250, **kwargs)
    elif method == 'svc':
        model = LinearSVC(loss='hinge', verbose=False, **kwargs)

    for j in tqdm(range(len(activations))):
        # Get train test
        X_train, _, y_train, _ = get_train_test(activations, prompts, j)

        model.fit(X_train, y_train)

        imps = pd.DataFrame({
            'layer': [j for i in range(n)],
            'neuron': np.arange(n),
            'importance': np.abs(model.coef_[0])
        })

        top_k_features.append(imps.sort_values(by='importance').iloc[-k:])

    return pd.concat(top_k_features)


@ignore_warnings(category=ConvergenceWarning)
def log_reg_score(X_train, y_train, X_test, y_test, **kwargs):
    lr = LogisticRegression(**kwargs)
    lr.fit(X_train, y_train)

    y_pred_prob = lr.predict(X_test)
    y_pred = np.round(y_pred_prob)

    return (precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0))


@ignore_warnings(category=ConvergenceWarning)
def log_reg_score_with_prompts(X_train, y_train, X_test, y_test, prompts_test, **kwargs):
    lr = LogisticRegression(**kwargs)
    lr.fit(X_train, y_train)

    y_pred_prob = lr.predict(X_test)
    y_pred = np.round(y_pred_prob)

    positively_predicted = [prompt for i, prompt in enumerate(prompts_test) if
                            y_test[i] == 1 and y_pred[i] == 1]

    return (precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0),
            positively_predicted)


def get_single_probing(activations, prompts, method, top_k_features=None, k=64):
    scores_df = {
        'L': [],
        'N': [],
        'P': [],
        'R': [],
        'F1': []
    }

    phrases_df = {
        'L': [],
        'N': [],
        'Phrases': [],
    }

    if top_k_features is None:
        top_k_features = get_top_neurons(activations, prompts, method=method, k=k)

    for i in tqdm(range(len(activations) * k)):
        l, n, *_ = top_k_features.iloc[i]

        X_train, X_test, y_train, y_test, prompts_test = get_train_test_with_prompts(activations, prompts, int(l))

        p, r, predicted = log_reg_score_with_prompts(X_train[:, int(n), None], y_train, X_test[:, int(n), None], y_test,
                                                     prompts_test)

        scores_df['L'].append(l)
        scores_df['N'].append(n)
        scores_df['P'].append(p)
        scores_df['R'].append(r)
        scores_df['F1'].append(2 * p * r / (p + r + 1e-9))

        phrases_df['L'].append(l)
        phrases_df['N'].append(n)
        phrases_df['Phrases'].append(predicted)

    return pd.DataFrame(scores_df), pd.DataFrame(phrases_df)


@ignore_warnings(category=ConvergenceWarning)
def get_C_value(activations, prompts, start, max_k, method, inc=0.25, k=64):
    C = start
    layers = {}

    stop = True
    print(f'Search for {max_k} started!')
    while stop:

        # regression for feature selection
        lr1 = LogisticRegression(
            class_weight='balanced',
            penalty='l1', solver='saga', n_jobs=-1, verbose=False, C=np.exp(C))

        # regression for scoring 
        lr2 = LogisticRegression(verbose=False)
        # class_weight='balanced', penalty='l2', solver='saga', n_jobs=-1)

        top_k_features = get_top_neurons(activations, prompts, method=method, k=k)

        for l in range(len(activations)):

            sub_df = top_k_features[top_k_features['layer'] == l]
            neurons = sub_df['neuron'].to_numpy()

            X_train, X_test, y_train, y_test = get_train_test(activations, prompts, int(l))

            # fit fs-lr
            fit1 = lr1.fit(X_train[:, neurons], y_train)

            coefs = fit1.coef_[0]

            if l not in layers:
                if np.sum(coefs > 0) >= max_k:
                    idxs = np.argsort(coefs)  # find relevant neurons

                    # fit sc-lr
                    fit2 = lr2.fit(X_train[:, neurons[idxs[:max_k]]], y_train)

                    y_pred_prob = lr2.predict(X_test[:, neurons[idxs[:max_k]]])
                    y_pred = np.round(y_pred_prob)

                    p = precision_score(y_test, y_pred, zero_division=0)
                    r = recall_score(y_test, y_pred, zero_division=0)

                    layers[l] = (neurons[idxs[:max_k]], p, r, 2 * p * r / (p + r))

        C += inc
        if len(layers) == len(activations):
            stop = False

    od = OrderedDict(sorted(layers.items()))
    return list(od.values())


def get_multi_probing(activations, prompts, method, k=64):
    scores_dfs = []
    optimal_C = {i: get_C_value(activations, prompts, -1, i, inc=0.25, k=k, method=method) for i in range(1, 6)}

    for C in optimal_C:
        df = {
            'L': list(range(len(activations))),
            'nN': [C for i in range(len(activations))],
            'N': [],
            'P': [],
            'R': [],
            'F1': []
        }
        for i in range(len(optimal_C[C])):
            df['N'].append(list_to_str(optimal_C[C][i][0]))
            df['P'].append(optimal_C[C][i][1])
            df['R'].append(optimal_C[C][i][2])
            df['F1'].append(optimal_C[C][i][3])

        scores_dfs.append(pd.DataFrame(df))

    return pd.concat(scores_dfs)


def get_multi_probing(activations, prompts, method, top_k_features=None, k=5):
    scores_df = {
        'L': [],
        'N': [],
        'P': [],
        'R': [],
        'F1': []
    }

    if top_k_features is None:
        top_k_features = get_top_neurons(activations, prompts, method=method, k=k)

    sel_neurons = {i: [] for i in range(len(activations))}
    for i in tqdm(range(len(activations) * k)):
        l, n, *_ = top_k_features.iloc[i]
        sel_neurons[int(l)].append(int(n))

    X_train, X_test, y_train, y_test = get_train_test(activations, prompts, int(l))

    for l in range(len(activations)):
        p, r = log_reg_score(X_train[:, sel_neurons[l]], y_train, X_test[:, sel_neurons[l]], y_test)

        scores_df['L'].append(int(l))
        scores_df['N'].append(sel_neurons[int(l)])
        scores_df['P'].append(p)
        scores_df['R'].append(r)
        scores_df['F1'].append(2 * p * r / (p + r + 1e-9))

    return pd.DataFrame(scores_df)
