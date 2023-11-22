import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

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
    model = None
    if method == 'mmd':
        raise NotImplementedError

        for j in tqdm(range(len(activations))):
            X = activations[j].numpy()
            y = prompts.labels.to_numpy()

            imps = pd.DataFrame({
                'layer': [j for i in range(n)],
                'neuron': np.arange(n),
                'importance': np.abs()
            })
            """
            mean_dif = np.abs(X[pos_class].mean(axis=0) -
                              X[~pos_class].mean(axis=0))
            """

            top_k_features.append(imps.sort_values(by='importance').iloc[-k:])

    if method == 'lr_l1':
        model = LogisticRegression(n_jobs=-1, verbose=False, max_iter=250, class_weight='balanced',
                                   C=0.1, penalty='l1', solver='saga', **kwargs)

    if method == 'lr_l2':
        model = LogisticRegression(n_jobs=-1, verbose=False, max_iter=250, class_weight='balanced', solver='saga',**kwargs)

    elif method == 'svc':
        model = LinearSVC(loss='hinge', dual='auto', verbose=False, **kwargs)

    if model:
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
    lr = LogisticRegression(class_weight='balanced',
                            penalty='l2',
                            solver='saga',
                            n_jobs=-1,
                            max_iter=200,
                            **kwargs)
    lr.fit(X_train, y_train)

    y_pred_prob = lr.predict(X_test)
    y_pred = np.round(y_pred_prob)

    return (precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0))

@ignore_warnings(category=ConvergenceWarning)
def log_reg_score_with_prompts(X_train, y_train, X_test, y_test, prompts_test, **kwargs):
    lr = LogisticRegression(class_weight='balanced',
                            penalty='l2',
                            solver='saga',
                            n_jobs=-1,
                            max_iter=200,
                            **kwargs)
    lr.fit(X_train, y_train)

    y_pred_prob = lr.predict(X_test)
    y_pred = np.round(y_pred_prob)

    positively_predicted = [prompt for i, prompt in enumerate(prompts_test) if
                            y_test[i] == 1 and y_pred[i] == 1]

    return (precision_score(y_test, y_pred, zero_division=0), recall_score(y_test, y_pred, zero_division=0),
            positively_predicted)


def get_single_probing(activations, prompts, method, top_k_features=None, k=64, print_activations = True):
    scores_df = {
        'L': [],
        'N': [],
        'P': [],
        'R': [],
        'F1': []
    }
    if print_activations:
        phrases_df = {
            'L': [],
            'N': [],
            'Phrases': [],
        }

    if top_k_features is None:
        top_k_features = get_top_neurons(activations, prompts, method=method, k=k)

    for i in tqdm(range(len(activations) * k), desc="Single probing"):
        l, n, *_ = top_k_features.iloc[i]

        if print_activations:
            X_train, X_test, y_train, y_test, prompts_test = get_train_test_with_prompts(activations, prompts, int(l))
            p, r, predicted = log_reg_score_with_prompts(X_train[:, int(n), None], y_train, X_test[:, int(n), None],
                                                         y_test,
                                                         prompts_test)
            phrases_df['L'].append(l)
            phrases_df['N'].append(n)
            phrases_df['Phrases'].append(predicted)

        else:
            X_train, X_test, y_train, y_test = get_train_test(activations, prompts, int(l))
            p, r = log_reg_score(X_train[:, int(n), None], y_train, X_test[:, int(n), None], y_test)

        scores_df['L'].append(l)
        scores_df['N'].append(n)
        scores_df['P'].append(p)
        scores_df['R'].append(r)
        scores_df['F1'].append(2 * p * r / (p + r + 1e-9))

    phrases_res = None if not print_activations else pd.DataFrame(phrases_df)
    return pd.DataFrame(scores_df), phrases_res


def get_multi_probing(activations, prompts, method, top_k_features=None, max_multi_k=5, batch_k=15):
    scores_df = {
        'L': [],
        'K': [],
        'N': [],
        'P': [],
        'R': [],
        'F1': []
    }

    top_neurons_to_source = batch_k + max_multi_k - 1

    if top_k_features is None:
        top_k_features = get_top_neurons(activations, prompts, method=method, k=top_neurons_to_source)

    # creating dict for top_k neurons, dictionary with layer: list of selected neurons
    sel_neurons = {i: [] for i in range(len(activations))}
    for i in range(len(activations) * top_neurons_to_source):  # layer * k
        l, n, *_ = top_k_features.iloc[i]
        sel_neurons[int(l)].append(int(n))  # creating a list of layer with selected indices

    for l in tqdm(range(len(activations)), desc="Multi probing"):
        X_train, X_test, y_train, y_test = get_train_test(activations, prompts, int(l))
        for k in range(1, max_multi_k + 1):
            for i in range(batch_k):
                neurons_to_consider = sel_neurons[int(l)][i:i + k]
                if len(neurons_to_consider) > 0:
                    p, r = log_reg_score(X_train[:, neurons_to_consider], y_train, X_test[:, neurons_to_consider], y_test)

                    scores_df['L'].append(int(l))
                    scores_df['K'].append(int(k))
                    scores_df['N'].append(neurons_to_consider)
                    scores_df['P'].append(p)
                    scores_df['R'].append(r)
                    scores_df['F1'].append(2 * p * r / (p + r + 1e-9))

    return pd.DataFrame(scores_df)
