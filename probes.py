import numpy as np
import pandas as pd
import os
from tqdm.auto import tqdm

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from joblib import Parallel, delayed

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

        for j in tqdm(range(len(activations))):
            X = activations[j].numpy()
            y = prompts.labels.to_numpy()

            imps = pd.DataFrame({
                'layer': [j for i in range(n)],
                'neuron': np.arange(n),
                'importance': np.abs(X[y > 0.5].mean(axis=0) -
                                     X[y < 0.5].mean(axis=0))
            })

            top_k_features.append(imps.sort_values(by='importance').iloc[-k:])

    if method == 'lr_l1':
        model = LogisticRegression(n_jobs=-1, verbose=False, max_iter=250, class_weight='balanced',
                                   C=0.1, penalty='l1', solver='saga', **kwargs)

    if method == 'lr_l2':
        model = LogisticRegression(n_jobs=-1, verbose=False, max_iter=250, class_weight='balanced', solver='saga',
                                   **kwargs)

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
def log_reg_score(train_ds, return_roc=False, **kwargs):
    X_train, X_test, y_train, y_test = train_ds
    lr = LogisticRegression(class_weight='balanced',
                            penalty='l2',
                            solver='saga',
                            n_jobs=-1,
                            max_iter=200,
                            **kwargs)
    lr.fit(X_train, y_train)
    y_pred_prob = lr.predict(X_test)
    mean = np.mean(y_pred_prob)
    variance = np.var(y_pred_prob)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    y_pred = np.round(y_pred_prob)

    ans = {'P':
               precision_score(y_test, y_pred, zero_division=0), 'R': recall_score(y_test, y_pred, zero_division=0),
           'AUC': auc(fpr, tpr),
           'Mean': mean, 'Variance': variance}

    if return_roc:
        ans['fpr'] = fpr
        ans['tpr'] = tpr

    return ans


@ignore_warnings(category=ConvergenceWarning)
def log_reg_score_with_prompts(train_ds, return_roc=False, **kwargs):
    X_train, X_test, y_train, y_test, prompts_test = train_ds
    lr = LogisticRegression(class_weight='balanced',
                            penalty='l2',
                            solver='saga',
                            n_jobs=-1,
                            max_iter=200,
                            **kwargs)
    lr.fit(X_train, y_train)

    y_pred_prob = lr.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    mean = np.mean(y_pred_prob)
    variance = np.var(y_pred_prob)
    y_pred = np.round(y_pred_prob)

    positively_predicted = [prompt for i, prompt in enumerate(prompts_test) if
                            y_test[i] == 1 and y_pred[i] == 1]

    ans = {'P':
               precision_score(y_test, y_pred, zero_division=0), 'R': recall_score(y_test, y_pred, zero_division=0),
           'AUC': auc(fpr, tpr),
           'Mean': mean, 'Variance': variance, 'Phrases': positively_predicted}
    if return_roc:
        ans['fpr'] = fpr
        ans['tpr'] = tpr
    return ans

def get_single_probing(activations, prompts, method, top_k_features=None, k=64, print_activations=True, add_roc = False):
    scores_df = {
        'L': [],
        'N': [],
        'P': [],
        'R': [],
        'F1': [],
        'AUC': [],
        'Mean': [],
        'Variance': []
    }
    if add_roc:
        scores_df['fpr'] = []
        scores_df['tpr'] = []

    if print_activations:
        phrases_df = {
            'L': [],
            'N': [],
            'Phrases': [],
        }

    if top_k_features is None:
        top_k_features = get_top_neurons(activations, prompts, method=method, k=k)

    ########
    bs = 64
    ########

    for i in tqdm(range(len(activations) * k // bs), desc="Single probing"):

        train_ds = []

        for j in range(i*bs, (i+1)*bs):
            l, n, *_ = top_k_features.iloc[j]

            scores_df['L'].append(l)
            scores_df['N'].append(n)
        
            if print_activations:
                X_train, X_test, y_train, y_test, prompts_test = get_train_test_with_prompts(activations, prompts, int(l))
                train_ds.append((X_train[:, int(n), None], X_test[:, int(n), None], y_train, y_test, prompts_test))
                
                phrases_df['L'].append(l)
                phrases_df['N'].append(n)

            else:
                X_train, X_test, y_train, y_test = get_train_test(activations, prompts, int(l))
                train_ds.append((X_train[:, int(n), None], X_test[:, int(n), None], y_train, y_test))

        
        if print_activations:
            res_dicts = Parallel(n_jobs=bs)(delayed(log_reg_score_with_prompts)(train_ds[i], return_roc=add_roc) for i in range(bs))
        else:
            res_dicts = Parallel(n_jobs=bs)(delayed(log_reg_score)(train_ds[i], return_roc=add_roc) for i in range(bs))
        
        for j in range(bs):
            p = res_dicts[j]['P']
            r = res_dicts[j]['R']
            scores_df['P'].append(p)
            scores_df['R'].append(r)
            scores_df['F1'].append(2 * p * r / (p + r + 1e-9))
            scores_df['AUC'].append(res_dicts[j]['AUC'])
            scores_df['Mean'].append(res_dicts[j]['Mean'])
            scores_df['Variance'].append(res_dicts[j]['Variance'])
            
            if print_activations:
                phrases_df['Phrases'].append(res_dicts[j]['Phrases'])

            if add_roc:
                scores_df['fpr'].append(res_dicts[j]['fpr'])
                scores_df['tpr'].append(res_dicts[j]['tpr'])

    phrases_res = None if not print_activations else pd.DataFrame(phrases_df)
    return pd.DataFrame(scores_df), phrases_res


def get_multi_probing(activations, prompts, method, top_k_features=None, max_multi_k=5, batch_k=15, add_roc = False):
    scores_df = {
        'L': [],
        'K': [],
        'N': [],
        'P': [],
        'R': [],
        'F1': [],
        'AUC': [],
        'Mean': [],
        'Variance': []
    }
    if add_roc:
        scores_df['fpr'] = []
        scores_df['tpr'] = []

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
                    res_dict = log_reg_score(X_train[:, neurons_to_consider], y_train,
                                                                    X_test[:, neurons_to_consider], y_test, add_roc)
                    p=res_dict['P']
                    r = res_dict['R']
                    scores_df['L'].append(int(l))
                    scores_df['K'].append(int(k))
                    scores_df['N'].append(neurons_to_consider)
                    scores_df['P'].append(p)
                    scores_df['R'].append(r)
                    scores_df['F1'].append(2 * p * r / (p + r + 1e-9))
                    scores_df['AUC'].append(res_dict['AUC'])
                    scores_df['Mean'].append(res_dict['Mean'])
                    scores_df['Variance'].append(res_dict['Variance'])
                    if add_roc:
                        scores_df['fpr'].append(res_dict['fpr'])
                        scores_df['tpr'].append(res_dict['tpr'])

    return pd.DataFrame(scores_df)
