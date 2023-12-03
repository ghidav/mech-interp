import pandas as pd
import numpy as np
import torch
import re


from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import standardize


def plotly_attribs(activ, title, k=50, n_heads=32, n_layers=32):  # tensor input
    deltas = activ.mean(dim=0)[:, 0].cpu().numpy().astype(np.float32) / 2  # n_layers * n_heads
    gammas = activ.std(dim=0)[:, 0].cpu().numpy().astype(np.float32) / 2  # n_layers * n_heads

    labels = [[f'L{j}H{i}' for i in range(n_heads)] + [f'L{j}MLP'] for j in range(n_layers)]
    colors = [['blue' for i in range(n_heads)] + ['red'] for j in range(n_layers)]
    labels = [j for i in labels for j in i]
    colors = [j for i in colors for j in i]

    idxs = np.argsort(deltas)[::-1]

    data = pd.DataFrame({
        'label': [labels[i] for i in idxs[:k]],
        'mean': deltas[idxs][:k],
        'std': gammas[idxs][:k],
        'color': [colors[i] for i in idxs[:k]]
    })

    fig = go.Figure(data=[go.Bar(
        x=[labels[i] for i in idxs[:k]],
        y=deltas[idxs][:k],
        marker_color=[colors[i] for i in idxs[:k]],  # marker color can be a single color value or an iterable
        error_y=dict(type='data', array=gammas[idxs][:k], visible=True)
    )])

    fig.update_layout(title_text=f'Logit attributions - {title}', yaxis=dict(title='Logit contribution', range=[0, 1]))

    return fig


# Function to plot PC of activations
def plot_pc(activations, prompts, rows, cols, n_comp=2):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(rows * cols)],
                        shared_yaxes=True)

    height = int(rows * 300)

    x_values = np.linspace(-3.5, 3.5, 100)  # np.linspace(-2, 1, 80)
    y_values = np.linspace(-2, 2, 100)  # np.linspace(-0.5, 0.5, 80)
    X, Y = np.meshgrid(x_values, y_values)
    positions = np.vstack([X.ravel(), Y.ravel()])

    alpha = 0.8

    pca = PCA(n_components=2)

    y = prompts.labels.to_numpy()
    m = int(np.sum(y))

    l = 0

    for i in range(rows):
        for j in range(cols):

            x = pca.fit_transform(activations[l])[:, :n_comp]
            x_std = standardize(x)

            kde_0 = gaussian_kde(x_std[:m, :].T)
            kde_1 = gaussian_kde(x_std[m:, :].T)

            if n_comp == 1:
                fig.add_trace(
                    go.Scatter(x=x_values, y=kde_0(x_values), mode='lines', name='Class 0', line=dict(color='red')),
                    row=i + 1, col=j + 1)
                fig.add_trace(
                    go.Scatter(x=x_values, y=kde_1(x_values), mode='lines', name='Class 1', line=dict(color='blues')),
                    row=i + 1, col=j + 1)
            else:
                Z_0 = np.reshape(kde_0(positions).T, X.shape)
                Z_1 = np.reshape(kde_1(positions).T, X.shape)

                fig.add_trace(
                    go.Heatmap(z=Z_0, x=x_values, y=y_values, colorscale='Reds', showscale=False, opacity=alpha),
                    row=i + 1,
                    col=j + 1
                )

                fig.add_trace(
                    go.Heatmap(z=Z_1, x=x_values, y=y_values, colorscale='Blues', showscale=False, opacity=alpha),
                    row=i + 1,
                    col=j + 1
                )

                fig.add_trace(
                    go.Contour(
                        z=Z_0, x=x_values, y=y_values,
                        contours=dict(coloring='heatmap', showlabels=True, start=0, end=1, size=0.1),
                        line=dict(width=2),
                        showscale=False, colorscale='Reds', opacity=0.3
                    ),
                    row=i + 1, col=j + 1
                )

                fig.add_trace(
                    go.Contour(
                        z=Z_1, x=x_values, y=y_values,
                        contours=dict(coloring='heatmap', showlabels=True, start=0, end=1, size=0.1),
                        line=dict(width=2),
                        showscale=False, colorscale='Blues', opacity=0.3
                    ),
                    row=i + 1, col=j + 1
                )

            l += 1

    fig.update_layout(height=height, width=1600, title_text="Logistic Regression on the 1st PC")

    return fig


def plot_mean_var(scores_df, rows, cols):
    n_layers = rows * cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(n_layers)],
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    height = int(rows * 300)

    scores_df['text'] = [
        f"N{int(scores_df.iloc[i, 1])} - F1: {np.round(scores_df['F1'].iloc[i] * 100, 2)}% - AUC: {np.round(scores_df['AUC'].iloc[i] * 100, decimals=2)}%"
        for i in range(len(scores_df))]

    for index, row in scores_df.iterrows():
        fig.add_trace(
            go.Scatter(x=[row['Mean']], y=[row['Variance']],
                       marker=dict(color=[row['F1']], colorscale='Viridis', cmin=0, cmax=1), text=[row['N']],
                       showlegend=False, hovertext=[row['text']]),
            # fill=[row['F1']],
            row=int(row['L'] // cols) + 1, col=int(row['L'] % cols) + 1
        )

    fig.update_layout(height=height, width=1600, title_text="Mean/Variance curves")
    return fig


def plot_mean_var_multi_probes(scores_df, rows, cols):
    n_layers = rows * cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(n_layers)],
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    height = int(rows * 300)

    scores_df['text'] = [
        f"N{int(scores_df.iloc[i, 1])} - F1: {np.round(scores_df['F1'].iloc[i] * 100, 2)}% - AUC: {np.round(scores_df['AUC'].iloc[i] * 100, decimals=2)}%"
        for i in range(len(scores_df))]

    mapping = {
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'orange',
        5: 'red'
    }

    def opacity(x):
        if 0 < x < 0.25: return 0.40
        if 0.25 < x < 0.50: return 0.60
        if 0.50 < x < 0.75: return 0.80
        if 0.75 < x < 1.0: return 1.0

    for index, row in scores_df.iterrows():
        fig.add_trace(
            go.Scatter(x=[row['Mean']], y=[row['Variance']],
                       marker=dict(color=[mapping[row['K']]]),
                       text=[row['N']],
                       showlegend=False, hovertext=[row['text']]),
            # fill=[row['F1']],
            row=int(row['L'] // cols) + 1, col=int(row['L'] % cols) + 1
        )

    fig.update_layout(height=height, width=1600, title_text="Mean/Variance curves")
    return fig


def plot_roc_curves(scores_df, rows, cols):
    n_layers = rows * cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(n_layers)],
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    height = int(rows * 300)

    scores_df['text'] = [
        f"N{int(scores_df.iloc[i, 1])} - F1: {np.round(scores_df['F1'].iloc[i] * 100, 2)}% - AUC: {np.round(scores_df['AUC'].iloc[i] * 100, decimals=2)}%"
        for i in range(len(scores_df))]

    for index, row in scores_df.iterrows():
        x = row["fpr"]
        y =row["tpr"]
        if type(x) == str:
            x = re.sub(r'\s+', " ", row["fpr"]).replace("[","").replace("]","").split(" ")
            x = [float(el) for el in x if el != ""]
        if type(y) == str:
            y = re.sub(r'\s+', " ", row["tpr"]).replace("[", "").replace("]", "").split(" ")
            y = [float(el) for el in y if el != ""]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines',
                       line=dict(shape = 'spline', smoothing =1),
                       text=[row['N']],
                       showlegend=False, hovertext=[row['text']]),
            # fill=[row['F1']],
            row=int(row['L'] // cols) + 1, col=int(row['L'] % cols) + 1
        )

    fig.update_layout(height=height, width=1600, title_text="ROC curves")
    return fig


def plot_roc_curves_multi_probes(scores_df, rows, cols):
    n_layers = rows * cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(n_layers)],
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    height = int(rows * 300)

    scores_df['text'] = [
        f"N{int(scores_df.iloc[i, 1])} - F1: {np.round(scores_df['F1'].iloc[i] * 100, 2)}% - AUC: {np.round(scores_df['AUC'].iloc[i] * 100, decimals=2)}%"
        for i in range(len(scores_df))]

    mapping = {
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'orange',
        5: 'red'
    }

    def opacity(x):
        if 0 < x < 0.25: return 0.40
        if 0.25 < x < 0.50: return 0.60
        if 0.50 < x < 0.75: return 0.80
        if 0.75 < x < 1.0: return 1.0

    for index, row in scores_df.iterrows():
        x = row["fpr"]
        y =row["tpr"]
        if type(x) == str:
            x = re.sub(r'\s+', " ", row["fpr"]).replace("[","").replace("]","").split(" ")
            x = [float(el) for el in x if el != ""]
        if type(y) == str:
            y = re.sub(r'\s+', " ", row["tpr"]).replace("[", "").replace("]", "").split(" ")
            y = [float(el) for el in y if el != ""]
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='lines',
                       line=dict(shape = 'spline', smoothing = 1, color=mapping[row['K']]),
                       text=[row['N']],
                       showlegend=False, hovertext=[row['text']]),
            # fill=[row['F1']],
            row=int(row['L'] // cols) + 1, col=int(row['L'] % cols) + 1
        )

    fig.update_layout(height=height, width=1600, title_text="ROC curves")
    return fig


def plot_single_probes(scores_df, rows, cols):
    n_layers = rows * cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(n_layers)],
                        horizontal_spacing=0.05, vertical_spacing=0.05)

    height = int(rows * 300)

    precision = np.linspace(0 + 1e-9, 1, 100)
    recall = np.linspace(0 + 1e-9, 1, 100)
    P, R = np.meshgrid(precision, recall)
    F1 = 2 * (P * R) / (P + R)
    F1[np.isnan(F1)] = 0  # Handle division by zero

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.add_trace(
                go.Contour(
                    z=F1, x=precision, y=recall,
                    contours=dict(coloring='heatmap', showlabels=True, start=0, end=1, size=0.1),
                    line=dict(width=2),
                    showscale=False, colorscale='Viridis', opacity=0.5
                ),
                row=i, col=j
            )

    scores_df['text'] = [
        f"N{int(scores_df.iloc[i, 1])} - F1: {np.round(scores_df['F1'].iloc[i] * 100, 2)}% - AUC: {np.round(scores_df['AUC'].iloc[i] * 100, decimals=2)}%"
        for i in range(len(scores_df))]

    for index, row in scores_df.iterrows():
        fig.add_trace(
            go.Scatter(x=[row['P']], y=[row['R']],
                       marker=dict(color=[row['F1']], colorscale='Viridis', cmin=0, cmax=1), text=[row['N']],
                       showlegend=False, hovertext=[row['text']]),
            # fill=[row['F1']],
            row=int(row['L'] // cols) + 1, col=int(row['L'] % cols) + 1
        )

    fig.update_layout(height=height, width=1600, title_text="Precision/Recall curves")
    return fig


def plot_multi_probes(scores_dfs, rows, cols):
    n_layers = rows * cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(n_layers)],
                        horizontal_spacing=0.05, vertical_spacing=0.05, shared_yaxes=True)

    height = int(rows * 300)

    precision = np.linspace(0 + 1e-9, 1, 100)
    recall = np.linspace(0 + 1e-9, 1, 100)
    P, R = np.meshgrid(precision, recall)
    F1 = 2 * (P * R) / (P + R)
    F1[np.isnan(F1)] = 0  # Handle division by zero

    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            fig.add_trace(
                go.Contour(
                    z=F1, x=precision, y=recall,
                    contours=dict(coloring='heatmap', showlabels=True, start=0, end=1, size=0.1),
                    line=dict(width=2),
                    showscale=False, colorscale='Viridis', opacity=0.5
                ),
                row=i, col=j
            )

    scores_dfs['text'] = [f"N{scores_dfs.iloc[i, 2]} - F1: {np.round(scores_dfs.iloc[i, 5] * 100, 2)}%" for i in
                          range(len(scores_dfs))]
    mapping = {
        1: 'blue',
        2: 'green',
        3: 'yellow',
        4: 'orange',
        5: 'red'
    }

    for index, row in scores_dfs.iterrows():
        fig.add_trace(
            go.Scatter(x=[row['P']], y=[row['R']], marker=dict(color=[mapping[row['K']]]),
                       text=[row['text']], showlegend=False, hovertext=[row['text']]),  # fill=[row['F1']],
            row=int(row['L'] // cols) + 1, col=int(row['L'] % cols) + 1
        )

    fig.update_layout(height=height, width=1600, title_text="Precision/Recall curves")
    return fig
