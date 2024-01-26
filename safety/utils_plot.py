import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd

from utils import FastPCA

def standardize(x):
    return (x - x.mean(0)) / x.std(0)

def plot_pc(activations, y, rows, cols, n_comp=2, center=True):
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f'L {i + 1}' for i in range(rows * cols)],
                        shared_yaxes=True)

    height = int(rows * 300)

    x_values = np.linspace(-3.5, 3.5, 100)  # np.linspace(-2, 1, 80)
    y_values = np.linspace(-2, 2, 100)  # np.linspace(-0.5, 0.5, 80)
    X, Y = np.meshgrid(x_values, y_values)
    positions = np.vstack([X.ravel(), Y.ravel()])

    alpha = 0.8

    pca = FastPCA(n_components=n_comp)

    m = int(np.sum(y))

    l = 0

    for i in range(rows):
        for j in range(cols):

            x = pca.fit_transform(activations[:, l], center=center)[:, :n_comp]
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

def plot_classification(safe_rocs, unsafe_rocs, components):

    nl = len(safe_rocs)

    # Safe activations
    df_safe = {f'{l}': safe_rocs[l] for l in range(nl)}
    df_safe['N Components'] = components
    df_safe = pd.DataFrame(df_safe).melt('N Components', [str(x) for x in range(32)], 'Layer', 'AUC')

    fig1 = px.line(df_safe, x='N Components', y='AUC', color='Layer', height=600, title='Safe activations')

    # Unsafe activations
    df_unsafe = {f'{l}': unsafe_rocs[l] for l in range(nl)}
    df_unsafe['N Components'] = components
    df_unsafe = pd.DataFrame(df_unsafe).melt('N Components', [str(x) for x in range(32)], 'Layer', 'AUC')

    fig2 = px.line(df_unsafe, x='N Components', y='AUC', color='Layer', height=600, title='Unsafe activations')

    # Create subplots with two rows and one column
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Add the plots
    for l in range(nl):
        fig.add_trace(go.Scatter(fig1.data[l]), row=1, col=1)
        fig.add_trace(go.Scatter(fig2.data[l]), row=2, col=1)

    # Update layout settings
    fig.update_layout(title_text='Classification AUC scores at different PC', height=600)
    fig.update_yaxes(title_text="Safe AUC", row=1, col=1)
    fig.update_xaxes(title_text="N Components", row=2, col=1)
    fig.update_yaxes(title_text="Unsafe AUC", row=2, col=1)

    fig.show()

