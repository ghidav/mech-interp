import plotly.graph_objects as go
import pandas as pd
import numpy as np
import torch

def plotly_attribs(activ, title, k=50, n_heads=32, n_layers=32): # tensor input
    deltas = activ.mean(dim=0)[:, 0].cpu().numpy().astype(np.float32) / 2 # n_layers * n_heads
    gammas = activ.std(dim=0)[:, 0].cpu().numpy().astype(np.float32) / 2 # n_layers * n_heads

    labels = [[f'L{j}H{i}' for i in range(n_heads)] + [f'L{j}MLP']  for j in range(n_layers)]
    colors = [['blue' for i in range(n_heads)] + ['red']  for j in range(n_layers)]
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
        marker_color=[colors[i] for i in idxs[:k]], # marker color can be a single color value or an iterable
        error_y=dict(type='data', array=gammas[idxs][:k], visible=True)
    )])
    
    fig.update_layout(title_text=f'Logit attributions - {title}', yaxis=dict(title='Logit contribution', range=[0, 1]))

    # Creating the bar plot
    fig.show()