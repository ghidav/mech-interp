import argparse

import pandas as pd
from plotting import plot_single_probes, plot_mean_var, plot_roc_curves, plot_multi_probes, plot_mean_var_multi_probes, \
    plot_roc_curves_multi_probes

parser = argparse.ArgumentParser()
parser.add_argument("-single_probes_dir", "--single_probes_dir", help="directory from project root where the csv where the single probes are stored", default='probes/pythia-6.9b/single_probes.csv')
parser.add_argument("-multi_probes_dir", "--multi_probes_dir", help="directory from project root where the csv where the multi probes are stored", default='probes/pythia-6.9b/multi_probes.csv')
parser.add_argument("-plot_probes", "--plot_probes", help="Whether to plot single probes", default=True)
parser.add_argument("-plot_mean_and_var", "--plot_mean_and_var", help="Whether to plot single probes", default=True)
parser.add_argument("-plot_ROC_curve", "--plot_ROC_curve", help="Whether to plot single probes", default=True)
parser.add_argument("-folder", "--folder", help="Folder where to save results", default="debug")
args = parser.parse_args()

#Single_probes
if args.single_probes_dir is not None:
    single_probes = pd.read_csv(args.single_probes_dir)
    n_layers = int(single_probes['L'].max())
    if n_layers % 6 == 0:
        cols = 6
    else:
        cols = 4
    rows = (n_layers // cols) + 1
    print('Creating single probes plot...')
    fig = plot_single_probes(single_probes, rows, cols)
    fig.write_image(f"images/{args.folder}/single_probes.svg")
    fig.write_html(f"images/{args.folder}/single_probes.html")
    print('Plot created and saved.')

    print('Creating single probes mean and variance plot for single probes...')
    fig = plot_mean_var(single_probes, rows, cols)
    fig.write_image(f"images/{args.folder}/mean_var_single_probes.svg")
    fig.write_html(f"images/{args.folder}/mean_var_single_probes.html")
    print('Plot created and saved.')

    print('Creating single probes ROC curves plot for single probes...')
    fig = plot_roc_curves(single_probes, rows, cols)
    fig.write_image(f"images/{args.folder}/roc_single_probes.svg")
    fig.write_html(f"images/{args.folder}/roc_single_probes.html")
    print('Plot created and saved.')

#Multi_probes
if args.multi_probes_dir is not None:
    multi_probes = pd.read_csv(args.multi_probes_dir)
    n_layers = int(multi_probes['L'].max())
    if n_layers % 6 == 0:
        cols = 6
    else:
        cols = 4
    rows = (n_layers // cols) + 1

    print('Creating multi probes plot...')
    fig = plot_multi_probes(multi_probes, rows, cols)
    fig.write_image(f"images/{args.folder}/multi_probes.svg")
    fig.write_html(f"images/{args.folder}/multi_probes.html")
    print('Plot created and saved.')

    print('Creating single probes mean and variance plot for multi probes...')
    fig = plot_mean_var_multi_probes(multi_probes, rows, cols)
    fig.write_image(f"images/{args.folder}/mean_var_multi_probes.svg")
    fig.write_html(f"images/{args.folder}/mean_var_multi_probes.html")
    print('Plot created and saved.')

    print('Creating single probes ROC curves plot for multi probes...')
    fig = plot_roc_curves_multi_probes(multi_probes, rows, cols)
    fig.write_image(f"images/{args.folder}/roc_multi_probes.svg")
    fig.write_html(f"images/{args.folder}/roc_multi_probes.html")
    print('Plot created and saved.')

