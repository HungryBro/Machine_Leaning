"""Performance estimation: resubstitution, holdout and k-fold CV."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'plots')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', 'Week3')))
from bias_variance_lab_compact import (SEED, EOUT_GRID, sin_target, fit_predict,  # pyright: ignore
                                       eout_parts, expected_eout, reference_data)

MODELS, METHODS = ['Constant', 'Linear', 'Linear through origin'], ['Resub', 'Holdout', 'KFold']


def gen_data(n, sigma):
    X = np.random.uniform(-1, 1, n)
    return X, sin_target(X) + np.random.normal(0, sigma, n)


def true_eout(model, X, y, sigma):
    return expected_eout(model, X, y, sin_target, EOUT_GRID, sigma)


def resub(model, X, y):
    return float(np.mean((fit_predict(model, X, y, X) - y) ** 2))


def holdout(model, X, y, frac=0.7):
    idx = np.random.permutation(len(X))
    cut = min(max(round(frac * len(X)), 1), len(X) - 1)
    tr, te = idx[:cut], idx[cut:]
    return float(np.mean((fit_predict(model, X[tr], y[tr], X[te]) - y[te]) ** 2))


def kfold(model, X, y, k=5):
    folds = np.array_split(np.random.permutation(len(X)), max(2, min(k, len(X))))
    err = []
    for i, te in enumerate(folds):
        tr = np.concatenate(folds[:i] + folds[i + 1:])
        err.extend((fit_predict(model, X[tr], y[tr], X[te]) - y[te]) ** 2)
    return float(np.mean(err))


def run(model, n, sigma, frac=0.7, k=5, reps=2000):
    out = {name: [] for name in ['true'] + METHODS}
    for _ in range(reps):
        X, y = gen_data(n, sigma)
        out['true'].append(true_eout(model, X, y, sigma))
        out['Resub'].append(resub(model, X, y))
        out['Holdout'].append(holdout(model, X, y, frac))
        out['KFold'].append(kfold(model, X, y, k))
    return {key: np.asarray(value) for key, value in out.items()}


def bias_var(data):
    out = {}
    for method in METHODS:
        error = data[method] - data['true']
        out[method] = dict(bias=float(error.mean()), var=float(error.var()),
                           mse=float(np.mean(error ** 2)))
    return out


def sweep(model, kind, values, n=20, sigma=0.3, reps=1000):
    data = [gen_data(n, sigma) for _ in range(reps)]
    truth = np.array([true_eout(model, X, y, sigma) for X, y in data])
    rows = []
    for value in values:
        est = np.array([holdout(model, X, y, value) if kind == 'frac'
                        else kfold(model, X, y, value) for X, y in data])
        error = est - truth
        rows.append([error.mean(), error.var(), np.mean(error ** 2),
                     est.mean(), truth.mean()])
    return np.asarray(rows)


def show_series(label, values, datasets):
    stats = [bias_var(data) for data in datasets]
    for value, data, result in zip(values, datasets, stats):
        print(f'\n  {label}={value} | mean true Eout={data["true"].mean():.4f}')
        print(f"    {'Estimator':<10} {'Bias':>10} {'Variance':>10} {'MSE':>10}")
        for method in METHODS:
            s = result[method]
            print(f"    {method:<10} {s['bias']:10.4f} {s['var']:10.4f} {s['mse']:10.4f}")
    return stats


def main():
    np.random.seed(SEED)
    n, sigma = 20, 0.3
    print('\n' + '=' * 100)
    print('WEEK 4: PERFORMANCE ESTIMATION')
    print('Metrics: Ein = Resubstitution MSE; Eout = signal error + sigma^2')
    print(f'Setup: target=sin(pi*x), x~U(-1,1), n={n}, sigma={sigma}, noise variance={sigma ** 2:.4f}')
    print('=' * 100)

    print('\n' + '-' * 100)
    print('1) REFERENCE DATASET')
    print('-' * 100)
    X, y = reference_data(n, sigma, SEED)
    print(f"  {'Model':<22} {'Ein':>10} {'Signal':>10} {'Noise^2':>10} {'Eout':>10}")
    print('  ' + '-' * 72)
    for model in MODELS:
        signal, noise, eout = eout_parts(model, X, y, sigma=sigma)
        print(f'  {model:<22} {resub(model, X, y):10.6f} {signal:10.6f} {noise:10.6f} '
              f'{eout:10.6f}')

    print('\n  ESTIMATORS')
    print(f"  {'Model':<22} {'Holdout':>10} {'KFold':>10}")
    print('  ' + '-' * 46)
    for model in MODELS:
        print(f'  {model:<22} {holdout(model, X, y):10.6f} {kfold(model, X, y):10.6f}')

    print('\n' + '-' * 100)
    print('2) ESTIMATOR ERROR OVER 2000 DATASETS')
    print(f'  Setup: n={n}, sigma={sigma}, noise variance={sigma ** 2:.4f}')
    print('-' * 100)
    datasets = {model: run(model, n, sigma) for model in MODELS}
    for model, data in datasets.items():
        print(f'\n  Model: {model} | mean true Eout={data["true"].mean():.4f}')
        print(f"  {'Estimator':<10} {'Bias':>10} {'Variance':>10} {'MSE':>10}")
        print('  ' + '-' * 48)
        for method, s in bias_var(data).items():
            print(f"  {method:<10} {s['bias']:10.4f} {s['var']:10.4f} {s['mse']:10.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, model in zip(axes, MODELS):
        err = np.column_stack([datasets[model][method] - datasets[model]['true'] for method in METHODS])
        ax.boxplot(err, tick_labels=METHODS, showmeans=True); ax.axhline(0, color='k', lw=0.8)
        ax.set(title=model, xlabel='Estimator', ylabel='estimate − true Eout'); ax.grid(alpha=0.3)
        ax.legend(handles=[
            Patch(facecolor='white', edgecolor='black', label='Box = middle 50% (IQR)'),
            Line2D([0], [0], color='#E67E22', lw=2, label='Orange = median'),
            Line2D([0], [0], marker='^', color='none', markerfacecolor='#2CA02C',
                   markeredgecolor='#2CA02C', markersize=7, label='Green triangle = mean'),
            Line2D([0], [0], marker='o', color='none', markerfacecolor='none',
                   markeredgecolor='black', markersize=5, label='Circle = outlier')],
            loc='upper right', fontsize=7, framealpha=0.92)
    plt.tight_layout(); plt.savefig(f'{OUT}/part2.png', dpi=150); plt.close(fig)

    print('\n' + '-' * 100)
    print('3) HOLDOUT FRACTION AND K-FOLD')
    print(f'  Setup: n={n}, sigma={sigma}, noise variance={sigma ** 2:.4f}')
    print('-' * 100)
    fracs, ks = [0.1, 0.3, 0.5, 0.7, 0.9], [2, 5, 10, 20]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    for row, model in enumerate(MODELS):
        hold, fold = sweep(model, 'frac', fracs), sweep(model, 'k', ks)
        hb, hv, _, _, _ = hold.T; kb, kv, _, _, _ = fold.T
        print(f'\n  Model: {model}')
        print(f"  {'Setting':<12} {'True Eout':>10} {'Estimate':>10} {'Bias':>10} {'Variance':>10} {'MSE':>10}")
        print('  ' + '-' * 78)
        for name, values, table in [('frac', fracs, hold), ('k', ks, fold)]:
            for value, (bias, var, mse, estimate, truth) in zip(values, table):
                print(f'  {name}={value:<7} {truth:10.4f} {estimate:10.4f} {bias:10.4f} {var:10.4f} {mse:10.4f}')
        for ax, x, yv, title, logy in [
            (axes[row, 0], fracs, hb, f'{model}: Holdout bias vs frac', False),
            (axes[row, 1], fracs, hv, f'{model}: Holdout var vs frac', True),
            (axes[row, 2], ks, kb, f'{model}: K-fold bias vs k', False),
            (axes[row, 3], ks, kv, f'{model}: K-fold var vs k', False)]:
            is_bias = 'bias' in title
            ax.plot(x, yv, 'o-', label='Bias (estimate − true Eout)' if is_bias else 'Variance of estimation error')
            ax.set(title=title, xlabel='Holdout train fraction' if 'Holdout' in title else 'k (number of folds)',
                   ylabel='Bias (estimate − true)' if is_bias else 'Variance')
            ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=7, framealpha=0.92)
            if logy: ax.set_yscale('log')
            else: ax.axhline(0, color='gray', lw=0.6)
    plt.tight_layout(); plt.savefig(f'{OUT}/part3.png', dpi=150); plt.close(fig)

    print('\n' + '-' * 100)
    print('4) EFFECT OF n AND sigma (all values show bias, variance and MSE)')
    print('-' * 100)
    n_list, sigma_list = [5, 10, 20, 50, 100], [0.0, 0.2, 0.4, 0.6, 0.8]
    fig, axes = plt.subplots(3, 4, figsize=(18, 10))
    for row, model in enumerate(MODELS):
        print(f'\n{model}: vary n, fixed sigma=0.3')
        bv_n = show_series('n', n_list, [run(model, value, 0.3, reps=800) for value in n_list])
        print(f'{model}: vary sigma, fixed n=20')
        bv_s = show_series('sigma', sigma_list, [run(model, 20, value, reps=800) for value in sigma_list])
        for ax, x, key, title in [
            (axes[row, 0], n_list, ('bias', bv_n), f'{model}: Bias vs n'),
            (axes[row, 1], n_list, ('var', bv_n), f'{model}: Variance vs n'),
            (axes[row, 2], sigma_list, ('bias', bv_s), f'{model}: Bias vs sigma'),
            (axes[row, 3], sigma_list, ('var', bv_s), f'{model}: Variance vs sigma')]:
            metric, results = key
            for method in METHODS:
                ax.plot(x, [result[method][metric] for result in results], 'o-', label=method)
            ax.set(title=title, xlabel='Training set size n' if x == n_list else 'Noise level σ',
                   ylabel='Bias' if metric == 'bias' else 'Variance')
            ax.legend(fontsize=7); ax.grid(alpha=0.3)
            if metric == 'bias': ax.axhline(0, color='gray', lw=0.6)
    plt.tight_layout(); plt.savefig(f'{OUT}/part4.png', dpi=150); plt.close(fig)
    print('\nSaved: plots/part2.png, plots/part3.png, plots/part4.png')


if __name__ == '__main__':
    main()
