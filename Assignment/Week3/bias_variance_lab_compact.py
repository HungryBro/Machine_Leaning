import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def sin_target(x):
    return np.sin(np.pi * np.asarray(x))


def square_target(x):
    return np.asarray(x) ** 2


TARGETS = {'sin(pi*x)': sin_target, 'x^2': square_target}
MODELS = ['Constant', 'Linear', 'Linear through origin']
NOISE = [0.0, 0.1, 0.3]
COLORS = {0.0: '#0072B2', 0.1: '#009E73', 0.3: '#D55E00'}
N_LIST = [2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100]
YMAX = {'x^2': 0.6, 'sin(pi*x)': 1.0}
EOUT_GRID = np.linspace(-1, 1, 4000)


def fit_predict(model, X, y, xq):
    X, y, xq = map(np.asarray, (X, y, xq))
    if model == 'Constant':
        return np.full(xq.shape, y.mean(), dtype=float)
    if model == 'Linear':
        w = np.linalg.lstsq(np.c_[np.ones(len(X)), X], y, rcond=None)[0]
        return w[0] + w[1] * xq
    w = np.linalg.lstsq(X[:, None], y, rcond=None)[0][0]
    return w * xq


def eout_parts(model, X, y, f=sin_target, x_test=EOUT_GRID, sigma=0.0):
    signal = float(np.mean((fit_predict(model, X, y, x_test) - f(x_test)) ** 2))
    noise = float(sigma ** 2)
    return signal, noise, signal + noise


def expected_eout(model, X, y, f=sin_target, x_test=EOUT_GRID, sigma=0.0):
    return eout_parts(model, X, y, f, x_test, sigma)[2]


def reference_data(n=20, sigma=0.3, seed=SEED):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, n)
    return X, sin_target(X) + rng.normal(0, sigma, n)


def simulate(f, model, n_datasets=50000, n_test=300):
    x = np.linspace(-1, 1, n_test)
    preds = np.empty((n_datasets, n_test))
    for i in range(n_datasets):
        X = np.random.uniform(-1, 1, 2)
        preds[i] = fit_predict(model, X, f(X), x)
    mean, std = preds.mean(0), preds.std(0)
    bias2 = np.mean((mean - f(x)) ** 2)
    variance = np.mean(np.var(preds, axis=0))
    return dict(bias2=float(bias2), variance=float(variance),
                eout=float(bias2 + variance), g_bar=mean, std=std, x_test=x)


def analytical_bias_variance(f, model, q=401, n_test=1000):
    """Numerically integrate over x and the two training inputs (n=2)."""
    x = np.linspace(-1, 1, n_test)
    fx = f(x)
    average_x = lambda values: float(np.trapezoid(values, x) / 2)

    if model == 'Constant':
        mean_f = average_x(fx)
        return (average_x((mean_f - fx) ** 2),
                0.5 * (average_x(fx ** 2) - mean_f ** 2))

    z = np.linspace(-1, 1, q)
    x1, x2 = np.meshgrid(z, z, indexing='ij')
    x1, x2 = x1.ravel(), x2.ravel()
    y1, y2 = f(x1), f(x2)
    eps = 1e-8

    if model == 'Linear':
        denominator = x1 - x2
        slope = np.divide(y1 - y2, denominator, out=np.zeros_like(denominator),
                          where=np.abs(denominator) > eps)
        near = np.abs(denominator) <= eps
        if np.any(near):
            h = 1e-6
            slope[near] = (f(x1[near] + h) - f(x1[near] - h)) / (2 * h)
        intercept = y1 - slope * x1
    else:
        denominator = x1 ** 2 + x2 ** 2
        slope = np.divide(x1 * y1 + x2 * y2, denominator,
                          out=np.zeros_like(denominator), where=denominator > eps)
        near = denominator <= eps
        if np.any(near):
            h = 1e-6
            slope[near] = (f(h) - f(-h)) / (2 * h)
        intercept = np.zeros_like(slope)

    grid_weights = np.ones(q)
    grid_weights[[0, -1]] = 0.5
    pair_weights = np.outer(grid_weights, grid_weights).ravel()
    normalizer = pair_weights.sum()
    mean_prediction = np.zeros(n_test)
    second_moment = np.zeros(n_test)
    batch = 5000
    for start in range(0, len(pair_weights), batch):
        stop = start + batch
        prediction = intercept[start:stop, None] + slope[start:stop, None] * x
        weights = pair_weights[start:stop, None]
        mean_prediction += (prediction * weights).sum(axis=0)
        second_moment += ((prediction ** 2) * weights).sum(axis=0)
    mean_prediction /= normalizer
    variance = np.maximum(second_moment / normalizer - mean_prediction ** 2, 0)
    return average_x((mean_prediction - fx) ** 2), average_x(variance)


def learning_curve(f, model, n_list=N_LIST, sigma=0.0, n_datasets=3000, n_test=1000):
    x_test, ein, eout = np.linspace(-1, 1, n_test), [], []
    for n in n_list:
        ins = outs = 0.0
        for _ in range(n_datasets):
            X = np.random.uniform(-1, 1, n)
            y = f(X) + np.random.normal(0, sigma, n)
            ins += np.mean((fit_predict(model, X, y, X) - y) ** 2)
            outs += expected_eout(model, X, y, f, x_test, sigma)
        ein.append(ins / n_datasets)
        eout.append(outs / n_datasets)
    return ein, eout


def print_reference_table():
    n, sigma = 20, 0.3
    X, y = reference_data(n, sigma, SEED)
    print('\n' + '-' * 72)
    print('REFERENCE DATASET')
    print(f'Setup: target=sin(pi*x), x~U(-1,1), n={n}, sigma={sigma}, noise variance={sigma ** 2:.4f}')
    print(f"{'Model':<22} {'Ein':>10} {'Signal':>10} {'Noise^2':>10} {'Eout':>10}")
    print('-' * 72)
    for model in MODELS:
        signal, noise, eout = eout_parts(model, X, y, sigma=sigma)
        ein = np.mean((fit_predict(model, X, y, X) - y) ** 2)
        print(f'{model:<22} {ein:10.6f} {signal:10.6f} {noise:10.6f} {eout:10.6f}')


def main():
    np.random.seed(SEED)
    print('\n' + '=' * 72)
    print('WEEK 3: BIAS-VARIANCE')
    print('Plots: average fit and learning curves')
    print('=' * 72)
    print_reference_table()

    print('\n' + '-' * 110)
    print('BIAS-VARIANCE: ANALYTICAL VS SIMULATION (n=2, sigma=0)')
    print(f"{'Target':<12} {'Model':<22} {'Bias^2 ana':>12} {'Var ana':>12} "
          f"{'Eout ana':>12} {'Eout sim':>12}")
    print('-' * 110)

    x_plot = np.linspace(-1, 1, 500)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey=True)
    for row, (name, f) in enumerate(TARGETS.items()):
        for col, model in enumerate(MODELS):
            s = simulate(f, model)
            bias2, variance = analytical_bias_variance(f, model)
            print(f'{name:<12} {model:<22} {bias2:12.4f} {variance:12.4f} '
                  f'{bias2 + variance:12.4f} {s["eout"]:12.4f}')
            ax = axes[row, col]
            ax.plot(x_plot, f(x_plot), 'g-', lw=2, label='Target f(x)')
            ax.plot(s['x_test'], s['g_bar'], 'r--', lw=2, label='g_bar(x)')
            ax.fill_between(s['x_test'], s['g_bar'] - s['std'], s['g_bar'] + s['std'],
                            color='red', alpha=0.3, label='±1 std')
            for _ in range(20):
                X = np.random.uniform(-1, 1, 2)
                ax.plot(x_plot, fit_predict(model, X, f(X), x_plot), 'k-', alpha=0.2, lw=0.5)
            ax.set(title=f'{name} | {model}', xlim=(-1, 1), ylim=(-4, 4))
            ax.legend(fontsize=9, loc='upper center'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'average_fit.png'), dpi=150); plt.close(fig)

    print('\n' + '=' * 72)
    print('Creating learning-curve plot...')
    print('=' * 72)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row')
    for row, (name, f) in enumerate(TARGETS.items()):
        for col, model in enumerate(MODELS):
            ax = axes[row, col]
            curves = {}
            for sigma in NOISE:
                ein, eout = learning_curve(f, model, sigma=sigma)
                curves[sigma] = (ein, eout)
                ax.plot(N_LIST, np.clip(ein, 0, YMAX[name]), '--', color=COLORS[sigma],
                        label=f'Ein σ={sigma}', alpha=0.7)
                ax.plot(N_LIST, np.clip(eout, 0, YMAX[name]), '-', color=COLORS[sigma],
                        label=f'Eout σ={sigma}', alpha=0.7)
            if row == 1: ax.set_xlabel('n')
            if col == 0: ax.set_ylabel('Expected Error')
            else: ax.tick_params(labelleft=False)
            ax.set(title=f'{name} | {model}', xscale='log', ylim=(0, YMAX[name]))
            ax.legend(fontsize=9, loc='upper center'); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, 'learning_curve.png'), dpi=150); plt.close(fig)
    print('\nSaved: plots/average_fit.png, plots/learning_curve.png')


if __name__ == '__main__':
    main()
