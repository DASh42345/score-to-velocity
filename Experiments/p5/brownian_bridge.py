"""
Броуновский мост для референсного процесса dX = sigma dW (как в DSBM).

Мост на [0, 1], закреплённый в X_0 = x0 и X_1 = x1, имеет маргинал

    X_t | x0, x1  ~  N( (1 - t) x0 + t x1,  sigma^2 t (1 - t) I ).

Функции:
    sample_marginal     одна точка x_t на мосту (для обучения bridge matching)
    sample_path_wt      траектория через конструкцию B_t = W_t - t W_1
    sample_path_seq     траектория точным последовательным сэмплированием
    dsbm_training_pair  батч (t, x_t, target) для forward bridge matching

Соглашения по формам: x0, x1 имеют форму (B, d), траектории (T, B, d).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


# ════════════════════════════ моменты моста ════════════════════════════


def bridge_mean(x0: Array, x1: Array, t: Array | float) -> Array:
    """Среднее моста: прямая между x0 и x1."""
    return (1 - t) * x0 + t * x1


def bridge_std(t: Array | float, sigma: float) -> Array:
    """Стандартное отклонение моста, sigma * sqrt(t (1 - t))."""
    return sigma * np.sqrt(t * (1 - t))


# ═════════════════════════════ сэмплирование ═════════════════════════════


def sample_marginal(
    x0: Array, x1: Array, t: Array, sigma: float, rng: np.random.Generator
) -> Array:
    """Одна точка x_t на мосту; t имеет форму (B, 1) и лежит в [0, 1]."""
    noise = rng.standard_normal(x0.shape)
    return bridge_mean(x0, x1, t) + bridge_std(t, sigma) * noise


def sample_path_wt(
    x0: Array, x1: Array, ts: Array, sigma: float, rng: np.random.Generator
) -> Array:
    """Траектория на сетке ts (ts[0] = 0, ts[-1] = 1).

    Строим броуновское движение W на сетке; B_t = W_t - t W_1 даёт мост 0 -> 0,
    который затем сдвигаем на прямую между x0 и x1.
    """
    shape = x0.shape
    dW = np.sqrt(np.diff(ts))[:, None, None] * rng.standard_normal(
        (len(ts) - 1, *shape)
    )
    W = np.concatenate([np.zeros((1, *shape)), np.cumsum(dW, axis=0)])

    t = ts[:, None, None]
    return bridge_mean(x0, x1, t) + sigma * (W - t * W[-1])


def sample_path_seq(
    x0: Array, x1: Array, ts: Array, sigma: float, rng: np.random.Generator
) -> Array:
    """Та же траектория, но шаг за шагом.

    Из X_s = x в момент t > s, обозначив alpha = (t - s) / (1 - s):
        mean = x + alpha (x1 - x)
        var  = sigma^2 alpha (1 - t)
    Формула точная, ошибки дискретизации нет.
    """
    path = np.empty((len(ts), *x0.shape))
    path[0] = x0

    for i, (s, t) in enumerate(zip(ts[:-1], ts[1:]), start=1):
        x = path[i - 1]
        alpha = (t - s) / (1 - s)
        mean = x + alpha * (x1 - x)
        std = sigma * np.sqrt(alpha * (1 - t))
        path[i] = mean + std * rng.standard_normal(x.shape)

    return path


def dsbm_training_pair(
    x0: Array, x1: Array, sigma: float, rng: np.random.Generator, eps: float = 1e-3
) -> tuple[Array, Array, Array]:
    """Батч (t, x_t, target) для forward bridge matching.

    Сеть учит дрифт моста (x1 - x_t) / (1 - t). Время обрезаем у единицы,
    потому что при t -> 1 таргет взрывается.
    """
    t = rng.uniform(eps, 1 - eps, size=(x0.shape[0], 1))
    xt = sample_marginal(x0, x1, t, sigma, rng)
    target = (x1 - xt) / (1 - t)
    return t, xt, target


# ═══════════════════════════ проверки и картинка ═══════════════════════════


def check_marginals(
    x0: Array,
    x1: Array,
    sigma: float,
    rng: np.random.Generator,
    mu0: float,
    mu1: float,
    endpoint_std: float,
    t: float = 0.3,
) -> None:
    """Сравнивает эмпирические моменты x_t с теорией для всех трёх сэмплеров.

    Концы случайные и независимые, x0 ~ N(mu0, s^2), x1 ~ N(mu1, s^2), поэтому
        E[x_t]   = (1 - t) mu0 + t mu1
        Var[x_t] = ((1 - t)^2 + t^2) s^2 + sigma^2 t (1 - t).
    """
    mean_th = bridge_mean(mu0, mu1, t)
    var_th = ((1 - t) ** 2 + t**2) * endpoint_std**2 + bridge_std(t, sigma) ** 2

    ts = np.linspace(0, 1, 11)
    k = int(np.argmin(np.abs(ts - t)))

    rows = [("theory", mean_th, var_th, None)]

    xt = sample_marginal(x0, x1, np.full((len(x0), 1), t), sigma, rng)
    rows.append(("marginal", xt.mean(), xt.var(), None))

    for name, sampler in [
        ("W_t - t W_1", sample_path_wt),
        ("sequential", sample_path_seq),
    ]:
        path = sampler(x0, x1, ts, sigma, rng)
        end_err = np.abs(path[-1] - x1).max()
        rows.append((name, path[k].mean(), path[k].var(), end_err))

    print(f"Моменты x_t при t = {t}:")
    print(f"  {'':<12}{'mean':>9}{'var':>9}{'|X_1 - x1|':>13}")
    for name, m, v, err in rows:
        err_str = f"{err:13.1e}" if err is not None else ""
        print(f"  {name:<12}{m:9.4f}{v:9.4f}{err_str}")


def plot_paths(
    x0: Array,
    x1: Array,
    sigma: float,
    rng: np.random.Generator,
    n_paths: int = 8,
    out: str = "brownian_bridge.png",
) -> None:
    """Несколько траекторий (d = 1) вместе с их коридорами ±2 std."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ts = np.linspace(0, 1, 501)
    a, b = x0[:n_paths], x1[:n_paths]
    paths = sample_path_seq(a, b, ts, sigma, rng)[..., 0]  # (T, n_paths)

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n_paths))
    std = bridge_std(ts, sigma)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, c in enumerate(colors):
        mean = bridge_mean(a[i, 0], b[i, 0], ts)
        ax.fill_between(ts, mean - 2 * std, mean + 2 * std, color=c, alpha=0.06, lw=0)
        ax.plot(ts, mean, color=c, lw=0.8, ls="--", alpha=0.6)
        ax.plot(ts, paths[:, i], color=c, lw=1.1)

    ax.scatter(
        np.zeros(n_paths),
        a[:, 0],
        color=colors,
        edgecolor="k",
        s=40,
        zorder=5,
        label="$x_0$",
    )
    ax.scatter(
        np.ones(n_paths),
        b[:, 0],
        color=colors,
        edgecolor="k",
        marker="s",
        s=40,
        zorder=5,
        label="$x_1$",
    )

    ax.set(
        xlabel="$t$", ylabel="$X_t$", xlim=(-0.02, 1.02), title=rf"$\sigma = {sigma:g}$"
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Картинка сохранена в {out}")


# ═════════════════════════════════ main ═════════════════════════════════


def main() -> None:
    rng = np.random.default_rng(0)

    sigma = 1.0  # шум референсного процесса
    B, d = 200_000, 1
    mu0, mu1 = -1.0, 2.0
    endpoint_std = 0.3  # разброс концов вокруг mu0 и mu1

    x0 = rng.normal(mu0, endpoint_std, size=(B, d))
    x1 = rng.normal(mu1, endpoint_std, size=(B, d))

    check_marginals(x0, x1, sigma, rng, mu0, mu1, endpoint_std)
    plot_paths(x0, x1, sigma, rng)


if __name__ == "__main__":
    main()
