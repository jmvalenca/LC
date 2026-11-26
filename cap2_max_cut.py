import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cp
    from ortools.sat.python import cp_model
    return cp, cp_model, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Cap. 2 · Corte máximo (*max-cut*): CP e relaxação SDP

    Verificação do exemplo «Corte máximo» do *Capítulo 2: Programação com Restrições*.

    - **CP (CP-SAT):** $\max \sum_e w_e y_e$ com $y_e \le x_u + x_v$, $y_e \le 2 - x_u - x_v$ e $x_{v_0}=0$.
    - **SDP primal:** $\max \tfrac14\langle L, X\rangle$ com $X_{vv}=1$ e $X \in \mathcal{K}^{sdp}_n$.
    - **SDP dual:** $\min \sum_v z_v$ com $\operatorname{Diag}(z) - \tfrac14 L \in \mathcal{K}^{sdp}_n$.
    - **Arredondamento de Goemans–Williamson:** $S = \{v \mid r^\top \mathbf{v}_v \ge 0\}$, com $r \sim \mathcal{N}(0,\mathrm{Id}_n)$.
    """)
    return


@app.cell
def _(mo):
    grafo_ui = mo.ui.dropdown(["C5", "aleatório"], value="C5", label="grafo")
    n_ui = mo.ui.slider(3, 30, value=12, label="n (nodos, aleatório)")
    dens_ui = mo.ui.slider(0.1, 1.0, step=0.05, value=0.4, label="densidade")
    pesos_ui = mo.ui.checkbox(value=False, label="pesos aleatórios em {1,…,5}")
    seed_ui = mo.ui.number(0, 10_000, value=1, label="semente")
    trials_ui = mo.ui.slider(1, 500, value=100, label="sorteios GW")
    mo.vstack([
        mo.hstack([grafo_ui, n_ui, dens_ui]),
        mo.hstack([pesos_ui, seed_ui, trials_ui]),
    ])
    return dens_ui, grafo_ui, n_ui, pesos_ui, seed_ui, trials_ui


@app.cell
def _(dens_ui, grafo_ui, n_ui, np, pesos_ui, seed_ui):
    rng = np.random.default_rng(seed_ui.value)
    if grafo_ui.value == "C5":
        n = 5
        E = [(k, (k + 1) % n) for k in range(n)]
    else:
        n = n_ui.value
        E = [(u, v) for u in range(n) for v in range(u + 1, n)
             if rng.random() < dens_ui.value]
    w = {e: (int(rng.integers(1, 6)) if pesos_ui.value else 1) for e in E}

    # laplaciano L = D - W
    L = np.zeros((n, n))
    for (u, v), we in w.items():
        L[u, u] += we; L[v, v] += we
        L[u, v] -= we; L[v, u] -= we
    return E, L, n, rng, w


@app.cell
def _(E, cp_model, n, w):
    def max_cut_cp():
        m = cp_model.CpModel()
        x = [m.NewBoolVar(f"x{v}") for v in range(n)]
        y = {e: m.NewBoolVar(f"y{e}") for e in E}
        for (u, v), ye in y.items():
            m.Add(ye <= x[u] + x[v])
            m.Add(ye <= 2 - x[u] - x[v])
        m.Add(x[0] == 0)                      # quebra da simetria S ↔ V∖S
        m.Maximize(sum(w[e] * y[e] for e in E))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 30
        st = s.Solve(m)
        S = {v for v in range(n) if s.Value(x[v])}
        return s.ObjectiveValue(), S, s.StatusName(st)

    MC, S_cp, status_cp = max_cut_cp()
    return MC, S_cp, status_cp


@app.cell
def _(L, cp, n, np):
    # primal
    X = cp.Variable((n, n), PSD=True)
    primal = cp.Problem(cp.Maximize(cp.trace(L @ X) / 4), [cp.diag(X) == 1])
    primal.solve(solver=cp.CLARABEL)

    # dual
    z = cp.Variable(n)
    dual = cp.Problem(cp.Minimize(cp.sum(z)), [cp.diag(z) - L / 4 >> 0])
    dual.solve(solver=cp.CLARABEL)

    Xv = X.value
    eig = np.linalg.eigvalsh(Xv)
    rank = int(np.sum(eig > 1e-6 * eig.max()))
    return Xv, dual, primal, rank


@app.cell
def _(E, Xv, n, np, rng, trials_ui, w):
    # X é PSD (em geral singular): decomposição espectral em vez de Cholesky
    lam, U = np.linalg.eigh(Xv)
    V = (U * np.sqrt(np.clip(lam, 0, None))).T     # colunas v_v, com X = VᵀV

    def peso(S):
        return sum(we for (u, v), we in w.items() if (u in S) != (v in S))

    cortes = []
    for _ in range(trials_ui.value):
        r = rng.standard_normal(n)
        S = {v for v in range(n) if r @ V[:, v] >= 0}
        cortes.append(peso(S))
    cortes = np.array(cortes)

    # valor esperado exato do arredondamento: Σ w_e θ_e / π
    esperado = sum(we * np.arccos(np.clip(Xv[u, v], -1, 1)) / np.pi
                   for (u, v), we in w.items())
    return cortes, esperado


@app.cell
def _(E, MC, S_cp, cortes, dual, esperado, mo, n, np, primal, rank, status_cp):
    alpha = min(2 / np.pi * t / (1 - np.cos(t)) for t in np.linspace(1e-3, np.pi, 200_000))
    exato = ""
    if n == 5 and len(E) == 5:
        exato = rf"""
    **$C_5$, valores exatos:** $\mathrm{{MC}}=4$ e $\mathrm{{SDP}} = (25+5\sqrt5)/8 = {(25 + 5 * 5**0.5) / 8:.6f}$.
    """
    mo.md(rf"""
    | grandeza | valor |
    |---|---|
    | nodos / ramos | {n} / {len(E)} |
    | MC (CP-SAT, {status_cp}) | {MC:g}, $S = {sorted(S_cp)}$ |
    | SDP primal | {primal.value:.6f} |
    | SDP dual | {dual.value:.6f} |
    | característica de $X$ | {rank} |
    | MC / SDP | {MC / primal.value:.4f} |
    | GW: média / máximo dos sorteios | {cortes.mean():.3f} / {cortes.max():g} |
    | GW: valor esperado $\sum_e w_e\theta_e/\pi$ | {esperado:.4f} |
    | $\alpha$ (Goemans–Williamson) | {alpha:.5f} |
    | esperado / SDP | {esperado / primal.value:.4f} |
    {exato}
    """)
    return


if __name__ == "__main__":
    app.run()
