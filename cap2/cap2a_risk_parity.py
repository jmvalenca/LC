import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import cvxpy as cvx
    return cvx, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Cap. 2a · *Risk parity portfolio* (RPP): verificação

    O problema RPP pede $x \ge 0$ com contribuições de risco

    $$x_i\,\varphi_i'(x) \;=\; \frac{x_i\,(Qx)_i}{x^\top Q x} \;=\; b_i, \qquad \textstyle\sum_i b_i = 1 .$$

    O texto resolve-o com variáveis normalizadas $z$ que satisfazem $z_i\,(Qz)_i = b_i$,
    obtidas minimizando $f(z) = \tfrac12 z^\top Q z - \sum_i b_i \log z_i$ com o cone exponencial.

    Este notebook verifica que:

    1. o código CVXPY do texto está **correto** (as contribuições de risco coincidem com $b$);
    2. a normalização indicada no texto, $z_i = x_i/(x^\top Q x)$, está **errada**: dá
       $z_i (Qz)_i = b_i/(x^\top Q x)$. A correta é $z_i = x_i/\sqrt{x^\top Q x} = x_i/\varphi(x)$.
    """)
    return


@app.cell
def _(mo):
    n_ui = mo.ui.slider(3, 60, value=10, label="n")
    seed_ui = mo.ui.number(0, 10_000, value=2, label="semente")
    mo.hstack([n_ui, seed_ui])
    return n_ui, seed_ui


@app.cell
def _(n_ui, np, seed_ui):
    n = n_ui.value
    rng = np.random.default_rng(seed_ui.value)
    _R = rng.random((n, n))
    Q = _R.T @ _R                      # QᵀQ é sempre semidefinida positiva
    b = rng.random(n)
    b = b / b.sum()
    return Q, b, n


@app.cell
def _(Q, b, cvx, n):
    # Código do texto (sem o ciclo while/psd_wrap, que é desnecessário)
    z = cvx.Variable(n)
    t = cvx.Variable(n)                # t_i ≤ log z_i  (igualdade no ótimo, porque b > 0)
    obj = 0.5 * cvx.quad_form(z, cvx.psd_wrap(Q)) - b @ t
    prob = cvx.Problem(cvx.Minimize(obj), [cvx.exp(t) <= z])
    prob.solve()
    z_val = z.value
    return prob, z_val


@app.cell
def _(Q, b, mo, np, prob, z_val):
    x = z_val / z_val.sum()
    contrib = x * (Q @ x) / (x @ Q @ x)
    z_texto = x / (x @ Q @ x)
    z_certo = x / np.sqrt(x @ Q @ x)
    err = lambda zz: np.abs(zz * (Q @ zz) - b).max()

    mo.md(f"""
    **Estado:** `{prob.status}`

    | verificação | erro máximo |
    |---|---|
    | $\\lvert x_i(Qx)_i/x^\\top Qx - b_i\\rvert$ (contribuições de risco) | {np.abs(contrib - b).max():.2e} |
    | $\\lvert z_i(Qz)_i - b_i\\rvert$ com $z$ do solver | {err(z_val):.2e} |
    | normalização do texto, $z = x/(x^\\top Q x)$ | {err(z_texto):.2e} |
    | normalização correta, $z = x/\\sqrt{{x^\\top Q x}}$ | {err(z_certo):.2e} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Notas

    * $\sum_i x_i\,\varphi_i'(x) = 1$ não vale “para qualquer função de risco”: vale quando
      $\mathcal R$ é homogénea de grau 2 (teorema de Euler aplicado a $\varphi = \sqrt{\mathcal R}$).
    * A justificação da nota do texto (“os zeros de $f$ são os ínfimos”) não se aplica, porque esta
      $f$ não é não-negativa. O argumento certo: $f$ é convexa em $z > 0$, logo o mínimo é o ponto
      onde $\nabla f = 0$, isto é $(Qz)_i - b_i/z_i = 0$.
    * A restrição $(z_i, 1, t_i) \in \mathcal K^{\exp}$ dá $t_i \le \log z_i$; a igualdade obtém-se
      no ótimo porque se minimiza $-b^\top t$ com $b > 0$.
    """)
    return


if __name__ == "__main__":
    app.run()
