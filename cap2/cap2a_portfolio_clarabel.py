import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import warnings
    import numpy as np
    import scipy.sparse as sp
    import clarabel
    import cvxpy as cvx

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    return clarabel, cvx, mo, np, sp


@app.cell
def _(mo):
    mo.md(r"""
    # Cap. 2a · Problema do *portfolio* em Clarabel: verificação do código

    Este notebook corre os dois exemplos Clarabel da secção *Problema do “portfolio” em KKT*
    e compara-os com uma resolução de referência em CVXPY.

    1. **1.º problema** (objetivo quadrático, restrições lineares):
       $\min \tfrac12 x^\top Q x - \gamma\,\mu^\top x$ com $\mathsf 1^\top x = 1,\ x\ge 0$.
       O código do texto está **correto**.
    2. **2.º problema** (objetivo linear, restrição de 2.ª ordem):
       $\min \gamma\,\sigma - \mu^\top x$ com $\sigma \ge \|G x\|,\ \mathsf 1^\top x = 1,\ x\ge 0$.
       Tal como está no texto, o Clarabel declara o problema **inviável**:

       * as linhas de $A$ estão na ordem $r \mid z\,(n) \mid s\,(n) \mid t$, mas os cones estão na ordem
         `ZeroConeT(1), NonnegativeConeT(n), SecondOrderConeT(n+1)`. O Clarabel associa os cones às
         linhas **pela ordem**, por isso o cone não-negativo fica sobre $z = Gx$ e o SOC sobre $(s, t)$;
       * `zero_ = sp.csc_matrix([1.0]*n)` é um vetor de **uns**, não de zeros.

       A versão corrigida ordena as linhas como $r \mid s \mid t \mid z$.
    """)
    return


@app.cell
def _(mo):
    n_ui = mo.ui.slider(4, 200, value=8, label="n (número de ativos)")
    gamma_ui = mo.ui.number(0.1, 10.0, value=1.0, step=0.1, label="γ")
    seed_ui = mo.ui.number(0, 10_000, value=1, label="semente")
    mo.hstack([n_ui, gamma_ui, seed_ui])
    return gamma_ui, n_ui, seed_ui


@app.cell
def _(np, sp):
    # Funções auxiliares, tal como no texto
    ZeroP = lambda n: sp.csc_matrix((n, n))
    ZeroC = lambda n: sp.csc_matrix(np.zeros(n).reshape((n, 1)))

    def QG(rng, n, chy=False, band=3, beta=0.1):
        """Gera G triangular superior (banda) e devolve G ou triu(GᵀG)."""
        G = np.identity(n) + sum(
            (beta**d) * np.diag(rng.random(n - d), k=d) for d in range(1, band + 1)
        )
        if chy:
            return sp.csc_matrix(np.triu(G))
        return sp.csc_matrix(np.triu(G.T @ G))

    return QG, ZeroC, ZeroP


@app.cell
def _(clarabel):
    def resolve(P, q, A, b, cones):
        """Chama o Clarabel e devolve a solução (o estado vem em .status).

        Nota: `solver.solve()` não lança exceção quando o problema é inviável,
        por isso o `try/except` do texto nunca mostra a inviabilidade.
        """
        settings = clarabel.DefaultSettings()
        settings.verbose = False          # no texto: clarabel.verbose = False (sem efeito)
        return clarabel.DefaultSolver(P, q, A, b, cones, settings).solve()

    return (resolve,)


@app.cell
def _(QG, gamma_ui, n_ui, np, seed_ui):
    n = n_ui.value
    gamma = gamma_ui.value
    rng = np.random.default_rng(seed_ui.value)
    Psym = QG(rng, n)            # triu(Q), Q = GᵀG
    G = QG(rng, n, chy=True)     # outra G (como no texto, gerada de novo)
    mu = rng.random(n)
    return G, Psym, gamma, mu, n


@app.cell
def _(mo):
    mo.md(r"""
    ## 1.º problema: objetivo quadrático
    """)
    return


@app.cell
def _(Psym, clarabel, cvx, gamma, mu, n, np, resolve, sp):
    _q = -gamma * mu
    _A = sp.vstack([sp.csc_matrix([1.0] * n), -sp.identity(n)]).tocsc()
    _b = np.concatenate([[1.0], np.zeros(n)])
    _cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n)]
    sol1 = resolve(Psym, _q, _A, _b, _cones)

    _Q = Psym.toarray()
    _Q = np.triu(_Q) + np.triu(_Q, 1).T
    _x = cvx.Variable(n)
    ref1 = cvx.Problem(
        cvx.Minimize(0.5 * cvx.quad_form(_x, cvx.psd_wrap(_Q)) - gamma * mu @ _x),
        [cvx.sum(_x) == 1, _x >= 0],
    )
    ref1.solve()
    return ref1, sol1


@app.cell
def _(mo, ref1, sol1):
    mo.md(f"""
    | | estado | valor ótimo |
    |---|---|---|
    | Clarabel (código do texto, `P = triu(Q)`) | `{sol1.status}` | {sol1.obj_val:.6f} |
    | CVXPY, $\\tfrac12 x^\\top Q x - \\gamma\\mu^\\top x$ | `{ref1.status}` | {ref1.value:.6f} |

    Os valores coincidem: o código usa $P = Q$ (forma de Markowitz com o fator $\\tfrac12$).
    No texto escreve-se $P \\equiv \\tfrac12 Q$, o que não corresponde a nenhuma das duas formas:
    para $x^\\top Q x$ seria $P = 2Q$; para $\\tfrac12 x^\\top Q x$ é $P = Q$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2.º problema: restrição de 2.ª ordem
    """)
    return


@app.cell
def _(G, ZeroC, ZeroP, clarabel, gamma, mu, n, np, resolve, sp):
    _P = ZeroP(n + 1)
    _q = np.concatenate([[gamma], -mu])
    _one = sp.csc_matrix([1.0] * n)
    _id = sp.identity(n)
    _cones = [clarabel.ZeroConeT(1), clarabel.NonnegativeConeT(n),
              clarabel.SecondOrderConeT(n + 1)]

    # (a) exatamente como no texto
    _zero_texto = sp.csc_matrix([1.0] * n)          # (sic) uns
    _A_texto = sp.block_array([[[0], _one], [ZeroC(n), G], [ZeroC(n), -_id],
                               [[-1], _zero_texto]]).tocsc()
    _b = np.concatenate([[1.0], np.zeros(2 * n + 1)])
    sol_texto = resolve(_P, _q, _A_texto, _b, _cones)

    # (b) corrigido: linhas r | s | t | z, pela ordem dos cones
    _zero = sp.csc_matrix((1, n))
    _A_ok = sp.block_array([[[0], _one],            # 1ᵀx + r = 1,  r ∈ {0}
                            [ZeroC(n), -_id],       # -x + s = 0,   s ∈ ℝ₊ⁿ
                            [[-1], _zero],          # -σ + t = 0
                            [ZeroC(n), -G]]).tocsc()  # -Gx + z = 0, (t, z) ∈ SOC
    sol_ok = resolve(_P, _q, _A_ok, _b, _cones)
    return sol_ok, sol_texto


@app.cell
def _(G, cvx, gamma, mu, n):
    _x = cvx.Variable(n)
    ref2 = cvx.Problem(cvx.Minimize(gamma * cvx.norm(G.toarray() @ _x) - mu @ _x),
                       [cvx.sum(_x) == 1, _x >= 0])
    ref2.solve()
    x_ref2 = _x.value
    return ref2, x_ref2


@app.cell
def _(mo, np, ref2, sol_ok, sol_texto, x_ref2):
    _x_ok = np.array(sol_ok.x)[1:]
    _dif = np.abs(_x_ok - x_ref2).max()
    _ov = lambda s: f"{s.obj_val:.6f}" if np.isfinite(s.obj_val) else "—"
    mo.md(f"""
    | | estado | valor ótimo |
    |---|---|---|
    | Clarabel, código do texto | `{sol_texto.status}` | {_ov(sol_texto)} |
    | Clarabel, linhas reordenadas e `zero_` corrigido | `{sol_ok.status}` | {_ov(sol_ok)} |
    | CVXPY, $\\gamma\\|Gx\\| - \\mu^\\top x$ | `{ref2.status}` | {ref2.value:.6f} |

    Diferença máxima entre o portfolio corrigido e o de referência: {_dif:.2e}.

    A matriz $A$ por blocos escrita no texto tem o mesmo problema de ordem e ainda
    $+\\mathsf{{Id}}_n$ onde deve ser $-\\mathsf{{Id}}_n$ (a restrição é $-x + s = 0$).
    """)
    return


if __name__ == "__main__":
    app.run()
