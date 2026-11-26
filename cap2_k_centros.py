import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import itertools
    import marimo as mo
    import numpy as np
    from ortools.sat.python import cp_model
    return cp_model, itertools, mo, np


@app.cell
def _(mo):
    mo.md(r"""
    # Cap. 2 · Problemas dos $K$-centros num espaço de vetores de *bits*

    Exemplo da secção «Problemas do $k$-centros num espaço de vetores de *bits*» do
    *Capítulo 2: Programação com Restrições*.

    O *dataset* $E$ tem $M$ palavras de $n$ *bits*; a distância é a de Hamming
    $d(x,y) = \sum_i x_i + y_i - 2\,x_i y_i$. Procuram-se $K$ centros tais que nenhuma
    experiência fique a distância superior a $t$ do centro que lhe é atribuído.

    **1.ª abordagem** — os centros $a_k \in \mathcal{B}_n$ são variáveis. O produto
    $X_{k,j}\cdot d(a_k, e_j)$ é eliminado com uma restrição *big-M*, que é linear porque $E$ é constante:

    $$
    \textstyle\sum_{i=1}^{n}\big(A_{k,i} + E_{j,i} - 2A_{k,i}E_{j,i}\big) \;\le\; t + n\,(1 - X_{k,j})
    \qquad \forall k, j .
    $$

    **2.ª abordagem** — os centros escolhem-se num conjunto finito $\mathcal{C}$ de $N$ candidatos,
    com a matriz constante $D_{i,j} = d(c_i, e_j)$:

    - (i) $k$ fixo, minimizar $t$: $\;\sum_i A_i = k$, $\;\sum_i X_{i,j} = 1$, $\;X_{i,j} \le A_i$, $\;\sum_i X_{i,j} D_{i,j} \le t$;
    - (ii) $t$ fixo, minimizar $k$: é um **Set Cover**, $\;\min \sum_i A_i$ com $\sum_{i\,:\,D_{i,j}\le t} A_i \ge 1$.

    Com $\mathcal{C} = \mathcal{B}_n$ (todas as $2^n$ palavras) a 2.ª abordagem tem de dar o mesmo
    ótimo que a 1.ª; com $\mathcal{C} = E$ (centros escolhidos entre as experiências) o ótimo pode ser maior.
    """)
    return


@app.cell
def _(mo):
    M_ui = mo.ui.slider(4, 40, value=12, label="M (experiências)")
    n_ui = mo.ui.slider(3, 10, value=6, label="n (bits)")
    K_ui = mo.ui.slider(1, 6, value=2, label="K (centros)")
    t_ui = mo.ui.slider(0, 10, value=2, label="t (estratégia ii)")
    seed_ui = mo.ui.number(0, 10_000, value=1, label="semente")
    mo.vstack([mo.hstack([M_ui, n_ui, K_ui]), mo.hstack([t_ui, seed_ui])])
    return K_ui, M_ui, n_ui, seed_ui, t_ui


@app.cell
def _(M_ui, n_ui, np, seed_ui):
    rng = np.random.default_rng(seed_ui.value)
    M, n = M_ui.value, n_ui.value
    E = rng.integers(0, 2, size=(M, n))               # dataset: M palavras de n bits

    def hamming(u, v):
        return int(np.sum(u != v))
    return E, M, hamming, n


@app.cell
def _(E, K_ui, M, cp_model, n):
    def abordagem_1(K):
        """Centros livres em B_n; restrição big-M linear; minimiza t."""
        mdl = cp_model.CpModel()
        A = [[mdl.NewBoolVar(f"A{k}_{i}") for i in range(n)] for k in range(K)]
        X = [[mdl.NewBoolVar(f"X{k}_{j}") for j in range(M)] for k in range(K)]
        t = mdl.NewIntVar(0, n, "t")
        for j in range(M):                                            # (a)
            mdl.Add(sum(X[k][j] for k in range(K)) == 1)
        for k in range(K):                                            # (b) com big-M
            for j in range(M):
                dist = sum(A[k][i] if E[j, i] == 0 else 1 - A[k][i] for i in range(n))
                mdl.Add(dist <= t + n * (1 - X[k][j]))
        mdl.Minimize(t)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 30
        st = s.Solve(mdl)
        centros = [[s.Value(A[k][i]) for i in range(n)] for k in range(K)]
        return int(s.Value(t)), centros, s.StatusName(st)

    t1, centros1, st1 = abordagem_1(K_ui.value)
    return centros1, st1, t1


@app.cell
def _(E, K_ui, M, cp_model, hamming, itertools, n, np, t_ui):
    def candidatos(tipo):
        if tipo == "E":
            return np.unique(E, axis=0)
        return np.array(list(itertools.product([0, 1], repeat=n)))

    def matriz_D(C):
        return [[hamming(C[i], E[j]) for j in range(M)] for i in range(len(C))]

    def estrategia_i(C, K):
        """k fixo, minimizar t (programação linear inteira)."""
        D = matriz_D(C)
        N = len(C)
        mdl = cp_model.CpModel()
        A = [mdl.NewBoolVar(f"A{i}") for i in range(N)]
        X = [[mdl.NewBoolVar(f"X{i}_{j}") for j in range(M)] for i in range(N)]
        t = mdl.NewIntVar(0, n, "t")
        mdl.Add(sum(A) == K)
        for j in range(M):
            mdl.Add(sum(X[i][j] for i in range(N)) == 1)
            mdl.Add(sum(D[i][j] * X[i][j] for i in range(N)) <= t)
            for i in range(N):
                mdl.Add(X[i][j] <= A[i])
        mdl.Minimize(t)
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 30
        st = s.Solve(mdl)
        return int(s.Value(t)), s.StatusName(st)

    def estrategia_ii(C, t):
        """t fixo, minimizar k: Set Cover."""
        D = matriz_D(C)
        N = len(C)
        mdl = cp_model.CpModel()
        A = [mdl.NewBoolVar(f"A{i}") for i in range(N)]
        for j in range(M):
            mdl.Add(sum(A[i] for i in range(N) if D[i][j] <= t) >= 1)
        mdl.Minimize(sum(A))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 30
        st = s.Solve(mdl)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, s.StatusName(st)
        return int(s.ObjectiveValue()), s.StatusName(st)

    resultados = {}
    for tipo in ("Bn", "E"):
        C = candidatos(tipo)
        resultados[tipo] = (len(C), estrategia_i(C, K_ui.value), estrategia_ii(C, t_ui.value))
    return (resultados,)


@app.cell
def _(K_ui, M, centros1, mo, n, resultados, st1, t1, t_ui):
    (NB, (tB, stB), (kB, skB)) = resultados["Bn"]
    (NE, (tE, stE), (kE, skE)) = resultados["E"]
    mo.md(rf"""
    **Dados:** $M = {M}$ experiências de $n = {n}$ *bits*, $K = {K_ui.value}$, $t = {t_ui.value}$ (estratégia ii).

    | formulação | candidatos $N$ | resultado |
    |---|---|---|
    | 1.ª abordagem (big-M), min $t$ | $2^{{{n}}}$ implícitos | $t^* = {t1}$ ({st1}) |
    | 2.ª abordagem (i), $\mathcal{{C}} = \mathcal{{B}}_n$, min $t$ | {NB} | $t^* = {tB}$ ({stB}) |
    | 2.ª abordagem (i), $\mathcal{{C}} = E$, min $t$ | {NE} | $t^* = {tE}$ ({stE}) |
    | 2.ª abordagem (ii), $\mathcal{{C}} = \mathcal{{B}}_n$, min $k$ | {NB} | $k^* = {kB}$ ({skB}) |
    | 2.ª abordagem (ii), $\mathcal{{C}} = E$, min $k$ | {NE} | $k^* = {kE}$ ({skE}) |

    **1.ª abordagem e 2.ª com $\mathcal{{C}} = \mathcal{{B}}_n$ coincidem?** {"sim ✓" if t1 == tB else "**não** ✗"}

    Centros da 1.ª abordagem: {["".join(map(str, c)) for c in centros1]}
    """)
    return


if __name__ == "__main__":
    app.run()
