import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from ortools.linear_solver import pywraplp
    return mo, np, pywraplp


@app.cell
def _(mo):
    mo.md(r"""
    # Cap. 2 · Alocação de operadores a máquinas

    Implementação do exemplo do capítulo, com OR-Tools (`SCIP` para IP/MIP, `GLOP` para LP).

    **Dados**: $N$ operadores, $M$ máquinas, matriz de habilitação $\mathbf{H}\in\{0,1\}^{N\times M}$,
    mínimos $\mathbf{t}_j$ de operadores por máquina e máximos $\mathbf{m}_i$ de máquinas por operador.

    **Variáveis**: $X_{i,j}\in\{0,1\}$ e, conforme a variante, as «flags» $Y_j$ (máquina ativa) e $Z_i$ (operador ativo).

    | variante | obrigações | limitações | objetivo |
    |---|---|---|---|
    | base | $\sum_i X_{i,j}\ge \mathbf{t}_j$ | $\sum_j X_{i,j}\le \mathbf{m}_i,\; X_{i,j}\le \mathbf{H}_{i,j}$ | (custo arbitrário de cada alocação) |
    | rendimento | $\sum_i X_{i,j}\ge \mathbf{t}_j\,Y_j$ | $+\; X_{i,j}\le Y_j$ | $\max \sum_j r_j Y_j$ |
    | custo | $\sum_i X_{i,j}\ge \mathbf{t}_j$ | $\sum_j X_{i,j}\le \mathbf{m}_i Z_i$ | $\min \sum_i c_i Z_i$ |
    | combinado | as da variante rendimento | as das duas variantes | $\max \sum_j r_j Y_j - \sum_i c_i Z_i$ |

    As restrições com $Y_j$ e $Z_i$ são **controladas** no sentido do texto: $\sum_i X_{i,j} - \mathbf{t}_j Y_j \ge 0$
    é uma obrigação controlada ($c_0=-\mathbf{t}_j$), e $X_{i,j}-Y_j\le 0$ e $\sum_j X_{i,j}-\mathbf{m}_iZ_i\le 0$
    são limitações controladas ($c_0=-1$ e $c_0=-\mathbf{m}_i$).
    """)
    return


@app.cell
def _(mo):
    N_ui = mo.ui.slider(2, 30, value=8, label="N (operadores)")
    M_ui = mo.ui.slider(2, 20, value=6, label="M (máquinas)")
    dens_ui = mo.ui.slider(0.1, 1.0, step=0.05, value=0.5, label="densidade de H")
    tmax_ui = mo.ui.slider(1, 4, value=2, label="máx. de t_j")
    mmax_ui = mo.ui.slider(1, 4, value=3, label="máx. de m_i")
    seed_ui = mo.ui.number(0, 10_000, value=1, label="semente")
    pesos_ui = mo.ui.checkbox(value=False, label="pesos r_j, c_i aleatórios (senão todos 1)")
    mo.vstack([
        mo.hstack([N_ui, M_ui, dens_ui]),
        mo.hstack([tmax_ui, mmax_ui, seed_ui, pesos_ui]),
    ])
    return M_ui, N_ui, dens_ui, mmax_ui, pesos_ui, seed_ui, tmax_ui


@app.cell
def _(M_ui, N_ui, dens_ui, mmax_ui, np, pesos_ui, seed_ui, tmax_ui):
    N, M = N_ui.value, M_ui.value
    rng = np.random.default_rng(seed_ui.value)
    H = (rng.random((N, M)) < dens_ui.value).astype(int)
    t = rng.integers(1, tmax_ui.value + 1, size=M)
    m = rng.integers(1, mmax_ui.value + 1, size=N)
    # garante que cada máquina tem, pelo menos, t_j operadores habilitados
    for _j in range(M):
        _falta = t[_j] - H[:, _j].sum()
        if _falta > 0:
            _livres = np.flatnonzero(H[:, _j] == 0)
            H[rng.choice(_livres, size=min(_falta, len(_livres)), replace=False), _j] = 1
    if pesos_ui.value:
        r = rng.integers(1, 6, size=M)
        c = rng.integers(1, 4, size=N)
    else:
        r = np.ones(M, dtype=int)
        c = np.ones(N, dtype=int)
    custoX = rng.integers(1, 10, size=(N, M))  # custo de cada alocação, só na variante base
    return H, M, N, c, custoX, m, r, rng, t


@app.cell
def _(H, M, N, c, m, mo, r, t):
    def _linha(i):
        return f"| op {i} | " + " | ".join(str(H[i, j]) for j in range(M)) + f" | **{m[i]}** | {c[i]} |"

    mo.md(
        "**Dados gerados** (H, com $\\mathbf{m}_i$ e $c_i$ por linha, $\\mathbf{t}_j$ e $r_j$ por coluna)\n\n"
        + "| | " + " | ".join(f"M{j}" for j in range(M)) + " | m | c |\n"
        + "|---" * (M + 3) + "|\n"
        + "\n".join(_linha(i) for i in range(N)) + "\n"
        + "| **t** | " + " | ".join(f"**{t[j]}**" for j in range(M)) + " | | |\n"
        + "| r | " + " | ".join(str(r[j]) for j in range(M)) + " | | |\n\n"
        + f"Capacidade total $\\sum_i \\mathbf{{m}}_i = {m.sum()}$; procura total $\\sum_j \\mathbf{{t}}_j = {t.sum()}$."
    )
    return


@app.cell
def _(H, M, N, c, custoX, m, np, pywraplp, r, t):
    ESTADO = {
        pywraplp.Solver.OPTIMAL: "ótimo",
        pywraplp.Solver.FEASIBLE: "admissível",
        pywraplp.Solver.INFEASIBLE: "inadmissível",
        pywraplp.Solver.UNBOUNDED: "ilimitado",
        pywraplp.Solver.ABNORMAL: "anormal",
        pywraplp.Solver.NOT_SOLVED: "não resolvido",
    }

    def resolve(variante, relaxar=False, agregado=False, custos=None):
        """Constrói e resolve uma variante do modelo.

        variante : 'base' | 'rendimento' | 'custo' | 'combinado'
        relaxar  : True → variáveis reais em [0,1] (LP, GLOP); False → binárias (IP, SCIP)
        agregado : True → substitui X_ij ≤ Y_j por  Σ_i X_ij ≤ (Σ_i H_ij)·Y_j
        custos   : matriz N×M de custos das alocações (só na variante base)
        """
        s = pywraplp.Solver.CreateSolver("GLOP" if relaxar else "SCIP")
        var = (lambda nome: s.NumVar(0, 1, nome)) if relaxar else (lambda nome: s.IntVar(0, 1, nome))

        X = {(i, j): var(f"X_{i}_{j}") for i in range(N) for j in range(M)}
        usaY = variante in ("rendimento", "combinado")
        usaZ = variante in ("custo", "combinado")
        Y = {j: var(f"Y_{j}") for j in range(M)} if usaY else {}
        Z = {i: var(f"Z_{i}") for i in range(N)} if usaZ else {}

        # habilitação: X_ij ≤ H_ij
        for i in range(N):
            for j in range(M):
                s.Add(X[i, j] <= int(H[i, j]))

        # obrigações (eventualmente controladas por Y_j)
        for j in range(M):
            col = s.Sum([X[i, j] for i in range(N)])
            s.Add(col >= int(t[j]) * Y[j] if usaY else col >= int(t[j]))

        # máquinas inativas não têm operadores
        if usaY:
            for j in range(M):
                if agregado:
                    s.Add(s.Sum([X[i, j] for i in range(N)]) <= int(H[:, j].sum()) * Y[j])
                else:
                    for i in range(N):
                        s.Add(X[i, j] <= Y[j])

        # limitações (eventualmente controladas por Z_i)
        for i in range(N):
            lin = s.Sum([X[i, j] for j in range(M)])
            s.Add(lin <= int(m[i]) * Z[i] if usaZ else lin <= int(m[i]))

        rend = s.Sum([int(r[j]) * Y[j] for j in range(M)]) if usaY else 0
        cust = s.Sum([int(c[i]) * Z[i] for i in range(N)]) if usaZ else 0
        if variante == "base":
            C = custoX if custos is None else custos
            s.Minimize(s.Sum([float(C[i, j]) * X[i, j] for i in range(N) for j in range(M)]))
        elif variante == "rendimento":
            s.Maximize(rend)
        elif variante == "custo":
            s.Minimize(cust)
        else:
            s.Maximize(rend - cust)

        st = s.Solve()
        ok = st in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE)
        Xv = np.array([[X[i, j].solution_value() for j in range(M)] for i in range(N)]) if ok else None
        Yv = np.array([Y[j].solution_value() for j in range(M)]) if ok and usaY else None
        Zv = np.array([Z[i].solution_value() for i in range(N)]) if ok and usaZ else None
        vals = [v.solution_value() for v in s.variables()] if ok else []
        return dict(
            variante=variante, relaxado=relaxar, agregado=agregado,
            estado=ESTADO.get(st, str(st)),
            objetivo=s.Objective().Value() if ok else None,
            X=Xv, Y=Yv, Z=Zv,
            # distância máxima de uma variável ao inteiro mais próximo
            frac=max((abs(v - round(v)) for v in vals), default=None),
            n_var=s.NumVariables(), n_restr=s.NumConstraints(),
            ms=s.wall_time(),
        )
    return (resolve,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1 · As quatro variantes (IP)

    Na variante base o objetivo é um custo arbitrário por alocação, só para ter um critério;
    o que interessa é a admissibilidade. Se a base for **inadmissível** (procura maior do que
    a capacidade, ou habilitações insuficientes), a variante *rendimento* continua a dar
    uma resposta: o maior número de máquinas que é possível operar.
    """)
    return


@app.cell
def _(mo, resolve):
    VARIANTES = ["base", "rendimento", "custo", "combinado"]
    sol_ip = {v: resolve(v) for v in VARIANTES}

    def resumo(d):
        X = d["X"]
        return {
            "variante": d["variante"] + (" (LP)" if d["relaxado"] else "") + (" agregada" if d["agregado"] else ""),
            "estado": d["estado"],
            "objetivo": None if d["objetivo"] is None else round(d["objetivo"], 4),
            "máquinas ativas": None if X is None else int((X.sum(axis=0) > 1e-9).sum()),
            "operadores ativos": None if X is None else int((X.sum(axis=1) > 1e-9).sum()),
            "alocações Σ X": None if X is None else round(float(X.sum()), 4),
            "máx. fracionalidade": None if d["frac"] is None else round(d["frac"], 4),
            "variáveis": d["n_var"],
            "restrições": d["n_restr"],
            "tempo (ms)": d["ms"],
        }

    mo.ui.table([resumo(sol_ip[v]) for v in VARIANTES], selection=None)
    return VARIANTES, resumo, sol_ip


@app.cell
def _(VARIANTES, mo):
    ver_ui = mo.ui.dropdown(VARIANTES, value="combinado", label="mostrar alocação da variante")
    ver_ui
    return (ver_ui,)


@app.cell
def _(H, M, N, mo, sol_ip, ver_ui):
    def matriz(d):
        X, Y, Z = d["X"], d["Y"], d["Z"]
        if X is None:
            return mo.md(f"Variante **{d['variante']}**: {d['estado']}.")
        sim = lambda i, j: "**1**" if X[i, j] > 0.5 else ("·" if H[i, j] else " ")
        cab = "| | " + " | ".join(
            f"M{j}" + ("" if Y is None else (" ✓" if Y[j] > 0.5 else " ✗")) for j in range(M)
        ) + " | Σ |" + ("" if Z is None else " Z |")
        sep = "|---" * (M + 2 + (Z is not None)) + "|"
        linhas = [
            f"| op {i} | " + " | ".join(sim(i, j) for j in range(M)) + f" | {int(round(X[i].sum()))} |"
            + ("" if Z is None else f" {int(round(Z[i]))} |")
            for i in range(N)
        ]
        tot = "| **Σ** | " + " | ".join(str(int(round(X[:, j].sum()))) for j in range(M)) + " | |" + ("" if Z is None else " |")
        return mo.md(
            f"Variante **{d['variante']}**, objetivo = {d['objetivo']:.0f}. "
            "**1** = alocado, · = habilitado mas não alocado, vazio = não habilitado; "
            "✓/✗ = máquina ativa/inativa ($Y_j$).\n\n"
            + "\n".join([cab, sep, *linhas, tot])
        )

    matriz(sol_ip[ver_ui.value])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2 · Relaxação LP e unimodularidade total

    Na variante **base** a matriz de restrições é a de um grafo bipartido (operadores × máquinas)
    mais limites simples, e é **totalmente unimodular**: todos os vértices do poliedro LP são
    inteiros. Por isso o `GLOP` (simplex, que devolve sempre um vértice) dá uma solução inteira
    sem qualquer arredondamento, e o «modelo estocástico» do texto não é necessário.

    Quando se acrescentam as flags $Y_j$ (com coeficiente $\mathbf{t}_j$) ou $Z_i$ (com coeficiente $\mathbf{m}_i$),
    a unimodularidade perde-se: a relaxação LP **pode** ter vértices fracionários e um ótimo
    estritamente melhor do que o do IP (o *gap* de integralidade). Com os valores por omissão
    isto vê-se na variante *custo*: o LP «ativa» frações de operadores ($Z_i=\sum_j X_{i,j}/\mathbf{m}_i$).
    """)
    return


@app.cell
def _(VARIANTES, mo, resolve, resumo, sol_ip):
    sol_lp = {v: resolve(v, relaxar=True) for v in VARIANTES}
    _linhas = []
    for _v in VARIANTES:
        _linhas.append(resumo(sol_ip[_v]))
        _linhas.append(resumo(sol_lp[_v]))
    mo.ui.table(_linhas, selection=None)
    return (sol_lp,)


@app.cell
def _(M, N, mo, np, resolve, rng):
    # repete a variante base com 200 vetores de custos aleatórios: o LP é sempre inteiro?
    _fr = []
    for _ in range(200):
        _d = resolve("base", relaxar=True, custos=rng.normal(size=(N, M)))
        if _d["frac"] is not None:
            _fr.append(_d["frac"])
    mo.md(
        f"**Teste de unimodularidade**: em {len(_fr)} LPs da variante base com custos aleatórios "
        f"(incluindo negativos), a maior fracionalidade observada foi **{max(_fr, default=float('nan')):.2e}**."
        if _fr else "A variante base é inadmissível com estes dados; altere a semente ou a densidade."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3 · Formulação desagregada vs. agregada

    A restrição «máquina inativa não tem operadores» pode escrever-se de duas formas equivalentes nos inteiros:

    - desagregada: $X_{i,j}\le Y_j$ para cada $i$ ($N\cdot M$ restrições);
    - agregada: $\sum_i X_{i,j}\le \big(\sum_i \mathbf{H}_{i,j}\big)\,Y_j$ ($M$ restrições).

    O IP tem o mesmo ótimo, mas a relaxação LP da forma agregada é mais fraca (dá um limite superior
    pior), o que se traduz em mais nós de *branch-and-bound* em instâncias grandes.
    Isto só se vê quando a variante *rendimento* não consegue operar todas as máquinas
    (a base é inadmissível). Experimente, por exemplo, densidade 0.5, máx. de $t_j$ = 3,
    máx. de $m_i$ = 2 e semente 7: o IP opera 4 máquinas, o LP desagregado dá 4.5 e o agregado 5.
    """)
    return


@app.cell
def _(mo, resolve, resumo):
    mo.ui.table(
        [
            resumo(resolve("rendimento")),
            resumo(resolve("rendimento", relaxar=True)),
            resumo(resolve("rendimento", agregado=True)),
            resumo(resolve("rendimento", relaxar=True, agregado=True)),
        ],
        selection=None,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4 · Alocação «estocástica»: marginais não chegam

    O texto propõe interpretar uma solução fracionária $P_{i,j}\in[0,1]$ como probabilidades e
    gerar cada $X_{i,j}$ aleatoriamente. Para ver o problema, construímos $P$ como a **média de
    $K$ alocações admissíveis** (cada uma obtida da variante base com custos aleatórios).
    Assim $P$ é exatamente a matriz de marginais de uma distribuição sobre alocações admissíveis.

    Comparamos duas formas de amostrar:

    - **independente**: cada $X_{i,j}\sim\text{Bernoulli}(P_{i,j})$, separadamente (o que o texto sugere);
    - **mistura**: escolher uma das $K$ alocações ao acaso (mesmas marginais, mas com a correlação certa).
    """)
    return


@app.cell
def _(mo):
    K_ui = mo.ui.slider(2, 50, value=10, label="K (alocações na mistura)")
    S_ui = mo.ui.slider(100, 5000, step=100, value=1000, label="amostras")
    mo.hstack([K_ui, S_ui])
    return K_ui, S_ui


@app.cell
def _(H, K_ui, M, N, S_ui, m, mo, np, resolve, rng, t):
    def admissivel(X):
        return bool((X.sum(axis=0) >= t).all() and (X.sum(axis=1) <= m).all() and (X <= H).all())

    _sols = []
    for _ in range(K_ui.value):
        _d = resolve("base", relaxar=True, custos=rng.normal(size=(N, M)))
        if _d["X"] is None:
            break
        _sols.append(np.rint(_d["X"]).astype(int))  # já é inteira (secção 2)

    if not _sols:
        _out = mo.md("A variante base é inadmissível com estes dados.")
    else:
        P = np.mean(_sols, axis=0)
        _S = S_ui.value
        _indep = [admissivel((rng.random((N, M)) < P).astype(int)) for _ in range(_S)]
        _mist = [admissivel(_sols[rng.integers(len(_sols))]) for _ in range(_S)]
        _viol_t = np.mean([((rng.random((N, M)) < P).sum(axis=0) < t).any() for _ in range(_S)])
        _viol_m = np.mean([((rng.random((N, M)) < P).sum(axis=1) > m).any() for _ in range(_S)])
        _out = mo.md(
            f"$P$ tem {int(((P > 0) & (P < 1)).sum())} entradas estritamente fracionárias "
            f"({len({s.tobytes() for s in _sols})} alocações distintas entre as {len(_sols)}).\n\n"
            "| amostragem | amostras admissíveis |\n|---|---|\n"
            f"| independente | **{100 * np.mean(_indep):.1f} %** |\n"
            f"| mistura | **{100 * np.mean(_mist):.1f} %** |\n\n"
            f"Na amostragem independente, {100 * _viol_t:.1f} % das amostras violam alguma obrigação "
            f"($\\sum_i X_{{i,j}}\\ge \\mathbf{{t}}_j$) e {100 * _viol_m:.1f} % violam alguma limitação "
            f"($\\sum_j X_{{i,j}}\\le \\mathbf{{m}}_i$). A habilitação nunca é violada, porque $P_{{i,j}}=0$ onde $\\mathbf{{H}}_{{i,j}}=0$."
        )
    _out
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5 · Pesos no objetivo combinado

    Com $r_j=c_i=1$, o objetivo $\sum_j Y_j-\sum_i Z_i$ compara máquinas com operadores
    na mesma escala. Uma máquina que exige $\mathbf{t}_j\ge 2$ operadores que não podem ser partilhados
    não compensa, e a solução nula ($X=Y=Z=0$, objetivo $0$) pode ser ótima.
    Para ver isto, ponha $\mathbf{m}_i=1$ (máx. de $m_i$ = 1), máx. de $t_j$ ≥ 2 e pesos unitários, e
    escolha a variante *combinado* acima. Com pesos aleatórios ($r_j\in 1..5$, $c_i\in 1..3$) a solução muda.
    """)
    return


@app.cell
def _(mo, sol_ip):
    _d = sol_ip["combinado"]
    mo.md(
        f"Variante combinada com os dados atuais: objetivo **{_d['objetivo']:.0f}**, "
        f"{int(_d['Y'].round().sum())} máquinas ativas, {int(_d['Z'].round().sum())} operadores ativos."
        if _d["X"] is not None else f"Variante combinada: {_d['estado']}."
    )
    return


if __name__ == "__main__":
    app.run()
