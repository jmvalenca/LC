import marimo

__generated_with = "0.26.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from ortools.sat.python import cp_model
    return cp_model, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Cap. 2 · Problema do Caixeiro Viajante (TSP)

    Exemplo «Problema do Caixeiro Viajante» do *Capítulo 2: Programação com Restrições*.

    Com $H_j \in \{0,1\}$ ($H_j = 1$ sse o ramo $j$ pertence ao ciclo) e
    $\delta(S)=\{(u,v)\in\mathcal{E} \mid u\in S,\ v\notin S\}$:

    $$
    \begin{aligned}
    \min_{H}\quad & \textstyle\sum_{j\in\mathcal{E}} c_j\,H_j\\
    \text{s.a.}\quad & \textstyle\sum_{j\in\delta(\{a\})} H_j = 2 && \forall\, a\in\mathcal{V}\\
    & \textstyle\sum_{j\in\delta(S)} H_j \ge 2 && \forall\, \emptyset\ne S\subsetneq\mathcal{V}\\
    & H\in\{0,1\}^{m}
    \end{aligned}
    $$

    Há $2^n-2$ restrições de corte, por isso não se escrevem todas.
    Usa-se **geração preguiçosa de restrições**:

    1. resolve-se o modelo só com as restrições de grau (CP-SAT);
    2. se a solução tem vários subciclos, cada subciclo $S$ dá uma restrição de corte violada
       ($\sum_{j\in\delta(S)} H_j = 0$), que se acrescenta ao modelo;
    3. repete-se até a solução ser um único ciclo — que é então um ciclo Hamiltoniano ótimo.

    Para verificar, o mesmo problema é resolvido com a restrição global `AddCircuit` do CP-SAT.
    """)
    return


@app.cell
def _(mo):
    grafo_ui = mo.ui.dropdown(
        ["exemplo do capítulo", "euclidiano aleatório"],
        value="exemplo do capítulo", label="grafo")
    custos_ui = mo.ui.dropdown(
        ["unitários", "aleatórios em {1,…,9}"],
        value="unitários", label="custos (exemplo)")
    n_ui = mo.ui.slider(4, 40, value=15, label="n (nodos, aleatório)")
    seed_ui = mo.ui.number(0, 10_000, value=1, label="semente")
    mo.vstack([
        mo.hstack([grafo_ui, custos_ui]),
        mo.hstack([n_ui, seed_ui]),
    ])
    return custos_ui, grafo_ui, n_ui, seed_ui


@app.cell
def _(custos_ui, grafo_ui, n_ui, np, seed_ui):
    rng = np.random.default_rng(seed_ui.value)
    if grafo_ui.value == "exemplo do capítulo":
        nomes = list("abcdef")
        pos = np.array([(0, 2), (1, 2), (2, 2), (1, 1), (0, 0), (2, 0)], float)
        a, b, c, d, e, f = range(6)
        # ramos 0..7 como na figura do capítulo
        E = [(a, b), (b, c), (c, f), (e, f), (a, e), (a, d), (b, d), (b, f)]
        if custos_ui.value == "unitários":
            custo = [1] * len(E)
        else:
            custo = [int(x) for x in rng.integers(1, 10, len(E))]
    else:
        n_ = n_ui.value
        nomes = [str(k) for k in range(n_)]
        pos = rng.random((n_, 2)) * 100
        # grafo completo; custo = distância euclidiana arredondada
        E = [(u, v) for u in range(n_) for v in range(u + 1, n_)]
        custo = [int(round(np.linalg.norm(pos[u] - pos[v]))) for u, v in E]
    n = len(nomes)
    m = len(E)

    def delta(S):
        """Índices dos ramos com exatamente um extremo em S."""
        return [j for j, (u, v) in enumerate(E) if (u in S) != (v in S)]
    return E, custo, delta, m, n, nomes, pos


@app.cell
def _(E, cp_model, custo, delta, m, n):
    def componentes(ramos):
        """Componentes ligadas do subgrafo formado pelos ramos escolhidos."""
        adj = {v: [] for v in range(n)}
        for j in ramos:
            u, v = E[j]
            adj[u].append(v)
            adj[v].append(u)
        vistos, comps = set(), []
        for s in range(n):
            if s in vistos:
                continue
            pilha, comp = [s], set()
            while pilha:
                x = pilha.pop()
                if x in comp:
                    continue
                comp.add(x)
                pilha.extend(adj[x])
            vistos |= comp
            comps.append(comp)
        return comps

    def tsp_preguicoso(max_iter=200):
        cortes = []          # conjuntos S cujas restrições de corte já foram acrescentadas
        historico = []       # (iteração, custo, nº de subciclos)
        for it in range(1, max_iter + 1):
            mdl = cp_model.CpModel()
            H = [mdl.NewBoolVar(f"H{j}") for j in range(m)]
            for a in range(n):
                mdl.Add(sum(H[j] for j in delta({a})) == 2)
            for S in cortes:
                mdl.Add(sum(H[j] for j in delta(S)) >= 2)
            mdl.Minimize(sum(c * h for c, h in zip(custo, H)))
            s = cp_model.CpSolver()
            s.parameters.max_time_in_seconds = 30
            st = s.Solve(mdl)
            if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                return None, None, historico, s.StatusName(st)
            ramos = [j for j in range(m) if s.Value(H[j])]
            comps = componentes(ramos)
            historico.append((it, int(s.ObjectiveValue()), len(comps)))
            if len(comps) == 1:
                return ramos, int(s.ObjectiveValue()), historico, s.StatusName(st)
            cortes.extend(comps)
        return None, None, historico, "limite de iterações"

    def tsp_circuit():
        """Verificação com a restrição global AddCircuit (arcos nos dois sentidos)."""
        mdl = cp_model.CpModel()
        H = [mdl.NewBoolVar(f"H{j}") for j in range(m)]
        arcos = []
        for j, (u, v) in enumerate(E):
            uv, vu = mdl.NewBoolVar(""), mdl.NewBoolVar("")
            arcos += [(u, v, uv), (v, u, vu)]
            mdl.Add(uv + vu == H[j])
        mdl.AddCircuit(arcos)
        mdl.Minimize(sum(c * h for c, h in zip(custo, H)))
        s = cp_model.CpSolver()
        s.parameters.max_time_in_seconds = 30
        st = s.Solve(mdl)
        if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            return None, s.StatusName(st)
        return int(s.ObjectiveValue()), s.StatusName(st)

    ciclo, custo_ciclo, historico, status = tsp_preguicoso()
    custo_circ, status_circ = tsp_circuit()
    return ciclo, custo_circ, custo_ciclo, historico, status, status_circ


@app.cell
def _(E, ciclo, custo, n, nomes, plt, pos):
    fig, ax = plt.subplots(figsize=(5, 4))
    for j, (u, v) in enumerate(E):
        no_ciclo = ciclo is not None and j in ciclo
        ax.plot(*zip(pos[u], pos[v]),
                color="C3" if no_ciclo else "0.8",
                lw=2.5 if no_ciclo else 0.8, zorder=1)
        if n <= 8:
            mx, my = (pos[u] + pos[v]) / 2
            ax.annotate(f"{j} (c={custo[j]})", (mx, my), fontsize=8,
                        ha="center", va="center",
                        bbox=dict(fc="white", ec="none", pad=0.5))
    ax.scatter(pos[:, 0], pos[:, 1], s=300 if n <= 8 else 40,
               c="white", edgecolors="k", zorder=2)
    if n <= 8:
        for k, nome in enumerate(nomes):
            ax.annotate(nome, pos[k], ha="center", va="center", zorder=3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("ciclo Hamiltoniano ótimo" if ciclo is not None else "sem ciclo Hamiltoniano")
    fig
    return


@app.cell
def _(ciclo, custo_circ, custo_ciclo, historico, m, mo, n, status, status_circ):
    linhas = "\n".join(f"| {it} | {c} | {k} |" for it, c, k in historico)
    resultado = (
        f"ramos $H = {sorted(ciclo)}$, custo **{custo_ciclo}**"
        if ciclo is not None else f"não há ciclo Hamiltoniano ({status})"
    )
    mo.md(rf"""
    **Grafo:** {n} nodos, {m} ramos.
    Todas as restrições de corte seriam $2^{{{n}}}-2 = {2**n - 2}$.

    **Geração preguiçosa:** {resultado}, em {len(historico)} iteração(ões),
    com {sum(k for _, _, k in historico[:-1]) if ciclo is not None else "—"} restrições de corte acrescentadas.

    | iteração | custo | subciclos |
    |---|---|---|
    {linhas}

    **Verificação (`AddCircuit`, {status_circ}):** custo {custo_circ}.
    """)
    return


if __name__ == "__main__":
    app.run()
