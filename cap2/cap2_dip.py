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
    # Cap. 2 · Programação Inteira Disjuntiva (DIP)

    Exemplo da secção «Programação Inteira Disjuntiva» do *Capítulo 2: Programação com Restrições*.

    Uma disjunção $\eta \equiv \varphi^{(1)} \vee \cdots \vee \varphi^{(m)}$, com
    $\varphi^{(k)} \equiv C^{(k)} x \le c_0^{(k)}$ e $x \in \mathbb{N}^n$, é convertida no
    problema conjuntivo (formulação de Balas)

    $$
    \mathcal{M} \;\equiv\; \Big(\textstyle\sum_{k} y^{(k)} = 1\Big) \;\wedge\;
    \bigwedge_{k} \big(C^{(k)} x^{(k)} \le c_0^{(k)}\, y^{(k)}\big) \;\wedge\; \big(x^{(k)}, y^{(k)} \ge 0\big),
    \qquad x = \textstyle\sum_k x^{(k)} .
    $$

    A construção é correta quando cada $\varphi^{(k)}$ é **fracamente convexa**
    (alguma linha de $C^{(k)}$ tem todos os coeficientes positivos): com $y^{(k)} = 0$
    essa linha força $x^{(k)} = 0^n$.

    O notebook enumera (com CP-SAT) os valores de $x$ admitidos por $\mathcal{M}$ numa caixa
    $[0, B]^n$ e compara-os com os modelos de $\eta$ calculados diretamente.
    """)
    return


@app.cell
def _(mo):
    exemplo_ui = mo.ui.dropdown(
        ["1D: (x ≤ 2) ∨ (5 ≤ x ≤ 7)", "2D: (x₁+x₂ ≤ 3) ∨ (x₁+x₂ ≤ 10 ∧ x₁ ≥ 4 ∧ x₂ ≥ 4)"],
        value="1D: (x ≤ 2) ∨ (5 ≤ x ≤ 7)", label="exemplo")
    retirar_ui = mo.ui.checkbox(
        value=False,
        label="retirar do 2.º disjunto a linha positiva (deixa de ser fracamente convexo)")
    B_ui = mo.ui.slider(5, 15, value=10, label="caixa de enumeração B")
    mo.vstack([exemplo_ui, retirar_ui, B_ui])
    return B_ui, exemplo_ui, retirar_ui


@app.cell
def _(exemplo_ui, np, retirar_ui):
    # cada disjunto é um par (C, c0) que representa C x <= c0
    if exemplo_ui.value.startswith("1D"):
        n = 1
        disjuntos = [
            (np.array([[1]]), np.array([2])),              # x <= 2
            (np.array([[1], [-1]]), np.array([7, -5])),    # x <= 7  ∧  -x <= -5
        ]
    else:
        n = 2
        disjuntos = [
            (np.array([[1, 1]]), np.array([3])),                           # x1+x2 <= 3
            (np.array([[1, 1], [-1, 0], [0, -1]]), np.array([10, -4, -4])),  # x1+x2 <= 10, x1 >= 4, x2 >= 4
        ]
    if retirar_ui.value:
        C2, c2 = disjuntos[1]
        disjuntos[1] = (C2[1:], c2[1:])       # a 1.ª linha é a única com coeficientes positivos

    def fracamente_convexo(C):
        return any(np.all(linha > 0) for linha in C)

    convexidade = [fracamente_convexo(C) for C, _ in disjuntos]
    return convexidade, disjuntos, n


@app.cell
def _(B_ui, cp_model, disjuntos, itertools, n):
    B = B_ui.value

    def modelos_eta():
        """Modelos de η na caixa [0, B]^n, por enumeração direta."""
        return {
            x for x in itertools.product(range(B + 1), repeat=n)
            if any(all(sum(C[i, j] * x[j] for j in range(n)) <= c0[i] for i in range(len(c0)))
                   for C, c0 in disjuntos)
        }

    def modelos_M():
        """Valores x = Σ_k x^(k) admitidos pela formulação de Balas, com x na caixa [0, B]^n."""
        mdl = cp_model.CpModel()
        m = len(disjuntos)
        xs = [[mdl.NewIntVar(0, B, f"x{k}_{j}") for j in range(n)] for k in range(m)]
        ys = [mdl.NewBoolVar(f"y{k}") for k in range(m)]
        x = [mdl.NewIntVar(0, B, f"x_{j}") for j in range(n)]
        mdl.Add(sum(ys) == 1)                                   # μ
        for k, (C, c0) in enumerate(disjuntos):                 # φ'_k
            for i in range(len(c0)):
                mdl.Add(sum(int(C[i, j]) * xs[k][j] for j in range(n)) <= int(c0[i]) * ys[k])
        for j in range(n):
            mdl.Add(x[j] == sum(xs[k][j] for k in range(m)))

        encontrados = set()

        class Recolhe(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):
                encontrados.add(tuple(self.Value(v) for v in x))

        s = cp_model.CpSolver()
        s.parameters.enumerate_all_solutions = True
        s.Solve(mdl, Recolhe())
        return encontrados

    eta = modelos_eta()
    M = modelos_M()
    return B, M, eta


@app.cell
def _(B, M, convexidade, disjuntos, eta, mo, n):
    def mostra(S):
        S = sorted(S)
        if len(S) > 40:
            return f"{len(S)} pontos, por exemplo {S[:12]} …"
        return str(S)

    linhas_disj = "\n".join(
        f"| $\\varphi^{{({k + 1})}}$ | {C.tolist()} | {c0.tolist()} | {'sim' if fc else '**não**'} |"
        for k, ((C, c0), fc) in enumerate(zip(disjuntos, convexidade)))
    iguais = eta == M
    mo.md(rf"""
    | disjunto | $C^{{(k)}}$ | $c_0^{{(k)}}$ | fracamente convexo |
    |---|---|---|---|
    {linhas_disj}

    **Modelos de $\eta$** em $[0,{B}]^{n}$: {mostra(eta)}

    **Valores de $x$ admitidos por $\mathcal{{M}}$:** {mostra(M)}

    **Coincidem?** {"sim ✓" if iguais else "**não** ✗"}
    {"" if iguais else f"— pontos a mais em $\\mathcal{{M}}$: {mostra(M - eta)}"}
    """)
    return


if __name__ == "__main__":
    app.run()
