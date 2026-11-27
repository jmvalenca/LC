import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pysmt.shortcuts import (Symbol, Real, Int, And, Or, Not, Implies, Equals,
                                 GE, GT, LE, LT, Ite, Plus, Solver)
    from pysmt.typing import INT, REAL
    from fractions import Fraction
    return (Fraction, And, Equals, GE, GT, INT, Implies, Int, Ite, LE, LT, Not, Or, Plus,
            REAL, Real, Solver, Symbol, mo)


@app.cell
def _(mo):
    mo.md(r"""
    # TP4 · Problema 1 — Pâncreas artificial (resolução de referência)

    Autómato híbrido com modos **BASAL**, **BOLUS**, **SUSP** (suspensão) e **ALARME**.
    Variáveis contínuas: glicose $G$ (mg/dL), insulina ativa $I$ (U) e relógio $t$ (min).
    Entrada não determinista: taxa de entrada de glicose $r \in [0, r_{max}]$ (refeições, fígado).

    Fluxos (aproximação de Euler com período de amostragem $\Delta$):

    $$G' = G + \Delta\,(r - a\,I) \qquad I' = I + \Delta\,(u - b\,I) \qquad t' = t + \Delta$$

    com $u = u_b$ em BASAL, $u = u_B$ em BOLUS e $u = 0$ em SUSP.
    O controlador usa a **glicose prevista** $G_p = G - \tfrac{a}{b} I$ (o valor para onde $G$ tende se a infusão parar e $r=0$).
    """)
    return


@app.cell
def _():
    # Parâmetros do enunciado (tempo em minutos)
    PARAMS = dict(
        D=10,      # período de amostragem Δ
        a=0.02,    # sensibilidade à insulina (mg/dL por U por min)
        b=0.05,    # taxa de eliminação da insulina (1/min)
        ub=1.0,    # infusão basal (U/min)
        uB=3.0,    # infusão em bolus (U/min)
        rmax=1.2,  # taxa máxima de entrada de glicose (mg/dL/min)
        Ls=90,     # limiar de suspensão sobre a glicose prevista
        Lh=180,    # limiar de hiperglicemia (entra em bolus)
        Lb=140,    # fim do bolus por glicose
        Lr=100,    # retoma da basal após suspensão
        TB=30,     # duração máxima do bolus
        TS=30,     # duração mínima da suspensão
    )
    BASAL, BOLUS, SUSP, ALARME = 0, 1, 2, 3
    NOMES = {BASAL: "BASAL", BOLUS: "BOLUS", SUSP: "SUSP", ALARME: "ALARME"}
    return ALARME, BASAL, BOLUS, NOMES, PARAMS, SUSP


@app.cell
def _(ALARME, And, BASAL, BOLUS, Equals, Fraction, GE, GT, INT, Int, Ite, LE,
      LT, Not, Or, REAL, Real, SUSP, Symbol):
    def cria_modelo(P):
        """Devolve (declare, init, trans) do FOTS para os parâmetros P."""
        R = lambda x: Real(Fraction(str(x)))  # valores exatos
        kappa = Fraction(str(P["a"])) / Fraction(str(P["b"]))

        def declare(i):
            return {
                "m": Symbol(f"m{i}", INT),
                "G": Symbol(f"G{i}", REAL),
                "I": Symbol(f"I{i}", REAL),
                "t": Symbol(f"t{i}", REAL),
                "r": Symbol(f"r{i}", REAL),   # entrada escolhida no passo i
            }

        def init(s):
            return And(Equals(s["m"], Int(BASAL)), Equals(s["t"], R(0)),
                       GE(s["G"], R(100)), LE(s["G"], R(180)),
                       GE(s["I"], R(0)), LE(s["I"], R(20)))

        def Gp(s):
            return s["G"] - R(kappa) * s["I"]

        def guardas(s):
            m, G, t = s["m"], s["G"], s["t"]
            em = lambda q: Equals(m, Int(q))
            alarme = Or(LT(G, R(50)), GT(G, R(300)))
            baixo = LT(Gp(s), R(P["Ls"]))
            return {
                "→ALARME": And(Not(em(ALARME)), alarme),
                "BASAL→SUSP": And(em(BASAL), Not(alarme), baixo),
                "BASAL→BOLUS": And(em(BASAL), Not(alarme), Not(baixo), GT(G, R(P["Lh"]))),
                "BOLUS→SUSP": And(em(BOLUS), Not(alarme), baixo),
                "BOLUS→BASAL": And(em(BOLUS), Not(alarme), Not(baixo),
                                   Or(GE(t, R(P["TB"])), LE(G, R(P["Lb"])))),
                "SUSP→BASAL": And(em(SUSP), Not(alarme), GT(G, R(P["Lr"])), GE(t, R(P["TS"]))),
            }

        def mantem(s, p, vs):
            return And([Equals(p[v], s[v]) for v in vs])

        def discreta(s, p):
            g = guardas(s)
            vai = lambda q: Equals(p["m"], Int(q))
            return Or(
                And(g["→ALARME"], vai(ALARME), mantem(s, p, "GIt")),
                And(g["BASAL→SUSP"], vai(SUSP), Equals(p["t"], R(0)), mantem(s, p, "GI")),
                And(g["BASAL→BOLUS"], vai(BOLUS), Equals(p["t"], R(0)), mantem(s, p, "GI")),
                And(g["BOLUS→SUSP"], vai(SUSP), Equals(p["t"], R(0)), mantem(s, p, "GI")),
                And(g["BOLUS→BASAL"], vai(BASAL), mantem(s, p, "GIt")),
                And(g["SUSP→BASAL"], vai(BASAL), mantem(s, p, "GIt")),
            )

        def temporizada(s, p):
            g = guardas(s)
            D = R(P["D"])
            m = s["m"]
            u = Ite(Equals(m, Int(BASAL)), R(P["ub"]),
                    Ite(Equals(m, Int(BOLUS)), R(P["uB"]), R(0)))
            urgente = Or(list(g.values()))   # transições discretas são urgentes
            fluxo = And(Not(Equals(m, Int(ALARME))), Not(urgente),
                        GE(s["r"], R(0)), LE(s["r"], R(P["rmax"])),
                        Equals(p["G"], s["G"] + D * (s["r"] - R(P["a"]) * s["I"])),
                        Equals(p["I"], s["I"] + D * (u - R(P["b"]) * s["I"])),
                        Equals(p["t"], s["t"] + D),
                        Equals(p["m"], m))
            parado = And(Equals(m, Int(ALARME)), mantem(s, p, "mGIt"))
            return Or(fluxo, parado)

        def trans(s, p):
            return Or(discreta(s, p), temporizada(s, p))

        return declare, init, trans, Gp

    return (cria_modelo,)


@app.cell
def _(NOMES, Not, Solver):
    def mostra(sol, tr, k):
        linhas = []
        for i in range(k + 1):
            v = {x: sol.get_py_value(tr[i][x]) for x in "mGItr"}
            linhas.append(f"{i:3d} {NOMES[v['m']]:7s} G={float(v['G']):8.3f} "
                          f"I={float(v['I']):6.2f} t={float(v['t']):5.1f} r={float(v['r']):4.2f}")
        return "\n".join(linhas)

    def bmc(modelo, prop, K, init=None):
        """BMC incremental: procura k <= K tal que um estado do traço viola prop."""
        declare, init0, trans, _ = modelo
        init = init or init0
        with Solver(name="z3") as sol:
            tr = [declare(0)]
            sol.add_assertion(init(tr[0]))
            for k in range(K + 1):
                sol.push()
                sol.add_assertion(Not(prop(tr[k])))
                if sol.solve():
                    return k, mostra(sol, tr, k)
                sol.pop()
                tr.append(declare(k + 1))
                sol.add_assertion(trans(tr[k], tr[k + 1]))
        return None, f"Propriedade válida em todos os traços com até {K} transições."

    def kinducao(modelo, prop, k):
        """k-indução: base (BMC até k-1) e passo indutivo."""
        declare, init, trans, _ = modelo
        base, _txt = bmc(modelo, prop, k - 1)
        if base is not None:
            return f"falha na base (contraexemplo com {base} passos)"
        with Solver(name="z3") as sol:
            tr = [declare(i) for i in range(k + 1)]
            for i in range(k):
                sol.add_assertion(prop(tr[i]))
                sol.add_assertion(trans(tr[i], tr[i + 1]))
            sol.add_assertion(Not(prop(tr[k])))
            if sol.solve():
                return "passo indutivo falha (há um contraexemplo-de-indução)"
            return f"provado por {k}-indução"

    return bmc, kinducao


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Segurança limitada: nunca há hipoglicemia ($G \ge 70$)

    Estados iniciais: BASAL, $t=0$, $G\in[100,180]$, $I\in[0,20]$.
O custo do BMC cresce exponencialmente com $k$ (20 transições ≈ 10 s; 30 transições já demora vários minutos), daí a necessidade da prova indutiva.
    """)
    return


@app.cell
def _(GE, PARAMS, Real, bmc, cria_modelo):
    modelo = cria_modelo(PARAMS)
    hipo = lambda s: GE(s["G"], Real(70))
    _k, _txt = bmc(modelo, hipo, 20)
    print("BMC G>=70:", _txt)
    return hipo, modelo


@app.cell
def _(ALARME, Equals, Int, Not, bmc, modelo):
    sem_alarme = lambda s: Not(Equals(s["m"], Int(ALARME)))
    _k, _txt = bmc(modelo, sem_alarme, 20)
    print("BMC sem ALARME:", _txt)
    return (sem_alarme,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Prova não limitada

    A propriedade $G \ge 70$ **não é indutiva** por si só: há estados não alcançáveis
    (p.ex. $G=71$ com muita insulina ativa) que a violam num passo.
    Reforçamos com um invariante sobre a glicose prevista $G_p = G - \frac{a}{b}I$, cuja dinâmica é
    $G_p' = G_p + \Delta\,(r - \tfrac{a}{b}u)$:

    - em SUSP ($u=0$) $G_p$ nunca desce;
    - em BASAL/BOLUS só há fluxo com $G_p \ge L_s$, e desce no máximo $\Delta\frac{a}{b}u_B = 12$ por passo.

    $$\mathit{Inv} \equiv m\neq\text{ALARME} \wedge 0\le I\le 60 \wedge 78 \le G_p \le 188 \wedge t\ge 0
    \wedge (m=\text{SUSP} \rightarrow G_p \le L_s + r_{max}\,t \wedge G_p \le 138)$$
    """)
    return


@app.cell
def _(hipo, kinducao, modelo, sem_alarme):
    for _k in (1, 2, 3, 4):
        print(f"G>=70, k={_k}:", kinducao(modelo, hipo, _k))
    print("sem ALARME, k=2:", kinducao(modelo, sem_alarme, 2))
    return


@app.cell
def _(And, BASAL, BOLUS, Equals, Fraction, GE, Implies, Int, LE, Not, Or,
      PARAMS, Real, SUSP, Solver, hipo, modelo, sem_alarme):
    def invariante(s, Gp):
        R = lambda x: Real(Fraction(str(x)))  # valores exatos
        return And(Or([Equals(s["m"], Int(q)) for q in (BASAL, BOLUS, SUSP)]),
                   GE(s["I"], R(0)), LE(s["I"], R(60)), GE(s["t"], R(0)),
                   GE(Gp(s), R(78)), LE(Gp(s), R(188)),
                   Implies(Equals(s["m"], Int(SUSP)),
                           And(LE(Gp(s), R(PARAMS["Ls"]) + R(PARAMS["rmax"]) * s["t"]),
                               LE(Gp(s), R(138)))))

    def valida(f):
        with Solver(name="z3") as sol:
            return not sol.is_sat(Not(f))

    _declare, _init, _trans, _Gp = modelo
    _s, _p = _declare(0), _declare(1)
    _inv = lambda x: invariante(x, _Gp)
    print("Init → Inv          :", valida(Implies(_init(_s), _inv(_s))))
    print("Inv ∧ T → Inv'      :", valida(Implies(And(_inv(_s), _trans(_s, _p)), _inv(_p))))
    print("Inv → G>=70         :", valida(Implies(_inv(_s), hipo(_s))))
    print("Inv → não ALARME    :", valida(Implies(_inv(_s), sem_alarme(_s))))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Vivacidade limitada: a hiperglicemia pós-refeição resolve-se

    Depois de uma refeição o doente está em BOLUS com $G=210$, $I=0$, $t=0$ e, a partir daí, $r \le 0.5$.
    Procuramos o traço **mais longo** em que $G > 180$ em todos os estados: aumentamos o comprimento $L$
    até o problema ficar insatisfazível. Se nenhum traço de comprimento $L$ mantém a hiperglicemia,
    então a glicose volta a $[70,180]$ em menos de $L$ transições.
    """)
    return


@app.cell
def _(And, BOLUS, Equals, Fraction, GT, Int, LE, PARAMS, Real, Solver,
      cria_modelo):
    def hiperglicemia_maxima(rpos=0.5, Lmax=60):
        declare, _i, trans, _g = cria_modelo(PARAMS)
        R = lambda x: Real(Fraction(str(x)))
        tr = [declare(0)]
        ultimo = None
        with Solver(name="z3") as sol:
            sol.add_assertion(And(Equals(tr[0]["m"], Int(BOLUS)), Equals(tr[0]["G"], R(210)),
                                  Equals(tr[0]["I"], R(0)), Equals(tr[0]["t"], R(0))))
            for L in range(1, Lmax + 1):
                tr.append(declare(L))
                sol.add_assertion(trans(tr[L - 1], tr[L]))
                sol.add_assertion(LE(tr[L - 1]["r"], R(rpos)))
                sol.add_assertion(GT(tr[L]["G"], R(PARAMS["Lh"])))
                if not sol.solve():
                    return L, ultimo
                fluxos = sum(1 for i in range(L)
                             if sol.get_py_value(tr[i + 1]["G"]) != sol.get_py_value(tr[i]["G"]))
                ultimo = (L, fluxos)
        return None, ultimo

    _L, (_Lw, _fl) = hiperglicemia_maxima()
    print(f"Nenhum traço mantém G>180 durante {_L} transições.")
    print(f"Pior caso: {_Lw} transições, das quais {_fl} temporizadas (~{_fl * PARAMS['D']} min).")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Contraexemplo: limiar de suspensão mal calibrado

    Com $L_s = 72$ a margem $L_s - \Delta\frac{a}{b}u_b$ deixa de cobrir a insulina já injetada.
    """)
    return


@app.cell
def _(PARAMS, bmc, cria_modelo, hipo):
    _P = dict(PARAMS, Ls=72)
    _k, _txt = bmc(cria_modelo(_P), hipo, 30)
    print(f"Ls=72 → violação em {_k} transições:\n{_txt}")
    return


if __name__ == "__main__":
    app.run()
