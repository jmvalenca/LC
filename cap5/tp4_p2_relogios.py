import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from fractions import Fraction
    from pysmt.shortcuts import (Symbol, Real, Int, Bool, And, Or, Not, Implies, Iff,
                                 Equals, GE, GT, LE, LT, Ite, Solver, ToReal)
    from pysmt.typing import INT, REAL, BOOL
    return (And, BOOL, Equals, Fraction, GE, GT, INT, Iff, Implies, Int, Ite,
            LE, LT, Not, Or, REAL, Real, Solver, Symbol, ToReal, mo)


@app.cell
def _(mo):
    mo.md(r"""
    # TP4 · Problema 2 — Sincronização de relógios numa rede de sensores (resolução de referência)

    Sistema = **Mestre** ∥ **Canal** ∥ **Sensor₁** ∥ **Sensor₂**, sincronizados por rótulos:

    | Rótulo | Participantes | Efeito |
    |---|---|---|
    | `tempo(τ)` | todos | o tempo avança τ>0 em todos os componentes |
    | `envia` | Mestre, Canal | $x = P$: $x:=0$; canal LIVRE→TRÂNSITO, $y:=0$ |
    | `entrega` | Canal, Sensor₁, Sensor₂ | $y\in[d_{min},d_{max}]$: canal →LIVRE; cada sensor **recebe** ou **perde** |

    Em vez dos relógios absolutos $c_i$ usamos o **desvio** $e_i = c_i - T$ face ao tempo real $T$
    (o mestre é a referência). Assim $\dot e_i \in [-\rho, \rho]$ e a receção faz
    $c_i := (T - y) + d_{min}$, isto é, $e_i := d_{min} - y$. As variáveis ficam limitadas, o que permite k-indução.
    """)
    return


@app.cell
def _():
    PARAMS = dict(
        P=10,        # período de sincronização
        rho=0.01,    # deriva máxima dos relógios (1%)
        dmin=0.1,    # atraso mínimo do canal
        dmax=0.5,    # atraso máximo do canal
        K=3,         # perdas consecutivas até DESSINC
    )
    TEMPO, ENVIA, ENTREGA = 0, 1, 2
    return ENTREGA, ENVIA, PARAMS, TEMPO


@app.cell
def _(And, BOOL, ENTREGA, ENVIA, Equals, Fraction, GE, GT, INT, Iff, Int,
      Ite, LE, Not, Or, REAL, Real, Symbol, TEMPO):
    def cria_sistema(P, sensores=(1, 2)):
        R = lambda v: Real(Fraction(str(v)))

        def declare(i):
            s = {"ev": Symbol(f"ev{i}", INT), "tau": Symbol(f"tau{i}", REAL),
                 "x": Symbol(f"x{i}", REAL), "tr": Symbol(f"tr{i}", BOOL),
                 "y": Symbol(f"y{i}", REAL), "w": Symbol(f"w{i}", REAL)}
            for j in sensores:
                s[f"e{j}"] = Symbol(f"e{j}_{i}", REAL)
                s[f"f{j}"] = Symbol(f"f{j}_{i}", INT)
                s[f"d{j}"] = Symbol(f"d{j}_{i}", BOOL)     # DESSINC?
                s[f"rec{j}"] = Symbol(f"rec{j}_{i}", BOOL)  # recebe na entrega do passo i?
            return s

        def init(s):
            c = [Equals(s["x"], R(0)), Not(s["tr"]), Equals(s["y"], R(0)), Equals(s["w"], R(0))]
            for j in sensores:
                c += [Equals(s[f"e{j}"], R(0)), Equals(s[f"f{j}"], Int(0)), Not(s[f"d{j}"])]
            return And(c)

        igual = lambda s, p, vs: And([Iff(p[v], s[v]) if v in ("tr",) or v[0] == "d"
                                      else Equals(p[v], s[v]) for v in vs])

        # ---- componentes: um dicionário rótulo → relação de transição local ----
        def mestre(s, p):
            return {
                TEMPO: And(Equals(p["x"], s["x"] + s["tau"]), LE(p["x"], R(P["P"]))),
                ENVIA: And(Equals(s["x"], R(P["P"])), Equals(p["x"], R(0))),
            }

        def canal(s, p):
            return {
                TEMPO: And(Iff(p["tr"], s["tr"]),
                           Ite(s["tr"], And(Equals(p["y"], s["y"] + s["tau"]), LE(p["y"], R(P["dmax"]))),
                               Equals(p["y"], s["y"]))),
                ENVIA: And(Not(s["tr"]), p["tr"], Equals(p["y"], R(0))),
                ENTREGA: And(s["tr"], GE(s["y"], R(P["dmin"])), Not(p["tr"]), Equals(p["y"], s["y"])),
            }

        def sensor(j):
            e, f, d, rec = f"e{j}", f"f{j}", f"d{j}", f"rec{j}"
            def rel(s, p):
                rt = R(P["rho"]) * s["tau"]
                recebe = And(Equals(p[e], R(P["dmin"]) - s["y"]), Equals(p[f], Int(0)), Not(p[d]))
                perde = And(Equals(p[e], s[e]), Equals(p[f], s[f] + 1),
                            Iff(p[d], GE(s[f] + 1, Int(P["K"]))))
                return {
                    TEMPO: And(GE(p[e], s[e] - rt), LE(p[e], s[e] + rt),
                               Equals(p[f], s[f]), Iff(p[d], s[d])),
                    ENTREGA: Ite(s[rec], recebe, perde),
                }
            return rel

        componentes = [(mestre, {"x"}), (canal, {"tr", "y"})] + \
                      [(sensor(j), {f"e{j}", f"f{j}", f"d{j}"}) for j in sensores]

        # ---- composição paralela com sincronização por rótulos ----
        def trans(s, p):
            ramos = []
            for ev in (TEMPO, ENVIA, ENTREGA):
                partes = [Equals(s["ev"], Int(ev))]
                for comp, vars_ in componentes:
                    rels = comp(s, p)
                    partes.append(rels[ev] if ev in rels else igual(s, p, sorted(vars_)))
                # relógio auxiliar w (tempo desde o início) avança só com o tempo
                partes.append(Equals(p["w"], s["w"] + s["tau"]) if ev == TEMPO else Equals(p["w"], s["w"]))
                if ev == TEMPO:
                    partes.append(GT(s["tau"], R(0)))
                ramos.append(And(partes))
            return Or(ramos)

        return declare, init, trans

    return (cria_sistema,)


@app.cell
def _(ENTREGA, ENVIA, Not, Solver, TEMPO):
    NOME_EV = {TEMPO: "tempo", ENVIA: "envia", ENTREGA: "entrega"}

    def mostra(sol, tr, k, sensores=(1, 2)):
        val = lambda s, v: sol.get_py_value(s[v])
        out = []
        for i in range(k + 1):
            s = tr[i]
            linha = f"{i:3d} x={float(val(s,'x')):6.2f} canal={'TRÂNSITO' if val(s,'tr') else 'LIVRE   '} " \
                    f"y={float(val(s,'y')):4.2f}"
            for j in sensores:
                linha += f" | e{j}={float(val(s,f'e{j}')):+.4f} f{j}={val(s,f'f{j}')}" \
                         f" {'DESS' if val(s,f'd{j}') else 'SINC'}"
            if i < k:
                ev = val(s, "ev")
                extra = f"(τ={float(val(s,'tau')):.2f})" if ev == TEMPO else ""
                if ev == ENTREGA:
                    extra = "(" + ",".join(("rec" if val(s, f"rec{j}") else "perde") for j in sensores) + ")"
                linha += f"   --{NOME_EV[ev]}{extra}-->"
            out.append(linha)
        return "\n".join(out)

    def bmc(sistema, prop, K, init=None, restr=None):
        declare, init0, trans = sistema
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
                if restr is not None:
                    sol.add_assertion(restr(tr[k]))
        return None, f"sem violações até {K} transições"

    return (bmc,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Segurança: desvio entre sensores sincronizados

    $\varphi_\delta \equiv (\neg d_1 \wedge \neg d_2) \rightarrow |e_1 - e_2| \le \delta$.
    O pior caso exige $K$ períodos de perdas, cerca de 4 transições por período,
    pelo que usamos BMC até 16 transições.
    """)
    return


@app.cell
def _(And, Fraction, Implies, LE, Not, Real):
    def skew_ok(delta):
        d = Real(Fraction(str(delta)))
        return lambda s: Implies(And(Not(s["d1"]), Not(s["d2"])),
                                 And(LE(s["e1"] - s["e2"], d), LE(s["e2"] - s["e1"], d)))
    return (skew_ok,)


@app.cell
def _(PARAMS, bmc, cria_sistema, skew_ok):
    sistema = cria_sistema(PARAMS)
    for _delta in (0.5, 1.0):
        _k, _txt = bmc(sistema, skew_ok(_delta), 16)
        print(f"δ={_delta}: " + (_txt if _k is None else f"violação em {_k} transições\n{_txt}"))
    return (sistema,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Menor δ observável (pesquisa binária com BMC)

    Para $k$ fixo, o maior desvio alcançável é o menor δ para o qual o BMC não encontra violação.
    Cada iteração é um BMC completo (a célula demora 2–3 minutos). Compara-se com o majorante analítico $(d_{max}-d_{min}) + 2\rho\,(KP + d_{max})$.
    """)
    return


@app.cell
def _(PARAMS, bmc, skew_ok, sistema):
    def menor_delta(k, lo=0.5, hi=1.1, eps=0.005):
        while hi - lo > eps:
            mid = round((lo + hi) / 2, 4)
            if bmc(sistema, skew_ok(mid), k)[0] is None:
                hi = mid
            else:
                lo = mid
        return hi

    _P = PARAMS
    _maj = (_P["dmax"] - _P["dmin"]) + 2 * _P["rho"] * (_P["K"] * _P["P"] + _P["dmax"])
    print(f"maior desvio observado (k=16): ≈ {menor_delta(16):.3f}")
    print(f"majorante analítico           : {_maj:.3f}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Prova não limitada por invariante indutivo

    Seja $A_i = f_i\,P + x + (P \text{ se o canal está em TRÂNSITO})$, um majorante do tempo desde a última
    receção do sensor $i$. O invariante é

    $$\mathit{Inv} \equiv 0\le x\le P \wedge (\text{TRÂNSITO} \rightarrow y = x \le d_{max}) \wedge
    \bigwedge_i \big( f_i \ge 0 \wedge (d_i \leftrightarrow f_i \ge K) \wedge d_{min}-d_{max}-\rho A_i \le e_i \le \rho A_i\big)$$

    Observação-chave: $x$ e $y$ são postos a 0 no mesmo `envia`, logo $y = x$ enquanto a mensagem está em trânsito.
    """)
    return


@app.cell
def _(And, Equals, Fraction, GE, Iff, Implies, Int, Ite, LE, Not, PARAMS,
      Real, Solver, ToReal, sistema, skew_ok):
    def invariante(s, P=PARAMS):
        R = lambda v: Real(Fraction(str(v)))
        c = [GE(s["x"], R(0)), LE(s["x"], R(P["P"])),
             Implies(s["tr"], And(Equals(s["y"], s["x"]), LE(s["y"], R(P["dmax"]))))]
        for j in (1, 2):
            A = ToReal(s[f"f{j}"]) * R(P["P"]) + s["x"] + Ite(s["tr"], R(P["P"]), R(0))
            c += [GE(s[f"f{j}"], Int(0)), Iff(s[f"d{j}"], GE(s[f"f{j}"], Int(P["K"]))),
                  GE(s[f"e{j}"], R(P["dmin"] - P["dmax"]) - R(P["rho"]) * A),
                  LE(s[f"e{j}"], R(P["rho"]) * A)]
        return And(c)

    def valida(f):
        with Solver(name="z3") as sol:
            return not sol.is_sat(Not(f))

    _declare, _init, _trans = sistema
    _s, _p = _declare(0), _declare(1)
    _P = PARAMS
    _maj = (_P["dmax"] - _P["dmin"]) + 2 * _P["rho"] * (_P["K"] * _P["P"] + _P["dmax"])
    print("Init → Inv         :", valida(Implies(_init(_s), invariante(_s))))
    print("Inv ∧ T → Inv'     :", valida(Implies(And(invariante(_s), _trans(_s, _p)), invariante(_p))))
    print(f"Inv → φ(δ={_maj:.3f}) :", valida(Implies(invariante(_s), skew_ok(_maj)(_s))))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Vivacidade limitada: recuperação de um sensor dessincronizado

    A partir de um estado qualquer que satisfaz *Inv* com o Sensor₁ em DESSINC, e com o canal a deixar de perder
    mensagens para esse sensor, o Sensor₁ volta a SINC em no máximo $B = P + d_{max}$ unidades de tempo.
    Formulamos como segurança com o relógio auxiliar $w$ (posto a 0 no início):
    $\psi_B \equiv d_1 \rightarrow w \le B$.
    """)
    return


@app.cell
def _(And, Equals, Fraction, GE, Iff, Implies, Int, LE, Not, Or, PARAMS,
      Real, Solver, bmc, sistema):
    _R = lambda v: Real(Fraction(str(v)))
    _declare, _init, _trans = sistema
    _P = PARAMS

    def ini_dess(s):
        return And(s["d1"], Equals(s["f1"], Int(_P["K"])), Equals(s["w"], _R(0)),
                   GE(s["f2"], Int(0)), Iff(s["d2"], GE(s["f2"], Int(_P["K"]))),
                   GE(s["x"], _R(0)), LE(s["x"], _R(_P["P"])),
                   Implies(s["tr"], And(Equals(s["y"], s["x"]), LE(s["y"], _R(_P["dmax"])))))

    sem_perdas = lambda s: s["rec1"]
    psi = lambda B: (lambda s: Implies(s["d1"], LE(s["w"], _R(B))))

    for _B in (_P["P"], _P["P"] + _P["dmax"]):
        _k, _txt = bmc(sistema, psi(_B), 12, init=ini_dess, restr=sem_perdas)
        print(f"B={_B}: " + (_txt if _k is None else f"violação em {_k} transições\n{_txt}"))

    # prova indutiva de ψ_B com B = P + dmax
    def inv_rec(s, B=_P["P"] + _P["dmax"]):
        resto = Or(And(Not(s["tr"]), LE(s["w"] + (_R(_P["P"]) - s["x"]) + _R(_P["dmax"]), _R(B))),
                   And(s["tr"], LE(s["w"] + (_R(_P["dmax"]) - s["y"]), _R(B))))
        return And(GE(s["x"], _R(0)), LE(s["x"], _R(_P["P"])),
                   Implies(s["tr"], And(Equals(s["y"], s["x"]), LE(s["y"], _R(_P["dmax"])))),
                   Implies(s["d1"], resto))

    def _valida(f):
        with Solver(name="z3") as sol:
            return not sol.is_sat(Not(f))

    _s, _p = _declare(0), _declare(1)
    print("ini → InvR            :", _valida(Implies(ini_dess(_s), inv_rec(_s))))
    print("InvR ∧ T ∧ rec1 → InvR':", _valida(Implies(And(inv_rec(_s), _trans(_s, _p), _s["rec1"]),
                                                   inv_rec(_p))))
    print("InvR → ψ(P+dmax)      :", _valida(Implies(inv_rec(_s), psi(_P["P"] + _P["dmax"])(_s))))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Contraexemplo: período de sincronização demasiado longo

    Especificação: δ = 1. Com $P = 10$ a propriedade verifica-se; com $P = 12$ a deriva acumulada
    durante $K$ períodos de perdas ultrapassa o limite.
    """)
    return


@app.cell
def _(PARAMS, bmc, cria_sistema, skew_ok):
    _k, _txt = bmc(cria_sistema(dict(PARAMS, P=12)), skew_ok(1.0), 16)
    print("P=12, δ=1: " + (_txt if _k is None else f"violação em {_k} transições\n{_txt}"))
    return


if __name__ == "__main__":
    app.run()
