"""
Sistema multiagente de coordinacion de horarios entre companeros de equipo.

Modelo:
    U_i(h) = w1*P_i + w2*Q_i - w3*C_i - w4*R_i - w5*D_i
    J = sum(U_i) - lambda * Var(U_1, ..., U_N)

Restricciones:
    - Tiempo total <= T_max
    - Precedencia entre bloques (tareas antes de reuniones dependientes)
    - Exclusion mutua de recursos compartidos (sala, zoom, ...)
    - Disponibilidad individual (sin traslapes por agente, ni con compromisos duros)
"""

from __future__ import annotations

import itertools
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

DIAS = ["Lunes", "Martes", "Miercoles", "Jueves", "Viernes"]

Intervalo = Tuple[int, float, float]  # (dia, inicio, fin)


def se_traslapan(dia1: int, i1: float, f1: float, dia2: int, i2: float, f2: float) -> bool:
    return dia1 == dia2 and i1 < f2 and i2 < f1


# ---------------------------------------------------------------------------
# Bloque de tiempo candidato
# ---------------------------------------------------------------------------

@dataclass
class Bloque:
    id: str
    nombre: str
    dia: int  # indice sobre DIAS
    inicio: float
    fin: float
    participantes: List[str]
    recurso: Optional[str] = None  # ej. "sala", "zoom"
    precedencias: List[str] = field(default_factory=list)  # ids de bloques requeridos antes
    prioridad_base: float = 3.0
    beneficio_base: float = 3.0
    prioridad_por_agente: Dict[str, float] = field(default_factory=dict)
    beneficio_por_agente: Dict[str, float] = field(default_factory=dict)

    @property
    def duracion(self) -> float:
        return self.fin - self.inicio

    def prioridad(self, agente_id: str) -> float:
        return self.prioridad_por_agente.get(agente_id, self.prioridad_base)

    def beneficio(self, agente_id: str) -> float:
        return self.beneficio_por_agente.get(agente_id, self.beneficio_base)

    def __repr__(self):
        return f"{self.id}:{self.nombre} ({DIAS[self.dia]} {self.inicio:.1f}-{self.fin:.1f}h)"


# ---------------------------------------------------------------------------
# Agente
# ---------------------------------------------------------------------------

@dataclass
class Agente:
    id: str
    nombre: str
    ocupado_duro: List[Intervalo] = field(default_factory=list)   # compromisos fijos, no negociables
    ocupado_blando: List[Intervalo] = field(default_factory=list)  # compromisos movibles (generan costo C)
    w1: float = 1.0  # peso prioridad
    w2: float = 1.0  # peso beneficio esperado
    w3: float = 0.5  # peso costo de mover otras cosas
    w4: float = 2.0  # peso riesgo de choque
    w5: float = 0.2  # peso duracion

    def calcular_utilidad(self, bloque: Bloque) -> float:
        if self.id not in bloque.participantes:
            return 0.0

        P = bloque.prioridad(self.id)
        Q = bloque.beneficio(self.id)

        C = sum(
            1 for (d, i, f) in self.ocupado_blando
            if se_traslapan(d, i, f, bloque.dia, bloque.inicio, bloque.fin)
        )

        R = 1.0 if any(
            se_traslapan(d, i, f, bloque.dia, bloque.inicio, bloque.fin)
            for (d, i, f) in self.ocupado_duro
        ) else 0.0

        D = bloque.duracion

        return self.w1 * P + self.w2 * Q - self.w3 * C - self.w4 * R - self.w5 * D


# ---------------------------------------------------------------------------
# Sistema multiagente
# ---------------------------------------------------------------------------

class SistemaMultiagente:
    def __init__(self, agentes: List[Agente], bloques: List[Bloque],
                 T_max: Optional[float] = None, lam: float = 0.5):
        self.agentes = {a.id: a for a in agentes}
        self.bloques = {b.id: b for b in bloques}
        self.T_max = T_max
        self.lam = lam

    # -- utilidades ---------------------------------------------------

    def evaluar_utilidades(self) -> Dict[str, Dict[str, float]]:
        """Utilidad de cada bloque para cada agente participante."""
        return {
            b.id: {
                a.id: a.calcular_utilidad(b)
                for a in self.agentes.values() if a.id in b.participantes
            }
            for b in self.bloques.values()
        }

    def utilidad_por_agente(self, seleccion: List[Bloque]) -> Dict[str, float]:
        utilidades = {a_id: 0.0 for a_id in self.agentes}
        for b in seleccion:
            for a_id in b.participantes:
                utilidades[a_id] += self.agentes[a_id].calcular_utilidad(b)
        return utilidades

    def utilidad_social(self, seleccion: List[Bloque], lam: Optional[float] = None) -> Tuple[float, Dict[str, float]]:
        lam = self.lam if lam is None else lam
        utilidades = self.utilidad_por_agente(seleccion)
        valores = list(utilidades.values())
        varianza = statistics.pvariance(valores) if len(valores) > 1 else 0.0
        J = sum(valores) - lam * varianza
        return J, utilidades

    # -- restricciones --------------------------------------------------

    def es_factible(self, seleccion: List[Bloque]) -> bool:
        # Tiempo total
        if self.T_max is not None:
            if sum(b.duracion for b in seleccion) > self.T_max:
                return False

        ids_seleccionados = {b.id for b in seleccion}

        # Precedencia: el bloque requerido debe estar seleccionado y terminar antes
        for b in seleccion:
            for prec_id in b.precedencias:
                if prec_id not in ids_seleccionados:
                    return False
                prec = self.bloques[prec_id]
                fin_prec = prec.dia * 24 + prec.fin
                inicio_b = b.dia * 24 + b.inicio
                if fin_prec > inicio_b:
                    return False

        # Exclusion mutua de recursos compartidos
        for b1, b2 in itertools.combinations(seleccion, 2):
            if b1.recurso is not None and b1.recurso == b2.recurso:
                if se_traslapan(b1.dia, b1.inicio, b1.fin, b2.dia, b2.inicio, b2.fin):
                    return False

        # Disponibilidad individual: sin traslapes por agente ni con compromisos duros
        for a_id, agente in self.agentes.items():
            bloques_agente = [b for b in seleccion if a_id in b.participantes]
            for b1, b2 in itertools.combinations(bloques_agente, 2):
                if se_traslapan(b1.dia, b1.inicio, b1.fin, b2.dia, b2.inicio, b2.fin):
                    return False
            for b in bloques_agente:
                for (d, i, f) in agente.ocupado_duro:
                    if se_traslapan(d, i, f, b.dia, b.inicio, b.fin):
                        return False

        return True

    # -- seleccion --------------------------------------------------

    def seleccionar_optimo(self, lam: Optional[float] = None) -> Tuple[List[Bloque], float, Dict[str, float]]:
        """Fuerza bruta sobre subconjuntos de bloques candidatos (ok para pocos bloques)."""
        lam = self.lam if lam is None else lam
        bloques = list(self.bloques.values())
        mejor_seleccion: List[Bloque] = []
        mejor_J = float("-inf")
        mejor_utilidades: Dict[str, float] = {}

        for r in range(len(bloques), -1, -1):
            pass  # no-op, iteramos todos los subconjuntos abajo sin importar tamano

        for mascara in range(2 ** len(bloques)):
            subset = [b for i, b in enumerate(bloques) if mascara & (1 << i)]
            if not self.es_factible(subset):
                continue
            J, utilidades = self.utilidad_social(subset, lam=lam)
            if J > mejor_J:
                mejor_J = J
                mejor_seleccion = subset
                mejor_utilidades = utilidades

        return mejor_seleccion, mejor_J, mejor_utilidades

    # -- bonus: replanificacion ante eventos inesperados --------------

    def registrar_cancelacion(self, agente_id: str, nuevo_ocupado_duro: Intervalo) -> None:
        """Un agente cambia disponibilidad (ej. se le cruza algo urgente)."""
        self.agentes[agente_id].ocupado_duro.append(nuevo_ocupado_duro)

    def replanificar(self, agente_id: str, nuevo_ocupado_duro: Intervalo,
                      lam: Optional[float] = None) -> Tuple[List[Bloque], float, Dict[str, float]]:
        self.registrar_cancelacion(agente_id, nuevo_ocupado_duro)
        return self.seleccionar_optimo(lam=lam)


# ---------------------------------------------------------------------------
# Q-Learning: seleccion de bloques como MDP episodico (Algoritmo 3 del reto)
# ---------------------------------------------------------------------------

Estado = Tuple[int, FrozenSet[str]]  # (paso en el orden de bloques, ids ya incluidos)


class QLearningPlanner:
    """Aprende, via Q-Learning tabular con politica epsilon-greedy, que
    subconjunto de bloques seleccionar.

    Cada episodio recorre los bloques candidatos en un orden fijo; en cada
    paso la accion es "incluir" (1) o "excluir" (0) el bloque actual, y solo
    se permite incluirlo si la seleccion resultante sigue siendo factible
    (ver SistemaMultiagente.es_factible). La recompensa es 0 en cada paso
    intermedio y J (la utilidad social del bloque 5) solo al terminar el
    episodio, sobre la seleccion final -- por eso el valor de una accion
    temprana depende del retorno futuro descontado, tal como en la ecuacion
    de Bellman (bloque 9).

    Sirve como alternativa a seleccionar_optimo(): en vez de enumerar los
    2**n subconjuntos por fuerza bruta, aprende una politica por experiencia,
    lo cual escala a instancias con muchos mas bloques candidatos.
    """

    def __init__(self, sistema: SistemaMultiagente, alpha: float = 0.1,
                 gamma: float = 0.95, epsilon: float = 1.0, epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.999,
                 lam: Optional[float] = None, semilla: Optional[int] = None):
        self.sistema = sistema
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lam = sistema.lam if lam is None else lam
        self.orden: List[str] = list(sistema.bloques.keys())
        self.Q: Dict[Estado, Dict[int, float]] = {}
        self.rng = random.Random(semilla)
        self.historial_J: List[float] = []
        self.historial_epsilon: List[float] = []
        self.historial_td_error: List[float] = []

    def _q(self, estado: Estado) -> Dict[int, float]:
        return self.Q.setdefault(estado, {0: 0.0, 1: 0.0})

    def _incluir_es_factible(self, seleccion_ids: FrozenSet[str], bloque_id: str) -> bool:
        candidato = [self.sistema.bloques[i] for i in seleccion_ids | {bloque_id}]
        return self.sistema.es_factible(candidato)

    def _elegir_accion(self, estado: Estado, acciones_validas: List[int], explorar: bool) -> int:
        if explorar and self.rng.random() < self.epsilon:
            return self.rng.choice(acciones_validas)
        q = self._q(estado)
        return max(acciones_validas, key=lambda a: q[a])

    def ejecutar_episodio(self, explorar: bool = True) -> Tuple[float, Dict[str, float], List[Bloque], float]:
        """Corre un episodio completo. Si explorar=True actualiza Q siguiendo
        el Algoritmo 3 (observar, actuar epsilon-greedy, actualizar Q); si es
        False corre la politica greedy pura, sin aprender (para evaluar).
        Devuelve tambien el error TD |delta| promedio del episodio (0 si no
        se exploro/actualizo), como medida de cuanto sigue "sorprendiendose"
        el agente."""
        seleccion_ids: FrozenSet[str] = frozenset()
        total_pasos = len(self.orden)
        J, utilidades = 0.0, {}
        deltas: List[float] = []

        for paso, bloque_id in enumerate(self.orden):
            estado: Estado = (paso, seleccion_ids)
            acciones_validas = [0]
            if self._incluir_es_factible(seleccion_ids, bloque_id):
                acciones_validas.append(1)

            accion = self._elegir_accion(estado, acciones_validas, explorar)
            nueva_seleccion = seleccion_ids | {bloque_id} if accion == 1 else seleccion_ids
            estado_sig: Estado = (paso + 1, nueva_seleccion)
            es_terminal = (paso + 1) == total_pasos

            if es_terminal:
                bloques_finales = [self.sistema.bloques[i] for i in nueva_seleccion]
                J, utilidades = self.sistema.utilidad_social(bloques_finales, lam=self.lam)
                r = J
            else:
                r = 0.0

            if explorar:
                q = self._q(estado)
                mejor_futuro = max(self._q(estado_sig).values()) if not es_terminal else 0.0
                delta = r + self.gamma * mejor_futuro - q[accion]
                q[accion] += self.alpha * delta
                deltas.append(abs(delta))

            seleccion_ids = nueva_seleccion

        bloques_finales = [self.sistema.bloques[i] for i in seleccion_ids]
        td_error_medio = statistics.fmean(deltas) if deltas else 0.0
        return J, utilidades, bloques_finales, td_error_medio

    def entrenar(self, n_episodios: int = 8000) -> List[float]:
        """Entrena n_episodios episodios, con epsilon decayendo geometricamente
        de self.epsilon hasta epsilon_min (mucha exploracion al inicio, casi
        pura explotacion al final). Devuelve el historial de J por episodio."""
        self.historial_J = []
        self.historial_epsilon = []
        self.historial_td_error = []
        for _ in range(n_episodios):
            J, _, _, td_error = self.ejecutar_episodio(explorar=True)
            self.historial_J.append(J)
            self.historial_epsilon.append(self.epsilon)
            self.historial_td_error.append(td_error)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.historial_J

    def mejor_politica(self) -> Tuple[List[Bloque], float, Dict[str, float]]:
        """Corre la politica greedy aprendida (sin exploracion ni aprendizaje)."""
        J, utilidades, bloques, _ = self.ejecutar_episodio(explorar=False)
        return bloques, J, utilidades


# ---------------------------------------------------------------------------
# Coordinacion por subasta: asignacion optima agente-tarea (Ejercicio 4)
# ---------------------------------------------------------------------------

class AsignadorSubasta:
    """Resuelve un problema de asignacion uno-a-uno entre agentes y tareas
    (p.ej. "quien entrega cada paquete", "quien se hace cargo de cada
    tramite"): cada tarea se le da a un solo agente candidato (de
    tarea.participantes) y cada agente toma como maximo una tarea.

    Es un problema distinto al de SistemaMultiagente.seleccionar_optimo (que
    decide que subconjunto de bloques *compartidos* entra al horario). Aqui
    se resuelve de dos formas, para comparar coordinacion centralizada vs.
    descentralizada:

      1. asignar_optimo(): algoritmo hungaro (Kuhn-Munkres) sobre la matriz
         de utilidades agente x tarea -- el optimo global, requiere ver el
         problema completo.
      2. asignar_por_subasta(): subasta secuencial -- las tareas se rematan
         una por una y se las lleva quien puje mas alto (su propia utilidad),
         sin que ningun agente necesite ver el problema completo.
    """

    def __init__(self, agentes: List[Agente], tareas: List[Bloque]):
        self.agentes = agentes
        self.tareas = tareas

    def matriz_utilidad(self) -> Dict[Tuple[str, str], float]:
        """Utilidad de cada agente por cada tarea (-inf si no es candidato)."""
        return {
            (a.id, t.id): a.calcular_utilidad(t) if a.id in t.participantes else float("-inf")
            for t in self.tareas
            for a in self.agentes
        }

    # -- optimo centralizado: algoritmo hungaro -----------------------

    def asignar_optimo(self) -> Tuple[Dict[str, str], float]:
        """Algoritmo hungaro (Kuhn-Munkres, O(n^3)) para maximizar la
        utilidad total de la asignacion agente->tarea. Devuelve
        {agente_id: tarea_id} y la utilidad total. Agentes/tareas sin
        contraparte candidata factible quedan sin asignar."""
        agentes, tareas = self.agentes, self.tareas
        n, m = len(agentes), len(tareas)
        dim = max(n, m, 1)

        CARO = 1e6  # penaliza asignaciones invalidas (agente no candidato)
        costo = [[CARO] * dim for _ in range(dim)]
        for i, a in enumerate(agentes):
            for j, t in enumerate(tareas):
                if a.id in t.participantes:
                    costo[i][j] = -a.calcular_utilidad(t)  # minimizar costo = maximizar utilidad
        for i in range(n, dim):
            costo[i] = [0.0] * dim
        for j in range(m, dim):
            for i in range(dim):
                costo[i][j] = 0.0

        asignacion = _hungaro(costo)

        resultado: Dict[str, str] = {}
        utilidad_total = 0.0
        for i, j in enumerate(asignacion):
            if i < n and j < m and agentes[i].id in tareas[j].participantes:
                resultado[agentes[i].id] = tareas[j].id
                utilidad_total += agentes[i].calcular_utilidad(tareas[j])
        return resultado, utilidad_total

    # -- coordinacion descentralizada: subasta secuencial --------------

    def asignar_por_subasta(self, orden: Optional[List[str]] = None,
                             semilla: Optional[int] = None) -> Tuple[Dict[str, str], float]:
        """Subasta secuencial: las tareas se rematan una por una (en `orden`,
        o en orden aleatorio si no se da). En cada remate, cada agente aun
        libre puja su propia utilidad por esa tarea (puja veraz) y gana quien
        puje mas alto (empates: el id de agente mas chico). Mecanismo
        descentralizado: ningun agente ve el problema completo, solo su
        propia utilidad por la tarea que se esta rematando."""
        rng = random.Random(semilla)
        tareas_por_id = {t.id: t for t in self.tareas}
        orden = list(orden) if orden is not None else list(tareas_por_id.keys())
        if orden is None:
            rng.shuffle(orden)

        libres = {a.id: a for a in self.agentes}
        resultado: Dict[str, str] = {}
        utilidad_total = 0.0

        for tarea_id in orden:
            tarea = tareas_por_id[tarea_id]
            pujas = sorted(
                ((a.calcular_utilidad(tarea), a.id) for a in libres.values()
                 if a.id in tarea.participantes),
                key=lambda pu: (-pu[0], pu[1]),
            )
            if not pujas:
                continue
            mejor_puja, ganador_id = pujas[0]
            if mejor_puja <= 0:
                continue  # nadie quiere la tarea a utilidad positiva: se queda sin asignar
            resultado[ganador_id] = tarea_id
            utilidad_total += mejor_puja
            del libres[ganador_id]

        return resultado, utilidad_total

    # -- linea base: asignacion aleatoria --------------------------------

    def asignar_aleatorio(self, semilla: Optional[int] = None) -> Tuple[Dict[str, str], float]:
        """Linea base sin ningun mecanismo de coordinacion: las tareas se
        recorren en orden aleatorio y cada una se le da a un agente
        candidato elegido al azar entre los que siguen libres (no al mejor
        postor). Sirve para medir cuanto aporta realmente la subasta, tal
        como una politica aleatoria sirve de referencia para Q-Learning."""
        rng = random.Random(semilla)
        tareas_por_id = {t.id: t for t in self.tareas}
        orden = list(tareas_por_id.keys())
        rng.shuffle(orden)

        libres = {a.id: a for a in self.agentes}
        resultado: Dict[str, str] = {}
        utilidad_total = 0.0

        for tarea_id in orden:
            tarea = tareas_por_id[tarea_id]
            candidatos = [a_id for a_id in libres if a_id in tarea.participantes]
            if not candidatos:
                continue
            ganador_id = rng.choice(candidatos)
            utilidad = libres[ganador_id].calcular_utilidad(tarea)
            resultado[ganador_id] = tarea_id
            utilidad_total += utilidad
            del libres[ganador_id]

        return resultado, utilidad_total


def _hungaro(costo: List[List[float]]) -> List[int]:
    """Algoritmo hungaro (Kuhn-Munkres) O(n^3) sobre una matriz de costo
    cuadrada, via potenciales (metodo de Jonker-Volgenant simplificado), sin
    dependencias externas. Devuelve, para cada fila i, la columna asignada
    que minimiza el costo total."""
    n = len(costo)
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)   # p[j] = fila (1-indexada) asignada a la columna j
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = -1
            for j in range(1, n + 1):
                if not used[j]:
                    cur = costo[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    asignacion_por_fila = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            asignacion_por_fila[p[j] - 1] = j - 1
    return asignacion_por_fila


# ---------------------------------------------------------------------------
# Negociacion y equilibrio de Nash entre dos agentes
# ---------------------------------------------------------------------------

Estrategia = str
PerfilEstrategias = Tuple[Estrategia, Estrategia]
Pago = Tuple[float, float]


class JuegoDosAgentes:
    """Modela una negociacion entre dos agentes como un juego en forma normal:
    cada agente elige una estrategia de un conjunto finito, y la matriz de
    pagos da la utilidad de cada uno para cada combinacion de estrategias.

    A diferencia del horario optimo o la subasta (donde un coordinador -- el
    propio algoritmo -- resuelve el problema completo), aqui no hay
    coordinador: cada agente decide por su cuenta, anticipando lo que hara
    el otro. El equilibrio de Nash es el punto donde ningun agente mejora
    cambiando unilateralmente de estrategia."""

    def __init__(self, agente1_id: str, agente2_id: str,
                 estrategias1: List[Estrategia], estrategias2: List[Estrategia],
                 pagos: Dict[PerfilEstrategias, Pago]):
        self.agente1_id = agente1_id
        self.agente2_id = agente2_id
        self.estrategias1 = estrategias1
        self.estrategias2 = estrategias2
        self.pagos = pagos

    def mejores_respuestas_1(self, e2: Estrategia) -> List[Estrategia]:
        """Estrategias del agente 1 que maximizan su pago si el agente 2 juega e2."""
        valores = {e1: self.pagos[(e1, e2)][0] for e1 in self.estrategias1}
        mejor = max(valores.values())
        return [e1 for e1, v in valores.items() if v == mejor]

    def mejores_respuestas_2(self, e1: Estrategia) -> List[Estrategia]:
        """Estrategias del agente 2 que maximizan su pago si el agente 1 juega e1."""
        valores = {e2: self.pagos[(e1, e2)][1] for e2 in self.estrategias2}
        mejor = max(valores.values())
        return [e2 for e2, v in valores.items() if v == mejor]

    def estrategia_dominante(self, jugador: int) -> Optional[Estrategia]:
        """La estrategia que es mejor respuesta a TODO lo que haga el rival
        (estrategia estrictamente dominante), si existe; None si no hay una
        unica estrategia dominante."""
        if jugador == 1:
            candidatas = [e1 for e1 in self.estrategias1
                          if all(e1 in self.mejores_respuestas_1(e2) for e2 in self.estrategias2)]
        else:
            candidatas = [e2 for e2 in self.estrategias2
                          if all(e2 in self.mejores_respuestas_2(e1) for e1 in self.estrategias1)]
        return candidatas[0] if len(candidatas) == 1 else None

    def equilibrios_nash_puros(self) -> List[PerfilEstrategias]:
        """Un perfil (e1, e2) es equilibrio de Nash en estrategias puras si
        e1 es mejor respuesta a e2 Y e2 es mejor respuesta a e1: ningun
        agente mejora desviandose unilateralmente."""
        equilibrios = []
        for e1 in self.estrategias1:
            for e2 in self.estrategias2:
                if e1 in self.mejores_respuestas_1(e2) and e2 in self.mejores_respuestas_2(e1):
                    equilibrios.append((e1, e2))
        return equilibrios

    def optimos_pareto(self) -> List[PerfilEstrategias]:
        """Perfiles donde no existe otro perfil que mejore a ambos agentes (o
        a uno sin empeorar al otro). Sirve para contrastar el equilibrio de
        Nash (estable) contra el optimo social (eficiente): no siempre
        coinciden, y esa brecha es la leccion central de la seccion."""
        perfiles = [(e1, e2) for e1 in self.estrategias1 for e2 in self.estrategias2]
        optimos = []
        for p in perfiles:
            u1, u2 = self.pagos[p]
            dominado = any(
                self.pagos[q][0] >= u1 and self.pagos[q][1] >= u2 and self.pagos[q] != (u1, u2)
                for q in perfiles
            )
            if not dominado:
                optimos.append(p)
        return optimos


def construir_caso_negociacion(agente1_id: str = "A1", agente2_id: str = "A3") -> JuegoDosAgentes:
    """Dos companeros deciden si Cooperar en una tarea compartida (ayudarse
    con la carga de trabajo) o No_cooperar (dejar que el otro cargue con
    todo). Es un dilema del prisionero: cooperar mutuamente da el mejor
    resultado conjunto (6, 6), pero cada agente tiene incentivo individual a
    no cooperar -- si el otro coopera, aprovecharse da mas (8 contra 2) -- y
    ese incentivo, compartido por ambos, empuja el sistema hacia el
    equilibrio de Nash (No_cooperar, No_cooperar) con pago (3, 3): peor para
    los dos que si hubieran cooperado."""
    pagos = {
        ("Cooperar", "Cooperar"): (6.0, 6.0),
        ("Cooperar", "No_cooperar"): (2.0, 8.0),
        ("No_cooperar", "Cooperar"): (8.0, 2.0),
        ("No_cooperar", "No_cooperar"): (3.0, 3.0),
    }
    return JuegoDosAgentes(agente1_id, agente2_id,
                            ["Cooperar", "No_cooperar"], ["Cooperar", "No_cooperar"],
                            pagos)


def imprimir_analisis_nash(sistema: "SistemaMultiagente", juego: JuegoDosAgentes) -> None:
    nombre1 = sistema.agentes[juego.agente1_id].nombre
    nombre2 = sistema.agentes[juego.agente2_id].nombre

    print(f"\n=== Negociacion entre {nombre1} ({juego.agente1_id}) y {nombre2} ({juego.agente2_id}) ===")
    print("Matriz de pagos (U1, U2):")
    for e1 in juego.estrategias1:
        fila = "  ".join(f"{e2}={juego.pagos[(e1, e2)]}" for e2 in juego.estrategias2)
        print(f"  {e1}: {fila}")

    dom1 = juego.estrategia_dominante(1)
    dom2 = juego.estrategia_dominante(2)
    print(f"Estrategia dominante de {nombre1}: {dom1 or 'ninguna'}")
    print(f"Estrategia dominante de {nombre2}: {dom2 or 'ninguna'}")

    equilibrios = juego.equilibrios_nash_puros()
    print(f"Equilibrios de Nash (estrategias puras): {equilibrios}")

    optimos = juego.optimos_pareto()
    print(f"Optimos de Pareto: {optimos}")

    if equilibrios and optimos and set(equilibrios) != set(optimos):
        print("El equilibrio de Nash NO coincide con el optimo de Pareto: "
              "el resultado estable no es el socialmente mas eficiente.")


# ---------------------------------------------------------------------------
# Simulacion de ejemplo
# ---------------------------------------------------------------------------

def construir_caso_ejemplo() -> Tuple[List[Agente], List[Bloque]]:
    agentes = [
        Agente("A1", "Ana", ocupado_duro=[(0, 8, 9)], ocupado_blando=[(1, 10, 11)]),
        Agente("A2", "Bruno", ocupado_duro=[(1, 15, 17)], ocupado_blando=[]),
        Agente("A3", "Carla", ocupado_duro=[], ocupado_blando=[(0, 13, 14), (2, 9, 10)]),
        Agente("A4", "Diego", ocupado_duro=[(2, 8, 10)], ocupado_blando=[]),
    ]

    bloques = [
        Bloque("B1", "Diseno de API (tarea)", dia=0, inicio=10, fin=12,
               participantes=["A1", "A3"], recurso=None,
               prioridad_base=4, beneficio_base=3),
        Bloque("B2", "Revision de API (reunion)", dia=0, inicio=13, fin=14,
               participantes=["A1", "A2", "A3"], recurso="sala",
               precedencias=["B1"], prioridad_base=4, beneficio_base=4),
        Bloque("B3", "Sync semanal", dia=1, inicio=9, fin=10,
               participantes=["A1", "A2", "A3", "A4"], recurso="zoom",
               prioridad_base=3, beneficio_base=3),
        Bloque("B4", "Pair programming", dia=1, inicio=11, fin=13,
               participantes=["A2", "A4"], recurso=None,
               prioridad_base=2, beneficio_base=4),
        Bloque("B5", "Demo a cliente", dia=2, inicio=11, fin=12,
               participantes=["A1", "A2", "A4"], recurso="sala",
               prioridad_base=5, beneficio_base=5),
        Bloque("B6", "Retro de sprint", dia=2, inicio=15, fin=16,
               participantes=["A1", "A2", "A3", "A4"], recurso="zoom",
               prioridad_base=3, beneficio_base=2),
        Bloque("B7", "1:1 mentoria", dia=3, inicio=9, fin=9.5,
               participantes=["A3", "A4"], recurso=None,
               prioridad_base=2, beneficio_base=3),
        Bloque("B8", "Planning siguiente sprint", dia=3, inicio=14, fin=15.5,
               participantes=["A1", "A2", "A3", "A4"], recurso="sala",
               prioridad_base=4, beneficio_base=3),
    ]

    return agentes, bloques


def construir_caso_asignacion(agentes: List[Agente]) -> List[Bloque]:
    """Tareas de reparto (una por agente responsable) para demostrar
    coordinacion por subasta / asignacion optima (Ejercicio 4)."""
    return [
        Bloque("T1", "Entrega paquete Norte", dia=0, inicio=9, fin=10,
               participantes=["A1", "A3", "A4"], prioridad_base=4, beneficio_base=4),
        Bloque("T2", "Entrega paquete Centro", dia=0, inicio=9, fin=9.5,
               participantes=["A1", "A2", "A3"], prioridad_base=3, beneficio_base=5),
        Bloque("T3", "Soporte cliente urgente", dia=0, inicio=11, fin=12,
               participantes=["A2", "A3", "A4"], prioridad_base=5, beneficio_base=3),
        Bloque("T4", "Inventario bodega", dia=0, inicio=14, fin=16,
               participantes=["A1", "A4"], prioridad_base=2, beneficio_base=2),
        Bloque("T5", "Ruta de recoleccion Sur", dia=1, inicio=9, fin=11,
               participantes=["A2", "A3", "A4"], prioridad_base=3, beneficio_base=4),
    ]


def imprimir_asignacion(agentes: List[Agente], titulo: str,
                         asignacion: Dict[str, str], utilidad_total: float) -> None:
    nombres = {a.id: a.nombre for a in agentes}
    print(f"\n=== {titulo} ===")
    print(f"Utilidad total = {utilidad_total:.3f}")
    if not asignacion:
        print("  (sin asignaciones)")
        return
    for a_id, t_id in sorted(asignacion.items()):
        print(f"  - {nombres[a_id]} ({a_id}) -> {t_id}")


def imprimir_seleccion(sistema: SistemaMultiagente, titulo: str,
                        seleccion: List[Bloque], J: float, utilidades: Dict[str, float]) -> None:
    print(f"\n=== {titulo} ===")
    print(f"J (utilidad social) = {J:.3f}")
    print("Bloques seleccionados:")
    for b in sorted(seleccion, key=lambda x: (x.dia, x.inicio)):
        print(f"  - {b}  [participantes: {', '.join(b.participantes)}]"
              f"{'  recurso=' + b.recurso if b.recurso else ''}")
    print("Utilidad por agente:")
    for a_id, u in utilidades.items():
        print(f"  - {sistema.agentes[a_id].nombre}: {u:.3f}")
    var = statistics.pvariance(list(utilidades.values())) if len(utilidades) > 1 else 0.0
    print(f"Varianza entre agentes: {var:.3f}")


def graficar_resultados(sistema: SistemaMultiagente,
                         seleccion_lam: List[Bloque], util_lam: Dict[str, float], J_lam: float,
                         seleccion_sin_lam: List[Bloque], util_sin_lam: Dict[str, float], J_sin_lam: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1) Utilidad por agente, con y sin penalizacion por varianza
    ax = axes[0]
    nombres = [sistema.agentes[a_id].nombre for a_id in util_lam]
    x = range(len(nombres))
    ancho = 0.35
    ax.bar([i - ancho / 2 for i in x], list(util_sin_lam.values()), width=ancho, label="lambda=0")
    ax.bar([i + ancho / 2 for i in x], list(util_lam.values()), width=ancho, label=f"lambda={sistema.lam}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(nombres)
    ax.set_ylabel("Utilidad U_i")
    ax.set_title("Utilidad por agente")
    ax.legend()

    # 2) Utilidad social J comparada
    ax = axes[1]
    ax.bar(["lambda=0", f"lambda={sistema.lam}"], [J_sin_lam, J_lam], color=["tab:blue", "tab:orange"])
    ax.set_ylabel("J")
    ax.set_title("Utilidad social J: efecto de la penalizacion por varianza")

    # 3) Horario final asignado (con lambda>0) como diagrama de bloques
    ax = axes[2]
    colores = plt.cm.tab10.colors
    color_por_bloque = {b.id: colores[i % len(colores)] for i, b in enumerate(sistema.bloques.values())}
    for b in seleccion_lam:
        ax.barh(y=DIAS[b.dia], width=b.duracion, left=b.inicio, height=0.5,
                color=color_por_bloque[b.id], edgecolor="black")
        ax.text(b.inicio + b.duracion / 2, DIAS[b.dia], b.id, ha="center", va="center", fontsize=8)
    ax.set_xlabel("Hora del dia")
    ax.set_title(f"Horario final asignado (lambda={sistema.lam})")
    ax.set_xlim(7, 18)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig("horario_resultado.png", dpi=150)
    print("\nGrafico guardado en horario_resultado.png")
    plt.show()


def simular():
    agentes, bloques = construir_caso_ejemplo()
    sistema = SistemaMultiagente(agentes, bloques, T_max=7.0, lam=2.5)

    print("Utilidad de cada bloque para cada agente participante:")
    utilidades_bloques = sistema.evaluar_utilidades()
    for b_id, utils in utilidades_bloques.items():
        legibles = ", ".join(f"{sistema.agentes[a].nombre}={u:.2f}" for a, u in utils.items())
        print(f"  {b_id} ({sistema.bloques[b_id].nombre}): {legibles}")

    # Seleccion optima con penalizacion por varianza (equidad)
    seleccion_lam, J_lam, util_lam = sistema.seleccionar_optimo(lam=sistema.lam)
    imprimir_seleccion(sistema, f"Seleccion optima (lambda={sistema.lam}, con equidad)", seleccion_lam, J_lam, util_lam)

    # Bonus: comparacion lambda=0 vs lambda>0
    seleccion_sin_lam, J_sin_lam, util_sin_lam = sistema.seleccionar_optimo(lam=0.0)
    imprimir_seleccion(sistema, "Seleccion optima (lambda=0, sin equidad)", seleccion_sin_lam, J_sin_lam, util_sin_lam)

    # Bonus: evento inesperado -> Diego cancela y queda ocupado el jueves 14-16h
    print("\n>>> Evento inesperado: Diego (A4) queda ocupado el Jueves 14:00-16:00h, se replanifica...")
    seleccion_repl, J_repl, util_repl = sistema.replanificar("A4", (3, 14, 16), lam=sistema.lam)
    imprimir_seleccion(sistema, "Replanificacion tras evento inesperado", seleccion_repl, J_repl, util_repl)

    # Bonus: Q-Learning como alternativa a la fuerza bruta (Algoritmo 3)
    print("\n>>> Entrenando Q-Learning (alpha=0.15, gamma=0.95, epsilon: 1.0 -> 0.05) "
          "para aprender la seleccion de bloques...")
    planner = QLearningPlanner(sistema, alpha=0.15, gamma=0.95, lam=sistema.lam, semilla=0)
    historial = planner.entrenar(n_episodios=8000)
    seleccion_ql, J_ql, util_ql = planner.mejor_politica()
    imprimir_seleccion(sistema, "Politica aprendida por Q-Learning", seleccion_ql, J_ql, util_ql)
    print(f"J promedio ultimos 200 episodios de entrenamiento: "
          f"{statistics.fmean(historial[-200:]):.3f}  (optimo por fuerza bruta: {J_lam:.3f})")

    # Bonus: coordinacion por subasta / asignacion optima (Ejercicio 4)
    print("\n>>> Asignacion optima agente-tarea via subasta (algoritmo hungaro vs. subasta secuencial)...")
    tareas = construir_caso_asignacion(agentes)
    asignador = AsignadorSubasta(agentes, tareas)

    asignacion_optima, utilidad_optima = asignador.asignar_optimo()
    imprimir_asignacion(agentes, "Asignacion optima (algoritmo hungaro, centralizado)",
                         asignacion_optima, utilidad_optima)

    asignacion_subasta, utilidad_subasta = asignador.asignar_por_subasta(semilla=0)
    imprimir_asignacion(agentes, "Asignacion por subasta secuencial (descentralizado)",
                         asignacion_subasta, utilidad_subasta)

    brecha = utilidad_optima - utilidad_subasta
    print(f"\nBrecha subasta vs. optimo = {brecha:.3f} "
          f"({'igual al optimo' if brecha <= 1e-9 else 'subasta se queda corta'})")

    # Bonus: negociacion y equilibrio de Nash
    print("\n>>> Negociacion entre dos agentes: cooperar o no en una tarea compartida...")
    juego = construir_caso_negociacion("A1", "A3")
    imprimir_analisis_nash(sistema, juego)

    try:
        graficar_resultados(sistema, seleccion_lam, util_lam, J_lam,
                             seleccion_sin_lam, util_sin_lam, J_sin_lam)
    except ImportError:
        print("\n(matplotlib no disponible: se omiten los graficos)")


if __name__ == "__main__":
    simular()
