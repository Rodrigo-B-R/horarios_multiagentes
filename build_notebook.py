"""Genera horarios_simulacion.ipynb (no se versiona a mano, se reconstruye con este script)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
"""# Simulacion: Sistema multiagente de coordinacion de horarios

Este notebook corre la simulacion definida en `horarios_multiagente.py`:
evalua utilidades, selecciona el horario optimo (con y sin penalizacion por
varianza), muestra el efecto de un evento inesperado (replanificacion), y
genera una animacion en video (`.mp4`) mostrando como se va construyendo el
horario final bloque a bloque y como crece la utilidad de cada companero."""
))

cells.append(nbf.v4.new_code_cell(
"""import sys, os
sys.path.insert(0, os.path.abspath("."))

import statistics
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import imageio_ffmpeg
from IPython.display import Video

plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()

import horarios_multiagente as hm

DIAS = hm.DIAS
DIR_SALIDA = "representaciones_graficas"
os.makedirs(DIR_SALIDA, exist_ok=True)

# colores consistentes por metodo, para distinguir de un vistazo cual
# mecanismo produjo cada grafico/animacion en todo el notebook
COLOR_FUERZA_BRUTA = "tab:blue"     # seleccion optima por fuerza bruta (secciones 3, 6)
COLOR_QLEARNING = "tab:orange"      # politica aprendida por Q-Learning (seccion 5, 7)
COLOR_HUNGARO = "tab:green"         # asignacion optima centralizada (seccion 8)
COLOR_SUBASTA = "tab:purple"        # subasta secuencial descentralizada (seccion 8)

print("Modulo cargado:", hm.__file__)"""
))

cells.append(nbf.v4.new_markdown_cell("## 1. Construir el caso de ejemplo (agentes y bloques candidatos)"))

cells.append(nbf.v4.new_code_cell(
"""agentes, bloques = hm.construir_caso_ejemplo()
sistema = hm.SistemaMultiagente(agentes, bloques, T_max=7.0, lam=2.5)

for a in agentes:
    print(a.id, a.nombre, "duro:", a.ocupado_duro, "blando:", a.ocupado_blando)
print()
for b in bloques:
    print(b)"""
))

cells.append(nbf.v4.new_markdown_cell("## 2. Utilidad de cada bloque para cada agente participante"))

cells.append(nbf.v4.new_code_cell(
"""utilidades_bloques = sistema.evaluar_utilidades()
for b_id, utils in utilidades_bloques.items():
    legibles = ", ".join(f"{sistema.agentes[a].nombre}={u:.2f}" for a, u in utils.items())
    print(f"{b_id} ({sistema.bloques[b_id].nombre}): {legibles}")"""
))

cells.append(nbf.v4.new_markdown_cell("## 3. Seleccion optima: con equidad (lambda>0) vs sin equidad (lambda=0)"))

cells.append(nbf.v4.new_code_cell(
"""seleccion_lam, J_lam, util_lam = sistema.seleccionar_optimo(lam=sistema.lam)
seleccion_sin_lam, J_sin_lam, util_sin_lam = sistema.seleccionar_optimo(lam=0.0)

def resumen(nombre, seleccion, J, utilidades):
    print(f"=== {nombre} ===")
    print(f"J = {J:.3f}")
    for b in sorted(seleccion, key=lambda x: (x.dia, x.inicio)):
        print(" ", b)
    for a_id, u in utilidades.items():
        print(f"  {sistema.agentes[a_id].nombre}: {u:.2f}")
    var = statistics.pvariance(list(utilidades.values()))
    print(f"  Varianza: {var:.3f}\\n")

resumen(f"lambda={sistema.lam} (con equidad)", seleccion_lam, J_lam, util_lam)
resumen("lambda=0 (sin equidad)", seleccion_sin_lam, J_sin_lam, util_sin_lam)"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 4. Bonus: evento inesperado y replanificacion\\n\\n"
"Diego (A4) queda ocupado el jueves 14:00-16:00h a mitad de la coordinacion; "
"el sistema vuelve a resolver la seleccion optima respetando la nueva restriccion."
))

cells.append(nbf.v4.new_code_cell(
"""seleccion_repl, J_repl, util_repl = sistema.replanificar("A4", (3, 14, 16), lam=sistema.lam)
resumen("Replanificacion tras evento inesperado", seleccion_repl, J_repl, util_repl)"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 5. Q-Learning: aprendiendo la seleccion de bloques (Algoritmo 3)\\n\\n"
"En vez de enumerar por fuerza bruta los `2**n` subconjuntos de bloques, "
"`QLearningPlanner` trata la seleccion como un MDP episodico: recorre los "
"bloques en un orden fijo y en cada paso decide incluir o excluir el bloque "
"actual (solo si mantiene la seleccion factible). La recompensa es 0 en "
"cada paso intermedio y `J` (la utilidad social) solo al terminar el "
"episodio, asi que el valor de una decision temprana depende del retorno "
"futuro descontado -- la misma logica de la ecuacion de Bellman. Se entrena "
"con politica epsilon-greedy (epsilon decae de 1.0 a 0.05) y se compara, "
"como evidencia empirica, contra el optimo de fuerza bruta y contra una "
"linea base aleatoria, promediando sobre varias semillas."
))

cells.append(nbf.v4.new_code_cell(
"""SEMILLAS = list(range(10))
N_EPISODIOS = 8000

curvas, curvas_td, J_finales = [], [], []
for s in SEMILLAS:
    planner = hm.QLearningPlanner(sistema, alpha=0.15, gamma=0.95, lam=sistema.lam, semilla=s)
    historial = planner.entrenar(n_episodios=N_EPISODIOS)
    curvas.append(historial)
    curvas_td.append(planner.historial_td_error)
    _, J_final, _ = planner.mejor_politica()
    J_finales.append(J_final)

curvas = np.array(curvas)  # (n_semillas, n_episodios)
curvas_td = np.array(curvas_td)
media = curvas.mean(axis=0)
desv = curvas.std(axis=0)

# Linea base: seleccion aleatoria de bloques respetando factibilidad
import random as _random

def episodio_aleatorio(sistema, orden, rng):
    seleccion = frozenset()
    for bloque_id in orden:
        candidato = seleccion | {bloque_id}
        bloques_cand = [sistema.bloques[i] for i in candidato]
        if sistema.es_factible(bloques_cand) and rng.random() < 0.5:
            seleccion = candidato
    J, _ = sistema.utilidad_social([sistema.bloques[i] for i in seleccion], lam=sistema.lam)
    return J

rng_base = _random.Random(123)
orden = list(sistema.bloques.keys())
J_aleatorio = [episodio_aleatorio(sistema, orden, rng_base) for _ in range(2000)]
media_aleatorio = np.mean(J_aleatorio)

print(f"Q-Learning, J final: media={np.mean(J_finales):.2f} +/- {np.std(J_finales):.2f} "
      f"sobre {len(SEMILLAS)} semillas")
print(f"Linea base aleatoria, J promedio: {media_aleatorio:.2f}")
print(f"Optimo por fuerza bruta (lambda={sistema.lam}): {J_lam:.2f}")"""
))

cells.append(nbf.v4.new_code_cell(
"""episodios = np.arange(1, N_EPISODIOS + 1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(episodios, media, color=COLOR_QLEARNING, label="Q-Learning (media sobre 10 semillas)")
ax.fill_between(episodios, media - desv, media + desv, color=COLOR_QLEARNING, alpha=0.2,
                label="+/- 1 desviacion estandar")
ax.axhline(J_lam, color=COLOR_FUERZA_BRUTA, linestyle="--", label=f"Optimo (fuerza bruta) J={J_lam:.1f}")
ax.axhline(media_aleatorio, color="tab:red", linestyle=":", label=f"Linea base aleatoria J={media_aleatorio:.1f}")
ax.set_xlabel("Episodio")
ax.set_ylabel("J (utilidad social) del episodio")
ax.set_title("Convergencia de Q-Learning en la seleccion de bloques")
ax.legend()
fig.tight_layout()
ruta_convergencia = os.path.join(DIR_SALIDA, "qlearning_convergencia.png")
fig.savefig(ruta_convergencia, dpi=150)
print(f"Grafico guardado en {ruta_convergencia}")
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 5.1 Horario segun la politica aprendida (ultima semilla) vs. el optimo\\n\\n"
"Se corre la politica greedy aprendida (sin exploracion) y se compara su "
"horario final contra la seleccion optima de fuerza bruta."
))

cells.append(nbf.v4.new_code_cell(
"""seleccion_ql, J_ql, util_ql = planner.mejor_politica()
resumen("Politica aprendida por Q-Learning (ultima semilla)", seleccion_ql, J_ql, util_ql)
resumen(f"Optimo por fuerza bruta (lambda={sistema.lam})", seleccion_lam, J_lam, util_lam)"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 5.2 Panel de diagnostico del entrenamiento\\n\\n"
"Cuatro vistas complementarias del mismo entrenamiento: (a) como decae "
"epsilon (exploracion -> explotacion), (b) el error TD promedio por "
"episodio -- cuanto se \"sorprende\" el agente, que debe ir bajando si "
"esta aprendiendo, (c) que tan variable es el resultado final entre las 10 "
"semillas frente al optimo, y (d) un acercamiento al ultimo 20% de "
"episodios para ver si la curva ya se estabilizo."
))

cells.append(nbf.v4.new_code_cell(
"""def suavizar(x, ventana=100):
    x = np.asarray(x, dtype=float)
    if len(x) < ventana:
        return x
    kernel = np.ones(ventana) / ventana
    return np.convolve(x, kernel, mode="valid")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
ax.plot(episodios, planner.historial_epsilon, color="tab:purple")
ax.set_xlabel("Episodio")
ax.set_ylabel("epsilon")
ax.set_title("(a) Decaimiento de epsilon")

ax = axes[0, 1]
td_medio = curvas_td.mean(axis=0)
td_suave = suavizar(td_medio, ventana=100)
ax.plot(np.arange(len(td_suave)) + 100, td_suave, color="tab:orange")
ax.set_xlabel("Episodio")
ax.set_ylabel("|error TD| promedio (suavizado)")
ax.set_title("(b) Que tanto se sigue \\"sorprendiendo\\" el agente")

ax = axes[1, 0]
ax.bar([str(s) for s in SEMILLAS], J_finales, color=COLOR_QLEARNING)
ax.axhline(J_lam, color=COLOR_FUERZA_BRUTA, linestyle="--", label=f"Optimo J={J_lam:.1f}")
ax.set_xlabel("Semilla")
ax.set_ylabel("J de la politica final (greedy)")
ax.set_title("(c) Variabilidad del resultado final entre semillas")
ax.legend()

ax = axes[1, 1]
desde = int(N_EPISODIOS * 0.8)
ax.plot(episodios[desde:], media[desde:], color=COLOR_QLEARNING)
ax.fill_between(episodios[desde:], (media - desv)[desde:], (media + desv)[desde:],
                color=COLOR_QLEARNING, alpha=0.2)
ax.axhline(J_lam, color=COLOR_FUERZA_BRUTA, linestyle="--")
ax.set_xlabel("Episodio")
ax.set_ylabel("J")
ax.set_title("(d) Zoom: ultimo 20% de episodios")

fig.suptitle("Panel de diagnostico del entrenamiento de Q-Learning", fontsize=14)
fig.tight_layout()
ruta_panel = os.path.join(DIR_SALIDA, "qlearning_panel_diagnostico.png")
fig.savefig(ruta_panel, dpi=150)
print(f"Grafico guardado en {ruta_panel}")
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 5.3 Animacion del entrenamiento\\n\\n"
"Se anima como se construye, episodio a episodio, la curva de convergencia "
"(media +/- desviacion estandar sobre las 10 semillas) junto con el "
"decaimiento de epsilon, para ver el aprendizaje \"en movimiento\" en vez de "
"solo la foto final. Se guarda como `qlearning_entrenamiento.mp4`."
))

cells.append(nbf.v4.new_code_cell(
"""PASO_FRAME = 100  # cada cuantos episodios se agrega un frame nuevo
checkpoints = list(range(PASO_FRAME, N_EPISODIOS + 1, PASO_FRAME))
if checkpoints[-1] != N_EPISODIOS:
    checkpoints.append(N_EPISODIOS)

y_min = min(media_aleatorio, float(curvas.min())) - 5
y_max = J_lam + 10

fig, (ax_curva, ax_eps) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Entrenamiento de Q-Learning en vivo")

def dibujar_frame(k):
    ax_curva.clear()
    ax_eps.clear()

    hasta = checkpoints[k]
    ep = episodios[:hasta]

    ax_curva.plot(ep, media[:hasta], color=COLOR_QLEARNING, label="Q-Learning (media 10 semillas)")
    ax_curva.fill_between(ep, (media - desv)[:hasta], (media + desv)[:hasta],
                           color=COLOR_QLEARNING, alpha=0.2, label="+/- 1 desv. estandar")
    ax_curva.axhline(J_lam, color=COLOR_FUERZA_BRUTA, linestyle="--", label=f"Optimo J={J_lam:.1f}")
    ax_curva.axhline(media_aleatorio, color="tab:red", linestyle=":", label=f"Aleatorio J={media_aleatorio:.1f}")
    ax_curva.set_xlim(0, N_EPISODIOS)
    ax_curva.set_ylim(y_min, y_max)
    ax_curva.set_xlabel("Episodio")
    ax_curva.set_ylabel("J (utilidad social)")
    ax_curva.set_title(f"Episodio {hasta}/{N_EPISODIOS}")
    ax_curva.legend(loc="lower right", fontsize=8)

    ax_eps.plot(ep, planner.historial_epsilon[:hasta], color="tab:purple")
    ax_eps.set_xlim(0, N_EPISODIOS)
    ax_eps.set_ylim(0, 1.05)
    ax_eps.set_xlabel("Episodio")
    ax_eps.set_ylabel("epsilon")
    ax_eps.set_title("Exploracion (epsilon) a lo largo del entrenamiento")

n_frames = len(checkpoints)
# se repiten los ultimos cuadros para apreciar el resultado final unos segundos
frames = list(range(n_frames)) + [n_frames - 1] * 10

ani = animation.FuncAnimation(fig, dibujar_frame, frames=frames, interval=120, repeat=False)

writer = animation.FFMpegWriter(fps=10, bitrate=1800)
salida_entrenamiento_mp4 = os.path.join(DIR_SALIDA, "qlearning_entrenamiento.mp4")
ani.save(salida_entrenamiento_mp4, writer=writer)
plt.close(fig)
print(f"Animacion guardada en {salida_entrenamiento_mp4}")"""
))

cells.append(nbf.v4.new_code_cell(
"""Video(salida_entrenamiento_mp4, embed=True, width=900)"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 6. Animacion del horario final construyendose bloque a bloque\\n\\n"
"Se anima la seleccion optima (con equidad, antes del evento inesperado): "
"en cada cuadro se agrega un bloque mas al horario semanal y se actualiza "
"la utilidad acumulada de cada companero. Se guarda como `horario_simulacion.mp4`."
))

cells.append(nbf.v4.new_code_cell(
"""orden = sorted(seleccion_lam, key=lambda b: (b.dia, b.inicio))
colores = plt.cm.tab10.colors
color_por_bloque = {b.id: colores[i % len(colores)] for i, b in enumerate(orden)}
nombres_agentes = [sistema.agentes[a_id].nombre for a_id in sistema.agentes]
ids_agentes = list(sistema.agentes.keys())

fig, (ax_bar, ax_sched) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Construccion del horario optimo (lambda={:.1f})".format(sistema.lam))

def dibujar_frame(k):
    ax_bar.clear()
    ax_sched.clear()

    parciales = orden[:k]
    util_parcial = sistema.utilidad_por_agente(parciales)
    valores = [util_parcial[a_id] for a_id in ids_agentes]

    ax_bar.bar(nombres_agentes, valores, color=COLOR_FUERZA_BRUTA)
    ax_bar.set_ylim(0, max(util_lam.values()) * 1.15)
    ax_bar.set_ylabel("Utilidad acumulada U_i")
    ax_bar.set_title(f"Bloques asignados: {k}/{len(orden)}")

    for b in parciales:
        ax_sched.barh(y=DIAS[b.dia], width=b.duracion, left=b.inicio, height=0.5,
                       color=color_por_bloque[b.id], edgecolor="black")
        ax_sched.text(b.inicio + b.duracion / 2, DIAS[b.dia], b.id,
                       ha="center", va="center", fontsize=8)
    ax_sched.set_xlim(7, 18)
    ax_sched.set_yticks(range(len(DIAS)))
    ax_sched.set_yticklabels(DIAS)
    ax_sched.invert_yaxis()
    ax_sched.set_xlabel("Hora del dia")
    ax_sched.set_title("Horario semanal")

n_frames = len(orden) + 1
# se repiten los ultimos cuadros para poder apreciar el resultado final unos segundos
frames = list(range(n_frames)) + [n_frames - 1] * 8

ani = animation.FuncAnimation(fig, dibujar_frame, frames=frames, interval=600, repeat=False)

writer = animation.FFMpegWriter(fps=2, bitrate=1800)
salida_mp4 = os.path.join(DIR_SALIDA, "horario_simulacion.mp4")
ani.save(salida_mp4, writer=writer)
plt.close(fig)
print(f"Animacion guardada en {salida_mp4}")"""
))

cells.append(nbf.v4.new_code_cell(
"""Video(salida_mp4, embed=True, width=900)"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 7. Animacion del horario segun Q-Learning (maximo local) vs. el optimo\\n\\n"
"El video anterior anima el **optimo global** encontrado por fuerza bruta. "
"Aqui se anima, de la misma forma, la **politica aprendida por Q-Learning** "
"-- un maximo local: se queda corta frente al optimo, pero se puede ver en "
"que difiere: que bloques omite o cambia, y como eso se refleja en la "
"utilidad acumulada de cada companero. Se guarda como `horario_qlearning.mp4`."
))

cells.append(nbf.v4.new_code_cell(
"""orden_ql = sorted(seleccion_ql, key=lambda b: (b.dia, b.inicio))
color_por_bloque_ql = {b.id: colores[i % len(colores)] for i, b in enumerate(orden_ql)}
techo_util = max(max(util_lam.values()), max(util_ql.values())) * 1.15

fig, (ax_bar, ax_sched) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Construccion del horario segun Q-Learning (J={J_ql:.1f}, optimo J={J_lam:.1f})")

def dibujar_frame(k):
    ax_bar.clear()
    ax_sched.clear()

    parciales = orden_ql[:k]
    util_parcial = sistema.utilidad_por_agente(parciales)
    valores = [util_parcial[a_id] for a_id in ids_agentes]

    ax_bar.bar(nombres_agentes, valores, color=COLOR_QLEARNING)
    ax_bar.set_ylim(0, techo_util)
    ax_bar.set_ylabel("Utilidad acumulada U_i")
    ax_bar.set_title(f"Bloques asignados: {k}/{len(orden_ql)}")

    for b in parciales:
        ax_sched.barh(y=DIAS[b.dia], width=b.duracion, left=b.inicio, height=0.5,
                       color=color_por_bloque_ql[b.id], edgecolor="black")
        ax_sched.text(b.inicio + b.duracion / 2, DIAS[b.dia], b.id,
                       ha="center", va="center", fontsize=8)
    ax_sched.set_xlim(7, 18)
    ax_sched.set_yticks(range(len(DIAS)))
    ax_sched.set_yticklabels(DIAS)
    ax_sched.invert_yaxis()
    ax_sched.set_xlabel("Hora del dia")
    ax_sched.set_title("Horario semanal (Q-Learning)")

n_frames = len(orden_ql) + 1
frames = list(range(n_frames)) + [n_frames - 1] * 8

ani = animation.FuncAnimation(fig, dibujar_frame, frames=frames, interval=600, repeat=False)

writer = animation.FFMpegWriter(fps=2, bitrate=1800)
salida_ql_mp4 = os.path.join(DIR_SALIDA, "horario_qlearning.mp4")
ani.save(salida_ql_mp4, writer=writer)
plt.close(fig)
print(f"Animacion guardada en {salida_ql_mp4}")"""
))

cells.append(nbf.v4.new_code_cell(
"""Video(salida_ql_mp4, embed=True, width=900)"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 7.1 Diferencia entre ambas soluciones\\n\\n"
"Que bloques elige uno y no el otro."
))

cells.append(nbf.v4.new_code_cell(
"""ids_lam = {b.id for b in seleccion_lam}
ids_ql = {b.id for b in seleccion_ql}

print(f"Solo en el optimo (fuerza bruta):  {sorted(ids_lam - ids_ql)}")
print(f"Solo en Q-Learning:                {sorted(ids_ql - ids_lam)}")
print(f"En ambos:                          {sorted(ids_lam & ids_ql)}")
print(f"\\nJ optimo = {J_lam:.2f}   J Q-Learning = {J_ql:.2f}   "
      f"brecha = {J_lam - J_ql:.2f} ({(J_lam - J_ql) / J_lam:.1%} por debajo del optimo)")"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 8. Coordinacion por subasta: asignacion optima agente-tarea (Ejercicio 4)\\n\\n"
"Problema distinto al horario: en vez de elegir que subconjunto de bloques "
"*compartidos* entra al calendario, aqui cada **tarea** (p.ej. una entrega) "
"se le asigna a un solo **agente responsable**, y cada agente toma como "
"maximo una tarea -- el clasico *problema de asignacion*. Se resuelve de "
"dos formas, para comparar coordinacion centralizada vs. descentralizada:\\n\\n"
"- **`asignar_optimo()`**: algoritmo **hungaro** (Kuhn-Munkres) sobre la "
"matriz de utilidades agente x tarea -- el optimo global, requiere ver el "
"problema completo.\\n"
"- **`asignar_por_subasta()`**: **subasta secuencial** -- las tareas se "
"rematan una por una y se las lleva quien puje mas alto (su propia "
"utilidad); ningun agente ve el problema completo."
))

cells.append(nbf.v4.new_code_cell(
"""tareas = hm.construir_caso_asignacion(agentes)
asignador = hm.AsignadorSubasta(agentes, tareas)

asignacion_opt, utilidad_opt = asignador.asignar_optimo()
orden_subasta = [t.id for t in tareas]
asignacion_sub, utilidad_sub = asignador.asignar_por_subasta(orden=orden_subasta, semilla=0)

print("=== Asignacion optima (algoritmo hungaro, centralizado) ===")
for a_id, t_id in sorted(asignacion_opt.items()):
    print(f"  {sistema.agentes[a_id].nombre} ({a_id}) -> {t_id}")
print(f"Utilidad total = {utilidad_opt:.2f}\\n")

print("=== Asignacion por subasta secuencial (descentralizado) ===")
for a_id, t_id in sorted(asignacion_sub.items()):
    print(f"  {sistema.agentes[a_id].nombre} ({a_id}) -> {t_id}")
print(f"Utilidad total = {utilidad_sub:.2f}\\n")

print(f"Brecha subasta vs. optimo = {utilidad_opt - utilidad_sub:.2f}")"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 8.1 Comparacion de utilidad por agente\\n\\n"
"Utilidad que obtiene cada agente con la tarea que le toco, optimo (verde) "
"vs subasta (morado) -- colores usados en el resto de esta seccion para "
"distinguirla de la seccion 3/6 (fuerza bruta, azul) y de la seccion 5/7 "
"(Q-Learning, naranja)."
))

cells.append(nbf.v4.new_code_cell(
"""ids_agentes_asig = list(sistema.agentes.keys())
nombres_asig = [sistema.agentes[a_id].nombre for a_id in ids_agentes_asig]
tareas_por_id = {t.id: t for t in tareas}

def utilidad_por_agente_asignacion(asignacion):
    resultado = {a_id: 0.0 for a_id in ids_agentes_asig}
    for a_id, t_id in asignacion.items():
        resultado[a_id] = sistema.agentes[a_id].calcular_utilidad(tareas_por_id[t_id])
    return resultado

util_opt_agente = utilidad_por_agente_asignacion(asignacion_opt)
util_sub_agente = utilidad_por_agente_asignacion(asignacion_sub)

fig, ax = plt.subplots(figsize=(8, 5))
x = range(len(ids_agentes_asig))
ancho = 0.35
ax.bar([i - ancho / 2 for i in x], [util_opt_agente[a] for a in ids_agentes_asig],
       width=ancho, color=COLOR_HUNGARO, label=f"Optimo (hungaro) J={utilidad_opt:.1f}")
ax.bar([i + ancho / 2 for i in x], [util_sub_agente[a] for a in ids_agentes_asig],
       width=ancho, color=COLOR_SUBASTA, label=f"Subasta secuencial J={utilidad_sub:.1f}")
ax.set_xticks(list(x))
ax.set_xticklabels(nombres_asig)
ax.set_ylabel("Utilidad de la tarea asignada")
ax.set_title("Asignacion optima (hungaro) vs. subasta secuencial (descentralizada)")
ax.legend()
fig.tight_layout()
ruta_subasta_comp = os.path.join(DIR_SALIDA, "subasta_comparacion.png")
fig.savefig(ruta_subasta_comp, dpi=150)
print(f"Grafico guardado en {ruta_subasta_comp}")
plt.show()"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 8.2 Animacion de la subasta secuencial\\n\\n"
"Se anima el remate ronda a ronda: en cada cuadro se resuelve una tarea "
"mas (gana quien puje mas alto) y se actualiza tanto la utilidad ganada "
"por cada agente como el acumulado total, comparado contra la linea "
"punteada del optimo centralizado (hungaro). Se guarda como "
"`subasta_secuencial.mp4`."
))

cells.append(nbf.v4.new_code_cell(
"""libres = {a.id: a for a in agentes}
historial_rondas = []  # (tarea_id, ganador_id o None, puja, utilidad_acumulada)
acumulado = 0.0
for tarea_id in orden_subasta:
    tarea = tareas_por_id[tarea_id]
    pujas = sorted(
        ((a.calcular_utilidad(tarea), a.id) for a in libres.values() if a.id in tarea.participantes),
        key=lambda pu: (-pu[0], pu[1]),
    )
    if pujas and pujas[0][0] > 0:
        mejor_puja, ganador_id = pujas[0]
        acumulado += mejor_puja
        del libres[ganador_id]
        historial_rondas.append((tarea_id, ganador_id, mejor_puja, acumulado))
    else:
        historial_rondas.append((tarea_id, None, 0.0, acumulado))

fig, (ax_bar, ax_total) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Subasta secuencial: remate de tareas una por una")

def dibujar_frame(k):
    ax_bar.clear()
    ax_total.clear()

    utils_parciales = {a_id: 0.0 for a_id in ids_agentes_asig}
    for tarea_id, ganador_id, puja, _ in historial_rondas[:k]:
        if ganador_id is not None:
            utils_parciales[ganador_id] = puja

    valores = [utils_parciales[a_id] for a_id in ids_agentes_asig]
    ax_bar.bar(nombres_asig, valores, color=COLOR_SUBASTA)
    ax_bar.set_ylim(0, max(util_opt_agente.values()) * 1.3)
    ax_bar.set_ylabel("Utilidad de la tarea ganada")
    if k == 0:
        titulo_ronda = "Antes de empezar"
    else:
        tarea_id, ganador_id, puja, _ = historial_rondas[k - 1]
        nombre_ganador = sistema.agentes[ganador_id].nombre if ganador_id else "nadie (puja <= 0)"
        titulo_ronda = f"Remate {k}/{len(historial_rondas)}: {tarea_id} -> {nombre_ganador}"
    ax_bar.set_title(titulo_ronda)

    rondas_x = list(range(len(historial_rondas) + 1))
    acumulados = [0.0] + [r[3] for r in historial_rondas]
    ax_total.plot(rondas_x[:k + 1], acumulados[:k + 1], color=COLOR_SUBASTA, marker="o",
                  label="Subasta (acumulado)")
    ax_total.axhline(utilidad_opt, color=COLOR_HUNGARO, linestyle="--",
                      label=f"Optimo (hungaro) J={utilidad_opt:.1f}")
    ax_total.set_xlim(0, len(historial_rondas))
    ax_total.set_ylim(0, utilidad_opt * 1.15)
    ax_total.set_xlabel("Ronda de remate")
    ax_total.set_ylabel("Utilidad total acumulada")
    ax_total.set_title("Utilidad total: subasta vs. optimo centralizado")
    ax_total.legend(loc="lower right")

n_frames = len(historial_rondas) + 1
frames = list(range(n_frames)) + [n_frames - 1] * 8

ani = animation.FuncAnimation(fig, dibujar_frame, frames=frames, interval=700, repeat=False)

writer = animation.FFMpegWriter(fps=2, bitrate=1800)
salida_subasta_mp4 = os.path.join(DIR_SALIDA, "subasta_secuencial.mp4")
ani.save(salida_subasta_mp4, writer=writer)
plt.close(fig)
print(f"Animacion guardada en {salida_subasta_mp4}")"""
))

cells.append(nbf.v4.new_code_cell(
"""Video(salida_subasta_mp4, embed=True, width=900)"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 8.3 Diferencia entre ambas asignaciones\\n\\n"
"En que agentes coincide la tarea asignada, y en cuales no."
))

cells.append(nbf.v4.new_code_cell(
"""comunes = [a_id for a_id in ids_agentes_asig
           if a_id in asignacion_opt and asignacion_opt.get(a_id) == asignacion_sub.get(a_id)]
print(f"Agentes con la misma tarea en ambos mecanismos: {sorted(comunes)}")
for a_id in ids_agentes_asig:
    if a_id not in comunes:
        print(f"  {sistema.agentes[a_id].nombre}: optimo={asignacion_opt.get(a_id, '-')}  "
              f"subasta={asignacion_sub.get(a_id, '-')}")
print(f"\\nJ optimo = {utilidad_opt:.2f}   J subasta = {utilidad_sub:.2f}   "
      f"brecha = {utilidad_opt - utilidad_sub:.2f} "
      f"({(utilidad_opt - utilidad_sub) / utilidad_opt:.1%} por debajo del optimo)")"""
))

cells.append(nbf.v4.new_markdown_cell(
"## 9. Negociacion y equilibrio de Nash\\n\\n"
"Un tercer mecanismo de coordinacion, distinto al horario optimo (un solo "
"planificador central) y a la subasta (un remate secuencial): aqui **no "
"hay coordinador en absoluto**. Dos agentes deciden si `Cooperar` en una "
"tarea compartida (repartirse la carga) o `No_cooperar` (dejarle todo al "
"otro), cada uno anticipando lo que hara el otro. `JuegoDosAgentes` "
"encuentra los equilibrios de Nash en estrategias puras (mejor respuesta "
"mutua) y los optimos de Pareto, para contrastar **estabilidad** "
"(Nash) contra **eficiencia social** (Pareto) -- no siempre coinciden."
))

cells.append(nbf.v4.new_code_cell(
"""juego = hm.construir_caso_negociacion("A1", "A3")
nombre1 = sistema.agentes[juego.agente1_id].nombre
nombre2 = sistema.agentes[juego.agente2_id].nombre

print("Matriz de pagos (U1, U2):")
for e1 in juego.estrategias1:
    fila = "  ".join(f"{e2}={juego.pagos[(e1, e2)]}" for e2 in juego.estrategias2)
    print(f"  {e1}: {fila}")

dom1 = juego.estrategia_dominante(1)
dom2 = juego.estrategia_dominante(2)
print(f"\\nEstrategia dominante de {nombre1}: {dom1 or 'ninguna'}")
print(f"Estrategia dominante de {nombre2}: {dom2 or 'ninguna'}")

equilibrios = juego.equilibrios_nash_puros()
optimos = juego.optimos_pareto()
print(f"Equilibrios de Nash (estrategias puras): {equilibrios}")
print(f"Optimos de Pareto: {optimos}")

if equilibrios and optimos and set(equilibrios) != set(optimos):
    print("\\nEl equilibrio de Nash NO coincide con el optimo de Pareto: "
          "el resultado estable no es el socialmente mas eficiente.")"""
))

cells.append(nbf.v4.new_markdown_cell(
"### 9.1 Matriz de pagos\\n\\n"
"Cada celda muestra (U1, U2). La celda resaltada en verde es el equilibrio "
"de Nash: el unico resultado del que ningun agente quiere desviarse "
"unilateralmente, aunque `(Cooperar, Cooperar)` -- en blanco, tambien "
"Pareto-optimo -- sea mejor para ambos."
))

cells.append(nbf.v4.new_code_cell(
"""COLOR_NASH = "tab:green"

fig, ax = plt.subplots(figsize=(6, 5))
n1, n2 = len(juego.estrategias1), len(juego.estrategias2)
ax.set_xlim(0, n2)
ax.set_ylim(0, n1)

for i, e1 in enumerate(juego.estrategias1):
    for j, e2 in enumerate(juego.estrategias2):
        u1, u2 = juego.pagos[(e1, e2)]
        y = n1 - 1 - i
        es_nash = (e1, e2) in equilibrios
        es_pareto = (e1, e2) in optimos
        color = COLOR_NASH if es_nash else ("white" if es_pareto else "#dddddd")
        ax.add_patch(plt.Rectangle((j, y), 1, 1, facecolor=color, edgecolor="black", linewidth=1.5))
        marca = " (Nash)" if es_nash else ("" if es_pareto else " (dominado)")
        ax.text(j + 0.5, y + 0.5, f"({u1:.0f}, {u2:.0f}){marca}", ha="center", va="center", fontsize=11)

ax.set_xticks([j + 0.5 for j in range(n2)])
ax.set_xticklabels(juego.estrategias2)
ax.set_yticks([n1 - 1 - i + 0.5 for i in range(n1)])
ax.set_yticklabels(juego.estrategias1)
ax.set_xlabel(f"{nombre2} ({juego.agente2_id})")
ax.set_ylabel(f"{nombre1} ({juego.agente1_id})")
ax.set_title("Negociacion sobre tarea compartida: matriz de pagos")
fig.tight_layout()
ruta_nash = os.path.join(DIR_SALIDA, "nash_matriz_pagos.png")
fig.savefig(ruta_nash, dpi=150)
print(f"Grafico guardado en {ruta_nash}")
plt.show()"""
))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("horarios_simulacion.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook creado: horarios_simulacion.ipynb")
