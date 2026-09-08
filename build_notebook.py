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
"## 5. Animacion del horario final construyendose bloque a bloque\\n\\n"
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

    ax_bar.bar(nombres_agentes, valores, color="tab:blue")
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
salida_mp4 = "horario_simulacion.mp4"
ani.save(salida_mp4, writer=writer)
plt.close(fig)
print(f"Animacion guardada en {salida_mp4}")"""
))

cells.append(nbf.v4.new_code_cell(
"""Video(salida_mp4, embed=True, width=900)"""
))

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("horarios_simulacion.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Notebook creado: horarios_simulacion.ipynb")
