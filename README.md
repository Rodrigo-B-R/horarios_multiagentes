# Como correr el proyecto (en orden)

Este proyecto se construyo en capas: cada paso agrega un mecanismo de
coordinacion nuevo sobre el mismo caso base (4 agentes, 8 bloques de
horario). Correr los pasos en este orden es la forma mas facil de entender
la progresion completa.

## 0. Instalar dependencias

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```
Linux/macOS:
```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## 1. `python horarios_multiagente.py` — el pipeline completo en consola

```bash
python horarios_multiagente.py
```

Imprime, en orden, **todos** los mecanismos implementados sobre el mismo
caso de ejemplo, y guarda `horario_resultado.png`:

1. **Utilidad** de cada bloque para cada agente (`Agente.calcular_utilidad`).
2. **Horario optimo** por fuerza bruta, con y sin penalizacion por equidad
   (`SistemaMultiagente.seleccionar_optimo`).
3. **Replanificacion** ante un evento inesperado (`replanificar`).
4. **Q-Learning** como alternativa a la fuerza bruta (`QLearningPlanner`).
5. **Asignacion por subasta**: optimo centralizado (algoritmo hungaro) vs.
   subasta secuencial descentralizada (`AsignadorSubasta`).
6. **Negociacion y equilibrio de Nash** entre dos agentes que deciden
   cooperar o no en una tarea compartida (`JuegoDosAgentes`).

Es el punto de entrada mas rapido para ver que hace cada pieza, sin
graficas ni animaciones.

## 2. `horarios_simulacion.ipynb` — la misma progresion, con graficas y animaciones

El notebook se genera desde `build_notebook.py` (no se edita a mano):

```bash
python build_notebook.py
```

Sus secciones siguen la misma progresion que el script, pero visualizada:

| # | Seccion | Mecanismo |
|---|---|---|
| 1–3 | Caso de ejemplo, utilidades, seleccion optima | Fuerza bruta |
| 4 | Replanificacion ante evento inesperado | — |
| 5 | Convergencia de Q-Learning (10 semillas) + panel de diagnostico | Q-Learning |
| 6–7 | Animacion del horario: optimo vs. Q-Learning | Fuerza bruta / Q-Learning |
| 8 | Asignacion optima (hungaro) vs. subasta secuencial + animacion | Subasta |
| 9 | Matriz de pagos y equilibrio de Nash | Negociacion |

Para ejecutarlo completo desde la terminal:

```bash
jupyter nbconvert --to notebook --execute --inplace horarios_simulacion.ipynb
```

> **Nota (Windows):** las secciones con animacion (`.mp4`, via `FFMpegWriter`)
> pueden colgar `jupyter nbconvert` indefinidamente en algunos entornos
> Windows por un problema conocido entre el pipe de ffmpeg y el kernel de
> Jupyter sobre `zmq`/asyncio. Si eso pasa, es mas confiable **abrir el
> notebook en Jupyter o VS Code y correr las celdas ahi** (Run All) en vez
> de ejecutarlo sin interfaz.

Todas las imagenes y videos se guardan en `representaciones_graficas/`.

## 3. `demostracion_cientifica.ipynb` — validar que los mecanismos realmente funcionan

Correr el sistema una vez y ver que "se ve bien" no es evidencia cientifica.
Este notebook (generado por `build_notebook_cientifico.py`, tampoco se edita
a mano) aplica el ciclo de validacion cientifica -- pregunta, hipotesis
falsable, diseno experimental, replicas con semillas, prueba *t* + tamano
del efecto (*d* de Cohen), conclusion -- a dos comparaciones:

1. **Subasta secuencial vs. asignacion aleatoria** (30 semillas).
2. **Q-Learning vs. politica aleatoria** sobre el problema de horario (20 semillas).

```bash
python build_notebook_cientifico.py
jupyter nbconvert --to notebook --execute --inplace demostracion_cientifica.ipynb
```

Esta notebook no tiene animaciones, asi que `nbconvert` sin interfaz
funciona sin el problema mencionado arriba.

## Resumen de la progresion

```
Utilidad + restricciones          (horarios_multiagente.py: Agente, SistemaMultiagente)
        |
Optimo centralizado (fuerza bruta) -> Q-Learning (aprendizaje)
        |
Asignacion optima (hungaro) -> Subasta secuencial (coordinacion descentralizada)
        |
Negociacion y equilibrio de Nash  (sin coordinador: cada agente decide solo)
        |
Demostracion cientifica            (los mecanismos anteriores, puestos a prueba con datos)
```

Los `.mp4`/`.png` generados en `representaciones_graficas/` y los dos
notebooks son artefactos versionados; los `.py` (`horarios_multiagente.py`,
`build_notebook.py`, `build_notebook_cientifico.py`) son la fuente de verdad.
