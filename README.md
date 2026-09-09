# Como correr el codigo

## 1. Instalar dependencias

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

## 2. Correr la simulacion

```bash
python horarios_multiagente.py
```

Esto imprime los resultados en consola y guarda `horario_resultado.png`.

## 3. Correr el notebook

El notebook `horarios_simulacion.ipynb` se genera con `build_notebook.py` (no se edita a mano):

```bash
python build_notebook.py
```

Para ejecutarlo completo desde la terminal (genera `horario_simulacion.mp4`, `qlearning_entrenamiento.mp4` y las imagenes `.png`):

```bash
jupyter nbconvert --to notebook --execute --inplace horarios_simulacion.ipynb
```

O abrirlo en Jupyter/VS Code y correr las celdas manualmente.
