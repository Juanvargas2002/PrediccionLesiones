# Predicción de Riesgo de Lesión Deportiva - Trabajo Corte 2

Este es nuestro trabajo del corte 2. Hicimos una aplicación web en Flask que estima la
probabilidad de lesión de un deportista a partir de 30 datos (biométricos y de
entrenamiento). Incluye todo el proceso: el entrenamiento de la red neuronal y la página
donde uno carga los valores y ve el resultado en pantalla (porcentaje, nivel de riesgo
y si está "En riesgo" o "No en riesgo").

## Estructura de carpetas (como la dejamos al final)

```
Simulalcion/
├── app.py
├── features.json
├── modelo_lesion_deportiva.keras
├── scaler.pkl
├── requirements.txt
├── README.md
├── templates/
│   └── form.html
├── static/
│   └── style.css
└── Crear_Modelo/
    ├── Riesgo_de_Lesion_Deportiva.ipynb
    ├── sports_multimodal_data.csv
    ├── modelo_lesion_deportiva.keras
    └── scaler.pkl
```

Nosotros lo ordenamos así para no mezclar: en la raíz quedó solo lo que necesita la web
para funcionar, y en `Crear_Modelo/` dejamos lo del entrenamiento (el notebook donde
entrenamos y probamos el modelo, el dataset y las copias originales del modelo y el scaler).

## Para qué sirve cada cosa

En la raíz (lo que usa la web):

- `app.py`: el servidor en Flask. Ahí están las rutas, la validación de los datos y la
  predicción.
- `features.json`: acá guardamos las 30 variables con sus rangos, unidades y valores por
  defecto, más el umbral (0.3) y los niveles de riesgo. La página arma el formulario
  leyendo este archivo.
- `modelo_lesion_deportiva.keras`: la red ya entrenada. Recibe 30 valores y devuelve
  un número entre 0 y 1.
- `scaler.pkl`: el StandardScaler que usamos en el entrenamiento. Hay que usar el mismo
  en la web para que los datos le lleguen al modelo igual que cuando lo entrenamos.
- `templates/form.html`: el formulario. Lo hicimos dinámico, o sea que genera los 30
  campos recorriendo el `features.json`.
- `static/style.css`: los estilos de la página.
- `requirements.txt`: lo necesario para instalar (`tensorflow`, `flask`, `numpy`,
  `scikit-learn`, `pandas`).

En `Crear_Modelo/` (no lo usa la web, es respaldo del entrenamiento):

- `Riesgo_de_Lesion_Deportiva.ipynb`: el notebook donde entrenamos y probamos el modelo.
- `sports_multimodal_data.csv`: el dataset que usamos (5430 filas, 30 columnas + `injury_risk`).
- `modelo_lesion_deportiva.keras` y `scaler.pkl`: las copias originales. Las copiamos a
  la raíz para que `app.py` las tenga a mano.

## Cómo ejecutarlo

1. Instalar las dependencias:

```bash
pip install -r requirements.txt
```

2. Correr el servidor:

```bash
python app.py
```

3. Abrir en el navegador: http://127.0.0.1:5000

Aclaración: en nuestra compu el comando `python` estaba roto,
así que lo corrimos con la ruta completa del Python 3.13:

```
C:\Users\juan2\AppData\Local\Python\bin\python.exe app.py
```
otra opción fue ejecutar el siguiente comando

```bash
py app.py
```

Si les pasa lo mismo, usen la ruta completa de su Python o prueben con la version con `py`.

## Cómo funciona

Cuando se entra a `/`, el servidor lee el `features.json` y muestra el formulario con
los 30 campos. Nosotros lo dejamos precargado con los valores promedio del dataset,
así arranca con un "deportista promedio" y solo hay que cambiar lo que se quiera probar.

Cuando se da clic en "Calcular riesgo" (POST a `/predecir`), el `app.py` hace esto:

1. Valida que cada campo sea un número y que esté dentro de su rango `[min, max]`.
   Si algo está mal, vuelve al formulario con el mensaje de error.
2. Arma el vector en el mismo orden en que se entrenó el scaler
   (`scaler.feature_names_in_`), porque si cambia el orden la predicción da mal.
3. Lo pasa por el `scaler.transform()`.
4. Lo pasa por el modelo con `model.predict()` y obtiene la probabilidad de 0 a 1.
5. Clasifica el resultado:
   - Umbral 0.3: si da 0.3 o más es "En riesgo", si da menos es "No en riesgo".
     Elegimos 0.3 y no 0.5 porque el dataset está muy desbalanceado (casi no hay
     lesionados) y preferimos que avise de más a que se le pase un caso.
   - Nivel: Bajo de 0 a 0.25, Moderado de 0.25 a 0.6, Alto de 0.6 a 1.

El modelo y el scaler se cargan una sola vez cuando arranca el servidor, no en cada
pedido.

## Pruebas que hicimos nosotros

Probamos la página a mano y notamos que andaba bien:

- El modelo carga con entrada de 30 valores y el scaler trae las mismas 30 features
  en el mismo orden que el `features.json`.
- Con los valores por defecto (el deportista promedio) da alrededor de 3.1% o 0.031, Nivel Bajo.
- Con un perfil exigido (fatiga alta, poco descanso, intensidad alta, con historial de
  lesión, etc.) nos dio alrededor de 85%, Nivel Alto y "En riesgo", que era lo esperado.
- Si falta un campo, tira error avisando cuál falta.
- Si se pone algo fuera de rango (por ejemplo FC en 999) o un texto en vez de número,
  también tira error y no deja predecir.

## Cosas que notamos y dejamos anotadas

- El dataset tiene pocos lesionados (como el 5% de las filas), por eso el umbral bajo
  de 0.3. Si usábamos 0.5 casi todo daba "No en riesgo".
- Hay columnas con valores negativos en el propio dataset (por ejemplo `jump_height`
  o `training_duration`), así que los rangos del formulario también los aceptan.
  Nosotros al principio pensamos que era un error, pero vimos que el modelo se entrenó
  así y lo dejamos igual.
- Al cargar el `scaler.pkl` sale un aviso de versión de scikit-learn (se guardó con
  una versión y lo corrimos con otra más nueva). Es solo un warning, el escalado funciona igual.
- Los rangos del `features.json` los sacamos de los min/max reales del CSV, así el
  modelo nunca recibe valores que no vio en el entrenamiento.

## Aclaración de uso

Esto es un trabajo de la universidad, no es una herramienta médica. Sirve para mostrar
cómo se conecta un modelo entrenado con una página web, pero no reemplaza la evaluación
de un profesional.
