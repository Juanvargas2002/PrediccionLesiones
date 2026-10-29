import os

# Optimizaciones de memoria para Render (Free Tier: 512MB RAM)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import pickle
import joblib
from datetime import datetime
import numpy as np
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from groq import Groq
from dotenv import load_dotenv

app = Flask(__name__)

# =======================================================
# 1. CARGA DE MODELO, SCALER Y CONFIGURACIÓN DE FEATURES
# =======================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "modelo_lesion_deportiva.keras")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "features.json")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_logs.json")
ENV_PATH = os.path.join(BASE_DIR, ".env")

# Cargar variables de entorno explícitamente desde .env en la misma carpeta
load_dotenv(ENV_PATH)

groq_api_key = os.environ.get("GROQ_API_KEY", "").strip()
if groq_api_key:
    print(f"[*] GROQ_API_KEY detectada: {groq_api_key[:8]}... (longitud {len(groq_api_key)})")
else:
    print("[!] ALERTA: No se encontró GROQ_API_KEY en el archivo .env")

# Cargar configuración desde features.json
with open(FEATURES_PATH, "r", encoding="utf-8") as f:
    features_config = json.load(f)

features_list = features_config.get("features", [])
decision_threshold = float(features_config.get("decision_threshold", 0.58))
risk_levels = features_config.get("risk_levels", [])

# Cargar modelo Keras y Scaler
modelo = tf.keras.models.load_model(MODEL_PATH)

try:
    scaler = joblib.load(SCALER_PATH)
except Exception:
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

# Inicializar cliente de Groq
groq_client = Groq(api_key=groq_api_key)

# Modelos disponibles (openai/gpt-oss-120b como modelo principal por defecto)
MODELOS_GROQ = [
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b"
]


# =======================================================
# 2. FUNCIÓN DE RECOMENDACIÓN CON GROQ (> 200 CARACT. Y <= 400 PALABRAS)
# =======================================================
def generar_recomendacion_groq(datos_atleta: dict, probabilidad: float, nivel_riesgo: str) -> tuple[str, str]:
    """
    Envía la predicción y el perfil biomecánico a Groq usando 'openai/gpt-oss-120b' (on_demand).
    Genera un plan preventivo en prosa natural y fluida (sin tablas, sin listas ni líneas separadoras).
    Asegura una extensión mayor a 200 caracteres y un máximo estricto de 400 palabras.
    """
    porcentaje = round(probabilidad * 100, 2)

    system_prompt = (
        "Eres un médico deportólogo, fisioterapeuta y preparador físico de alto rendimiento. "
        "Tu tarea es comunicar de forma humana, clara y cercana un plan de recomendaciones preventivas al atleta. "
        "REGLAS OBLIGATORIAS DE FORMATO: "
        "1. Escribe EXCLUSIVAMENTE en texto corrido y prosa natural (párrafos continuos y fluidos). "
        "2. Está ESTRICTAMENTE PROHIBIDO usar tablas, líneas divisorias ('---' o '___'), viñetas, guiones de lista o caracteres decorativos. "
        "3. La respuesta debe superar obligatoriamente los 200 caracteres de longitud y NO DEBE SUPERAR NUNCA las 400 palabras. "
        "Habla en un tono profesional, empático y directo."
    )

    detalles_atleta = ", ".join([f"{k}: {v}" for k, v in datos_atleta.items()])

    user_prompt = f"""
El modelo de inteligencia artificial ha estimado una probabilidad de lesión del {porcentaje}% con un nivel de riesgo clasificado como '{nivel_riesgo}' (umbral: {decision_threshold}).
Parámetros biomecánicos registrados del deportista: {detalles_atleta}.

Con base en estos datos, redacta tu recomendación preventiva en dos o tres párrafos de texto continuo y natural.
Explica cómo debe ajustar las cargas de entrenamiento, qué medidas de descanso y recuperación muscular debe tomar, y qué precauciones articulares son prioritarias.
Recuerda: solo prosa natural, sin tablas, sin listas con viñetas ni líneas divisorias. Máximo 400 palabras y más de 200 caracteres.
""".strip()

    prompt_completo = f"SYSTEM:\n{system_prompt}\n\nUSER:\n{user_prompt}"

    ultimo_error = None
    for modelo_nombre in MODELOS_GROQ:
        try:
            print(f"[*] Solicitando recomendación a Groq (Modelo: {modelo_nombre} - on_demand)...")
            chat_completion = groq_client.chat.completions.create(
                model=modelo_nombre,
                service_tier="on_demand",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                max_tokens=600
            )
            recomendacion = chat_completion.choices[0].message.content.strip()

            # Limpiar posibles líneas divisorias o artefactos de tablas si se generaron
            lineas = [l for l in recomendacion.splitlines() if not l.strip().startswith(("---", "___", "===", "|"))]
            recomendacion = "\n\n".join([l.strip() for l in "\n".join(lineas).split("\n\n") if l.strip()])

            # Asegurar longitud mínima (> 200 caracteres) en texto natural
            if len(recomendacion) < 200:
                recomendacion += (
                    f" Considerando que el riesgo estimado es del {porcentaje}%, es fundamental reducir "
                    "inmediatamente la intensidad general de los entrenamientos en los próximos dos días, "
                    "priorizar una adecuada hidratación junto con descanso nocturno y solicitar una valoración "
                    "fisioterapéutica preventiva para asegurar la salud de los tejidos conectivos."
                )

            # Asegurar límite máximo de 400 palabras
            palabras = recomendacion.split()
            if len(palabras) > 400:
                # Recortar en 400 palabras buscando el último punto
                texto_recortado = " ".join(palabras[:400])
                ultimo_punto = texto_recortado.rfind(".")
                if ultimo_punto != -1 and ultimo_punto > 200:
                    recomendacion = texto_recortado[:ultimo_punto + 1]
                else:
                    recomendacion = texto_recortado + "..."

            print(f"[+] Recomendación generada con éxito ({len(recomendacion.split())} palabras, {len(recomendacion)} caracteres).")
            return recomendacion, prompt_completo

        except Exception as err:
            ultimo_error = err
            print(f"[!] Falló modelo '{modelo_nombre}': {err}. Intentando alternativa...")

    print(f"[-] Todos los modelos de Groq fallaron. Último error: {ultimo_error}")
    fallback_recomendacion = (
        f"El modelo ha estimado una probabilidad de lesión del {porcentaje}% ({nivel_riesgo}). Como medida "
        "preventiva inmediata, te aconsejamos reducir la carga general de entrenamiento en un 25% durante los "
        "próximos días y poner especial atención al calentamiento dinámico articular. Asegura un descanso diario "
        "de entre siete y ocho horas para facilitar la recuperación neuromuscular e hidrátate de manera continua. "
        "Si experimentas sobrecarga persistente o molestia localizada, acude a valoración fisioterapéutica antes de "
        "retomar sesiones de alta exigencia."
    )
    return fallback_recomendacion, prompt_completo


def calcular_nivel_riesgo(probabilidad: float) -> str:
    """Determina la categoría de riesgo a partir de risk_levels en features.json"""
    for nivel in risk_levels:
        if nivel["min"] <= probabilidad <= nivel["max"]:
            return nivel["name"]
    return "Alto" if probabilidad >= decision_threshold else "Bajo"


# =======================================================
# 3. RUTAS FLASK (VISTA, PREDICCIÓN Y RLHF)
# =======================================================
@app.route("/", methods=["GET"])
def index():
    return render_template("form.html", features=features_list, valores=None, resultado=None)


@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        datos_formulario = {}
        vector_entrada = []

        # Extraer parámetros en el orden exacto esperado por el scaler y modelo
        for f in features_list:
            nombre = f["name"]
            valor_raw = request.form.get(nombre, f.get("default", 0.0))
            valor = float(valor_raw)
            datos_formulario[nombre] = valor
            vector_entrada.append(valor)

        # 1. Transformar datos con el scaler
        matriz_entrada = np.array([vector_entrada])
        datos_escalados = scaler.transform(matriz_entrada)

        # 2. Inferencia con el modelo de Keras
        prediccion_raw = modelo.predict(datos_escalados)
        probabilidad = float(prediccion_raw[0][0])
        porcentaje = round(probabilidad * 100, 2)

        # 3. Clasificación según umbral y niveles
        nivel = calcular_nivel_riesgo(probabilidad)
        clasificacion = "en riesgo" if probabilidad >= decision_threshold else "no en riesgo"

        # 4. Generación de recomendación preventiva con Groq
        recomendacion, prompt_original = generar_recomendacion_groq(datos_formulario, probabilidad, nivel)

        resultado = {
            "probabilidad": porcentaje,
            "nivel": nivel,
            "umbral": decision_threshold,
            "clasificacion": clasificacion,
            "recomendacion": recomendacion,
            "prompt_original": prompt_original
        }

        # Respuesta JSON para peticiones AJAX/Fetch
        if request.headers.get("X-Requested-With") == "XMLHttpRequest" or request.is_json:
            return jsonify({
                "resultado": resultado,
                "valores": datos_formulario
            })

        # Renderizar en la plantilla form.html
        return render_template(
            "form.html",
            features=features_list,
            valores=request.form,
            resultado=resultado
        )

    except Exception as e:
        return render_template(
            "form.html",
            features=features_list,
            valores=request.form,
            error=f"Error durante el procesamiento: {str(e)}"
        ), 500


@app.route("/feedback", methods=["POST"])
def feedback():
    """
    Ruta para capturar retroalimentación humana (RLHF).
    Recibe la calificación del usuario (ej. 1 a 5 o positivo/negativo),
    comentario, el prompt original y la respuesta generada por el LLM.
    Guarda cada nuevo feedback como un objeto en 'feedback_logs.json' sin sobrescribir los anteriores.
    """
    try:
        # Permite recepción vía JSON (fetch) o desde un formulario estándar
        data = request.get_json(silent=True) or request.form.to_dict()
        if not data:
            return jsonify({"status": "error", "message": "No se recibieron datos de feedback"}), 400

        # Estructura del nuevo feedback para RLHF
        nuevo_feedback = {
            "id": int(datetime.utcnow().timestamp() * 1000),
            "timestamp": datetime.utcnow().isoformat(),
            "calificacion": data.get("calificacion") or data.get("rating"),
            "comentario": data.get("comentario", "").strip(),
            "prompt_original": data.get("prompt_original", ""),
            "respuesta_llm": data.get("respuesta_llm") or data.get("recomendacion", ""),
            "metadata_modelo": {
                "probabilidad": data.get("probabilidad"),
                "nivel": data.get("nivel"),
                "clasificacion": data.get("clasificacion")
            }
        }

        # Cargar historial existente sin sobrescribir
        logs = []
        if os.path.exists(FEEDBACK_FILE):
            try:
                with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
                    contenido = f.read().strip()
                    if contenido:
                        logs = json.loads(contenido)
                        if not isinstance(logs, list):
                            logs = [logs]
            except (json.JSONDecodeError, IOError) as err_read:
                print(f"Aviso al leer logs previos: {err_read}. Se reinicializa la lista.")
                logs = []

        # Agregar el nuevo registro
        logs.append(nuevo_feedback)

        # Guardar lista completa con formato legible (indent=4)
        with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(logs, f, ensure_ascii=False, indent=4)

        return jsonify({
            "status": "success",
            "message": "Feedback registrado con éxito en feedback_logs.json",
            "total_registros": len(logs),
            "feedback_id": nuevo_feedback["id"]
        }), 200

    except Exception as e:
        print(f"Error registrando feedback RLHF: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
