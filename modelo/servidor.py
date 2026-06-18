import os
import json
import tempfile
import requests
import stanza
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS
from lse_simple import generatorText2Gloss

WHISPER_URL      = "http://127.0.0.1:8080/inference"
RULES_PATH       = "rules.csv"
CORPUS_PATH      = "corpus-generado.csv"
DICCIONARIO_PATH = "palabras_definidas.txt"
PORT             = 5050

app = Flask(__name__)
CORS(app)  # permite peticiones desde cualquier origen (necesario para WebGL)

print("Cargando modelo Stanza (español)...")
nlp = stanza.Pipeline(lang='es', processors='tokenize,mwt,pos,lemma', verbose=False)
print("Stanza listo.")

# diccionario
diccionario = {}

def cargar_diccionario():
    if not os.path.exists(DICCIONARIO_PATH):
        print(f"ERROR: No existe {DICCIONARIO_PATH}")
        return
    with open(DICCIONARIO_PATH, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if not linea or linea.startswith("#"):
                continue
            partes = linea.split("=")
            if len(partes) == 2:
                diccionario[partes[0].strip().lower()] = partes[1].strip()

cargar_diccionario()
print(f"Diccionario: {len(diccionario)} entradas")

# lógica principal

def procesar_frase(frase: str) -> list[str]:
    """Convierte una glosa LSE en una lista de IDs de animación."""
    acentos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'Á': 'A', 'É': 'E', 'Í': 'I', 'Ó': 'O', 'Ú': 'U'
    }
    frase_normalizada = ''.join(acentos.get(c, c) for c in frase)
    secuencia = []

    for palabra in frase_normalizada.split():
        palabra = palabra.split("-")[0]
        clave = palabra.strip().lower()
        if not clave:
            continue
        if clave in diccionario:
            secuencia.append(diccionario[clave])
        else:
            for letra in clave:
                if letra in diccionario:
                    secuencia.append(diccionario[letra])
    return secuencia


def transcribir(ruta_audio: str) -> str:
    """Convierte el audio a WAV y lo envía a whisper.cpp."""
    ruta_wav = ruta_audio.replace(os.path.splitext(ruta_audio)[1], "_conv.wav")
    subprocess.run([
        "ffmpeg", "-y", "-i", ruta_audio,
        "-ar", "16000", "-ac", "1", "-f", "wav", ruta_wav
    ], check=True, capture_output=True)

    try:
        with open(ruta_wav, "rb") as f:
            resp = requests.post(
                WHISPER_URL,
                files={"file": f},
                data={"language": "es"},
                timeout=60
            )
        resp.raise_for_status()
        return resp.json().get("text", "").strip()
    finally:
        if os.path.exists(ruta_wav):
            os.unlink(ruta_wav)


def pipeline_completo(ruta_audio: str) -> dict:
    """Audio -> texto -> glosa →-> IDs. Devuelve dict con todos los pasos."""
    texto = transcribir(ruta_audio)
    glosa = generatorText2Gloss(texto, RULES_PATH, CORPUS_PATH, nlp)
    ids   = procesar_frase(glosa)
    return {
        "texto_original": texto,
        "glosa_lse":      glosa.strip(),
        "animaciones":    ids,             # lista de IDs para Unity
        "secuencia":      ",".join(ids)    # string listo para SendMessage
    }


@app.route("/traducir", methods=["POST"])
def traducir_audio():
    if "audio" not in request.files:
        return jsonify({"error": "Falta el campo 'audio'"}), 400

    audio_file = request.files["audio"]

    # Guardar temporalmente (whisper.cpp necesita un fichero en disco)
    suffix = os.path.splitext(audio_file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp.name)
        ruta_tmp = tmp.name

    try:
        resultado = pipeline_completo(ruta_tmp)
        return jsonify(resultado)
    except requests.exceptions.ConnectionError:
        return jsonify({"error": "No se puede conectar con whisper.cpp en " + WHISPER_URL}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(ruta_tmp)


@app.route("/traducir_texto", methods=["POST"])
def traducir_texto():
    data = request.get_json(silent=True)
    if not data or "texto" not in data:
        return jsonify({"error": "Falta el campo 'texto'"}), 400

    try:
        glosa = generatorText2Gloss(data["texto"], RULES_PATH, CORPUS_PATH, nlp)
        ids   = procesar_frase(glosa)
        return jsonify({
            "texto_original": data["texto"],
            "glosa_lse":      glosa.strip(),
            "animaciones":    ids,
            "secuencia":      ",".join(ids)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/palabras", methods=["GET"])
def palabras():
    """Devuelve el diccionario de glosas disponibles."""
    return jsonify(diccionario)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "palabras": len(diccionario)})


if __name__ == "__main__":
    print(f"\n✓ Servidor LSE arrancado en http://localhost:{PORT}")
    print(f"  POST /traducir       → audio → LSE")
    print(f"  POST /traducir_texto → texto → LSE")
    print(f"  GET  /palabras       → diccionario\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)