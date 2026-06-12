from fastapi import FastAPI, Request
import uvicorn
import time
import os
import numpy as np

from datetime import datetime

from PIL import Image
from tensorflow.keras.models import load_model

import mysql.connector

# =====================================
# FASTAPI
# =====================================
app = FastAPI()

# =====================================
# CREAR CARPETA CAPTURAS
# =====================================
os.makedirs("capturas", exist_ok=True)

# =====================================
# CONEXIÓN MYSQL
# =====================================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="tesinabonetto"
)

cursor = db.cursor()

print("✅ Conectado a MySQL")

# =====================================
# CARGAR MODELO TEACHABLE MACHINE
# =====================================
model = load_model("keras_model.h5", compile=False)

print("✅ Modelo cargado")

# =====================================
# LEER LABELS
# =====================================
with open("labels.txt", "r", encoding="utf-8") as f:

    class_names = []

    for line in f:

        line = line.strip()

        partes = line.split(" ", 1)

        if len(partes) == 2:
            class_names.append(partes[1])
        else:
            class_names.append(partes[0])

print("✅ Labels cargados:", class_names)

# =====================================
# PREDICCIÓN TEACHABLE MACHINE
# =====================================
def predict_image(path):

    img = Image.open(path).convert("RGB")

    img = img.resize((224, 224))

    image = np.asarray(img)

    normalized_image = (image.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    data[0] = normalized_image

    prediction = model.predict(data, verbose=0)

    index = np.argmax(prediction)

    category = class_names[index]

    confidence = float(prediction[0][index])

    return category, confidence

# =====================================
# ENDPOINT CLASIFICACIÓN
# =====================================
@app.post("/classify")
async def classify(request: Request):

    print("\n=================================")
    print("📸 Imagen recibida")
    print("=================================")

    image_bytes = await request.body()

    filename = f"capturas/captura_{int(time.time())}.jpg"

    with open(filename, "wb") as f:
        f.write(image_bytes)

    print("✅ Imagen guardada:", filename)

    # =====================================
    # CLASIFICACIÓN IA
    # =====================================
    category, confidence = predict_image(filename)

    print("♻️ Clasificación:", category)
    print("📊 Confianza:", round(confidence * 100, 2), "%")

    # =====================================
    # AUTOMÁTICA O MANUAL
    # =====================================
    if confidence >= 0.50:
        tipo = "automatica"
    else:
        tipo = "manual"

    print("🤖 Tipo:", tipo)

    # =====================================
    # FECHA Y HORA
    # =====================================
    fecha_hora = datetime.now()

    # =====================================
    # GUARDAR EN MYSQL
    # =====================================
    try:

        sql = """
        INSERT INTO clasificaciones
        (
            fecha_hora,
            residuo,
            confianza,
            clasificacion,
            imagen
        )
        VALUES (%s, %s, %s, %s, %s)
        """

        valores = (
            fecha_hora,
            category,
            confidence,
            tipo,
            filename
        )

        cursor.execute(sql, valores)

        db.commit()

        print("💾 Registro guardado en MySQL")

    except Exception as e:

        print("❌ Error guardando en MySQL:")
        print(e)

    # =====================================
    # RESPUESTA PARA ESP32
    # =====================================
    return {
        "status": "ok",
        "message": "Imagen recibida y procesada",
        "category": category,
        "confidence": confidence,
        "tipo": tipo
    }

# =====================================
# INICIO DEL SERVIDOR
# =====================================
if __name__ == "__main__":

    print("\n🚀 Iniciando servidor EcoS-cam...")
    print("📡 Esperando imágenes de la ESP32...\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )