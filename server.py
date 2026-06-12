from fastapi import FastAPI, Request, Query
import uvicorn
import time
import os
import numpy as np
import random
import string
from threading import Timer
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model
import mysql.connector
from pydantic import BaseModel

print("🔥 SERVER PRO ACTIVO - VERSION CORRECTA")
print("INICIO SERVER OK")

# =========================
# MODELO PARA REGISTRO
# =========================
class RegistroDispositivo(BaseModel):
    nombre: str

# =========================
# FUNCIÓN GENERAR CÓDIGO
# =========================
def generar_codigo(longitud=6):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=longitud))

# =========================
# APP
# =========================
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ESTADO EN TIEMPO REAL (RAM)
# =========================
estado_actual = {
    "estado": "espera",
    "categoria": None,
    "confianza": None,
    "timestamp": None
}

last_classification_id = None

# =========================
# RESET AUTOMÁTICO
# =========================
def reset_estado():
    global estado_actual
    estado_actual = {
        "estado": "espera",
        "categoria": None,
        "confianza": None,
        "timestamp": None
    }

def volver_a_espera():
    global estado_actual
    estado_actual = {
        "estado": "espera",
        "categoria": None,
        "confianza": None,
        "timestamp": None
    }

def ciclo_gracias():
    global estado_actual
    estado_actual = {
        "estado": "gracias",
        "categoria": estado_actual.get("categoria"),
        "confianza": estado_actual.get("confianza"),
        "timestamp": estado_actual.get("timestamp")
    }
    Timer(4, volver_a_espera).start()

# =========================
# CARPETA IMÁGENES
# =========================
os.makedirs("capturas", exist_ok=True)

# =========================
# MYSQL
# =========================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="tesinabonetto"
)
cursor = db.cursor()

# =========================
# MODELO IA
# =========================
print("ANTES DE MODELO")
model = load_model("keras_model.h5", compile=False)
print("DESPUES DE MODELO")

with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip().split(" ", 1)[-1] for line in f]

# =========================
# PREDICCIÓN
# =========================
def predict_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224))
    image = np.asarray(img)
    image = (image.astype(np.float32) / 127.5) - 1
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = image
    prediction = model.predict(data, verbose=0)
    index = np.argmax(prediction)
    category = class_names[index]
    confidence = float(prediction[0][index])
    return category, confidence

# =========================
# ENDPOINT: REGISTRAR DISPOSITIVO
# =========================
@app.post("/registrar-dispositivo")
async def registrar_dispositivo(disp: RegistroDispositivo):
    codigo = generar_codigo()
    try:
        cursor.execute(
            "INSERT INTO dispositivos (nombre, codigo_activacion) VALUES (%s, %s)",
            (disp.nombre, codigo)
        )
        db.commit()
        return {"status": "ok", "codigo": codigo}
    except Exception as e:
        return {"status": "error", "detalle": str(e)}

# =========================
# ENDPOINT: CLASIFICAR (modificado con código)
# =========================
@app.post("/classify")
async def classify(request: Request, codigo: str = Query(...)):
    global estado_actual, last_classification_id

    # Validar código del dispositivo
    cursor.execute("SELECT id FROM dispositivos WHERE codigo_activacion = %s", (codigo,))
    row = cursor.fetchone()
    if not row:
        return {"status": "error", "message": "Código inválido"}
    dispositivo_id = row[0]

    image_bytes = await request.body()
    filename = f"capturas/{int(time.time())}.jpg"
    with open(filename, "wb") as f:
        f.write(image_bytes)

    category, confidence = predict_image(filename)
    tipo = "automatico" if confidence >= 0.5 else "manual"

    estado_actual = {
        "estado": "automatico" if tipo == "automatico" else "manual",
        "categoria": category,
        "confianza": confidence,
        "timestamp": datetime.now().isoformat(),
        "classification_id": None
    }

    if tipo == "automatico":
        Timer(8, volver_a_espera).start()

    try:
        # Incluir dispositivo_id en el INSERT
        cursor.execute("""
            INSERT INTO clasificaciones
            (dispositivo_id, fecha_hora, residuo, confianza, clasificacion, imagen)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (dispositivo_id, datetime.now(), category, confidence, tipo, filename))

        db.commit()
        classification_id = cursor.lastrowid
        print("ID GUARDADO:", classification_id)    
        estado_actual["classification_id"] = classification_id
    except Exception as e:
        print("Error MySQL:", e)

    return {
        "status": "ok",
        "category": category,
        "confidence": confidence,
        "tipo": tipo
    }

# =========================
# ENDPOINT: SELECCIÓN MANUAL (sin cambios)
# =========================
@app.post("/seleccion-manual")
async def seleccion_manual(request: Request):
    global estado_actual
    data = await request.json()
    categoria = data["categoria"]
    try:
        classification_id = estado_actual.get("classification_id")
        print("ID RECUPERADO:", classification_id)
        print("CATEGORIA ELEGIDA:", categoria) 
        if classification_id is not None:
            cursor.execute("""
                UPDATE clasificaciones
                SET categoria_final = %s, seleccion_manual = %s
                WHERE id = %s
            """, (categoria, categoria, classification_id))
            db.commit()
            print(f"Registro {classification_id} actualizado a {categoria}")
    except Exception as e:
        print("Error UPDATE:", e)

    estado_actual = {
        "estado": "gracias",
        "categoria": categoria,
        "confianza": estado_actual.get("confianza"),
        "timestamp": datetime.now().isoformat(),
        "classification_id": classification_id
    }
    Timer(4, volver_a_espera).start()
    return {"ok": True}

# =========================
# ENDPOINT: ESTADO (sin cambios)
# =========================
@app.get("/estado")
def get_estado():
    return estado_actual

# =========================
# INICIO DEL SERVIDOR
# =========================
if __name__ == "__main__":
    print("🚀 ARRANCANDO UVICORN...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)