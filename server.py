from fastapi import FastAPI, Request, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from PIL import Image
from threading import Lock, Timer
from mysql.connector import pooling
from datetime import datetime
import uvicorn
import numpy as np
import os
import random
import string
import time

APP = FastAPI(title="EcoS-cam API")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_POOL = pooling.MySQLConnectionPool(
    pool_name="ecoscam_pool",
    pool_size=10,
    pool_reset_session=True,
    host="localhost",
    user="root",
    password="",
    database="tesinabonetto",
)

os.makedirs("capturas", exist_ok=True)

print("======================================")
print(" SERVER ACTIVO - ECOS-CAM")
print("======================================")

model = load_model("keras_model.h5", compile=False)
with open("labels.txt", "r", encoding="utf-8") as f:
    class_names = [line.strip().split(" ", 1)[-1] for line in f if line.strip()]

model_lock = Lock()
state_lock = Lock()
states = {}
timers = {}

manual_commands = {}

def db_query_one(sql, params=()):
    conn = DB_POOL.get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(sql, params)
        return cur.fetchone()
    finally:
        cur.close()
        conn.close()


def db_query_all(sql, params=()):
    conn = DB_POOL.get_connection()
    cur = conn.cursor(dictionary=True)
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    finally:
        cur.close()
        conn.close()


def db_execute(sql, params=()):
    conn = DB_POOL.get_connection()
    cur = conn.cursor()
    try:
        cur.execute(sql, params)
        conn.commit()
        return cur.lastrowid, cur.rowcount
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def generar_codigo(longitud=6):
    chars = string.ascii_uppercase + string.digits
    for _ in range(100):
        codigo = "".join(random.choices(chars, k=longitud))
        if not db_query_one("SELECT id FROM dispositivos WHERE codigo_activacion=%s", (codigo,)):
            return codigo
    raise RuntimeError("No se pudo generar un código único")


def estado_espera():
    return {
        "estado": "espera",
        "categoria": None,
        "confianza": None,
        "timestamp": None,
        "classification_id": None,
        "codigo": None,
    }


def get_state(codigo):
    with state_lock:
        if codigo not in states:
            states[codigo] = estado_espera() | {"codigo": codigo}
        return dict(states[codigo])


def set_state(codigo, state):
    old_timer = None
    with state_lock:
        states[codigo] = dict(state)
        old_timer = timers.pop(codigo, None)
    if old_timer:
        old_timer.cancel()


def schedule_reset(codigo, seconds):
    def reset():
        set_state(codigo, estado_espera() | {"codigo": codigo})

    with state_lock:
        old = timers.get(codigo)
        if old:
            old.cancel()
        timer = Timer(seconds, reset)
        timer.daemon = True
        timers[codigo] = timer
        timer.start()


class RegistroDispositivo(BaseModel):
    nombre: str


class HabilitarDispositivo(BaseModel):
    codigo: str


class SeleccionManual(BaseModel):
    codigo: str
    categoria: str
    classification_id: int | None = None


# ========================== PREDICCIÓN ==========================
def predict_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    image = np.asarray(img).astype(np.float32)
    normalized = (image / 127.5) - 1.0
    data = np.ndarray((1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized

    with model_lock:
        prediction = model.predict(data, verbose=0)

    index = int(np.argmax(prediction))
    category = class_names[index]
    confidence = float(prediction[0][index])
    return category, confidence


# ===================== REGISTRAR DISPOSITIVO ====================
@APP.post("/registrar-dispositivo")
async def registrar_dispositivo(disp: RegistroDispositivo):
    codigo = generar_codigo()
    last_id, _ = db_execute(
        """
        INSERT INTO dispositivos (nombre, codigo_activacion, habilitado)
        VALUES (%s, %s, 0)
        """,
        (disp.nombre.strip(), codigo),
    )
    return {
        "status": "ok",
        "id": last_id,
        "codigo": codigo,
        "habilitado": False,
    }


# ======================= HABILITAR ==============================
@APP.post("/habilitar-dispositivo")
async def habilitar_dispositivo(data: HabilitarDispositivo):
    codigo = data.codigo.strip().upper()
    row = db_query_one(
        "SELECT id FROM dispositivos WHERE codigo_activacion=%s",
        (codigo,),
    )
    if not row:
        return {"status": "error", "message": "Código inválido"}

    db_execute(
        "UPDATE dispositivos SET habilitado=1 WHERE codigo_activacion=%s",
        (codigo,),
    )
    return {"status": "ok", "codigo": codigo, "habilitado": True}


@APP.post("/deshabilitar-dispositivo")
async def deshabilitar_dispositivo(data: HabilitarDispositivo):
    codigo = data.codigo.strip().upper()
    row = db_query_one(
        "SELECT id FROM dispositivos WHERE codigo_activacion=%s",
        (codigo,),
    )
    if not row:
        return {"status": "error", "message": "Código inválido"}

    db_execute(
        "UPDATE dispositivos SET habilitado=0 WHERE codigo_activacion=%s",
        (codigo,),
    )
    set_state(codigo, estado_espera() | {"codigo": codigo})
    return {"status": "ok", "codigo": codigo, "habilitado": False}


@APP.get("/estado-dispositivo")
def estado_dispositivo(codigo: str = Query(...)):
    codigo = codigo.strip().upper()
    row = db_query_one(
        "SELECT habilitado FROM dispositivos WHERE codigo_activacion=%s",
        (codigo,),
    )
    if not row:
        return {"status": "error", "message": "Código inválido", "habilitado": False}
    return {"status": "ok", "habilitado": bool(row["habilitado"])}


# ========================= CLASSIFY =============================
@APP.post("/classify")
async def classify(request: Request, codigo: str = Query(...), trigger: str = Query("auto")):
    codigo = codigo.strip().upper()
    dispositivo = db_query_one(
        "SELECT id, habilitado FROM dispositivos WHERE codigo_activacion=%s",
        (codigo,),
    )
    if not dispositivo:
        raise HTTPException(status_code=404, detail="Código inválido")
    if not dispositivo["habilitado"]:
        raise HTTPException(status_code=403, detail="Dispositivo no habilitado")

    image_bytes = await request.body()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Imagen vacía")

    filename = f"capturas/captura_{int(time.time() * 1000)}_{codigo}.jpg"
    with open(filename, "wb") as f:
        f.write(image_bytes)

    category, confidence = predict_image(filename)
    tipo = "automatica" if confidence >= 0.50 else "manual"
    fecha_hora = datetime.now()

    classification_id, _ = db_execute(
        """
        INSERT INTO clasificaciones
        (fecha_hora, residuo, confianza, clasificacion, imagen, dispositivo_id)
        VALUES (%s, %s, %s, %s, %s, %s)
        """,
        (
            fecha_hora,
            category,
            confidence,
            tipo,
            filename,
            dispositivo["id"],
        ),
    )

    state = {
        "estado": "automatico" if tipo == "automatica" else "manual",
        "categoria": category,
        "confianza": confidence,
        "timestamp": fecha_hora.isoformat(),
        "classification_id": classification_id,
        "codigo": codigo,
        "trigger": trigger,
    }
    set_state(codigo, state)

    if tipo == "automatica":
        schedule_reset(codigo, 8)

    return {
        "status": "ok",
        "category": category,
        "confidence": confidence,
        "tipo": tipo,
        "classification_id": classification_id,
        "codigo": codigo,
    }


# ====================== SELECCIÓN MANUAL ========================
@APP.post("/seleccion-manual")
async def seleccion_manual(data: SeleccionManual):
    codigo = data.codigo.strip().upper()
    categoria = data.categoria.strip().lower()
    permitidas = {"papel", "plastico", "vidrio", "organico"}
    if categoria not in permitidas:
        raise HTTPException(status_code=400, detail="Categoría inválida")

    dispositivo = db_query_one(
        "SELECT id, habilitado FROM dispositivos WHERE codigo_activacion=%s",
        (codigo,),
    )
    if not dispositivo:
        raise HTTPException(status_code=404, detail="Código inválido")

    state = get_state(codigo)
    classification_id = data.classification_id or state.get("classification_id")
    if classification_id is None:
        raise HTTPException(status_code=409, detail="No hay clasificación pendiente")

    row = db_query_one(
        """
        SELECT id, dispositivo_id, clasificacion
        FROM clasificaciones
        WHERE id=%s AND dispositivo_id=%s
        """,
        (classification_id, dispositivo["id"]),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Clasificación no encontrada")

    db_execute(
        """
        UPDATE clasificaciones
        SET categoria_final=%s, seleccion_manual=%s, clasificacion='manual'
        WHERE id=%s AND dispositivo_id=%s
        """,
        (categoria, categoria, classification_id, dispositivo["id"]),
    )
    with state_lock:manual_commands[codigo] = categoria
    gracias = {
        "estado": "gracias",
        "categoria": categoria,
        "confianza": state.get("confianza"),
        "timestamp": datetime.now().isoformat(),
        "classification_id": classification_id,
        "codigo": codigo,
    }
    set_state(codigo, gracias)
    schedule_reset(codigo, 4)

    return {"ok": True, "classification_id": classification_id, "codigo": codigo}


@APP.get("/comando-manual")
def comando_manual(codigo: str = Query(...)):
    codigo = codigo.strip().upper()

    dispositivo = db_query_one(
        "SELECT id, habilitado FROM dispositivos WHERE codigo_activacion=%s",
        (codigo,),
    )

    if not dispositivo:
        raise HTTPException(
            status_code=404,
            detail="Código inválido"
        )

    if not dispositivo["habilitado"]:
        return {
            "ok": True,
            "hay_comando": False
        }

    with state_lock:

        categoria = manual_commands.pop(codigo, None)

    if categoria is None:

        return {
            "ok": True,
            "hay_comando": False
        }

    return {
        "ok": True,
        "hay_comando": True,
        "categoria": categoria
    }


# =========================== ESTADO =============================
@APP.get("/estado")
def get_estado(codigo: str = Query(...)):
    codigo = codigo.strip().upper()
    row = db_query_one(
        "SELECT id FROM dispositivos WHERE codigo_activacion=%s",
        (codigo,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Código inválido")
    return get_state(codigo)


if __name__ == "__main__":
    print("🚀 ARRANCANDO UVICORN...")
    uvicorn.run(APP, host="0.0.0.0", port=8000, reload=False)
