import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# =====================================
# PARÁMETROS
# =====================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

# =====================================
# GENERADOR DE DATOS
# =====================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,

    rotation_range=25,
    zoom_range=0.25,
    horizontal_flip=True,

    brightness_range=[0.6, 1.4],

    width_shift_range=0.15,
    height_shift_range=0.15
)

# =====================================
# DATASET ENTRENAMIENTO
# =====================================
train_gen = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

# =====================================
# DATASET VALIDACIÓN
# =====================================
val_gen = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# =====================================
# MODELO BASE
# =====================================
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# =====================================
# FINE TUNING
# =====================================
base_model.trainable = True

# Congelar la mayoría de capas
for layer in base_model.layers[:-30]:
    layer.trainable = False

# =====================================
# MODELO FINAL
# =====================================
model = Sequential([
    base_model,

    GlobalAveragePooling2D(),

    Dropout(0.3),

    Dense(128, activation="relu"),

    Dropout(0.2),

    Dense(4, activation="softmax")
])

# =====================================
# COMPILAR
# =====================================
model.compile(
    optimizer=Adam(learning_rate=0.00001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =====================================
# RESUMEN
# =====================================
model.summary()

# =====================================
# ENTRENAR
# =====================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# =====================================
# GUARDAR MODELO
# =====================================
model.save("waste_classifier.h5")

print("\n✅ Modelo entrenado y guardado como waste_classifier.h5")