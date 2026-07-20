import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Cargar el pipeline
with open("modelo_completo.pkl", "rb") as archivo:
    modelo_pipeline = pickle.load(archivo)

clases_vino = ["Variedad Altiplano", "Variedad Valle Central", "Variedad Costa"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None
    if request.method == "POST":
        try:
            alcohol = float(request.form["alcohol"])
            malic_acid = float(request.form["malic_acid"])
            ash = float(request.form["ash"])
            alcalinity = float(request.form["alcalinity"])

            datos_entrada = np.array([[alcohol, malic_acid, ash, alcalinity]])
            id_clase = modelo_pipeline.predict(datos_entrada)[0]
            prediccion = clases_vino[id_clase]
        except ValueError:
            error = "Por favor, introduce valores numéricos válidos."
    
    return render_template("index.html", prediccion=prediccion, error=error)

if __name__ == "__main__":
    # En Colab usamos el puerto 5000
    app.run(host="0.0.0.0", port=5000)
