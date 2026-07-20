from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# cargo el modelo al arrancar la app
with open('modelo_fraude.pkl', 'rb') as f:
    datos = pickle.load(f)

modelo       = datos['modelo']
scaler       = datos['scaler']
umbral       = datos['umbral']
features     = datos['features']
es_ensemble  = datos['es_ensemble']
mejor_nombre = datos['mejor_nombre']


def predecir(valores_dict):
    """Recibe un dict con los valores de la transaccion y devuelve probabilidad y etiqueta."""
    # armo el array en el orden correcto
    fila = np.array([[valores_dict[f] for f in features]])

    if es_ensemble:
        probas = [m.predict_proba(fila)[:, 1][0] for m in modelo.values()
                  if hasattr(m, 'predict_proba')]
        proba = float(np.mean(probas))
    else:
        proba = float(modelo.predict_proba(fila)[:, 1][0])

    etiqueta = int(proba >= umbral)
    return proba, etiqueta


@app.route('/')
def index():
    return render_template('index.html', umbral=umbral, modelo=mejor_nombre)


@app.route('/predecir', methods=['POST'])
def predecir_endpoint():
    try:
        data = request.get_json()

        # normalizo Amount y Time igual que en el entrenamiento
        amount_scaled = float(scaler.transform([[float(data['Amount'])]])[0][0])
        time_scaled   = float(scaler.transform([[float(data['Time'])]])[0][0])

        # armo el dict con todas las features
        valores = {}
        for i in range(1, 29):
            valores[f'V{i}'] = float(data.get(f'V{i}', 0))
        valores['Amount_scaled'] = amount_scaled
        valores['Time_scaled']   = time_scaled

        proba, etiqueta = predecir(valores)

        return jsonify({
            'probabilidad': round(proba * 100, 2),
            'prediccion'  : etiqueta,
            'resultado'   : 'FRAUDE DETECTADO' if etiqueta == 1 else 'Transaccion Normal',
            'umbral'      : round(umbral * 100, 2)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
