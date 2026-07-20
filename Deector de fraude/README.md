# Detector de Fraude en Tarjetas de Credito

App web para detectar transacciones fraudulentas usando un modelo de Machine Learning
entrenado con el dataset de Kaggle Credit Card Fraud Detection.

## Estructura del proyecto

```
app/
├── app.py                      # aplicacion Flask
├── modelo_fraude.pkl           # modelo entrenado (generado por el notebook)
├── requirements.txt            # dependencias
├── render.yaml                 # configuracion para Render
├── templates/
│   └── index.html              # interfaz web
└── README.md
```

## Como correr localmente

1. Primero correr el notebook `modelo_riesgo_crediticio.ipynb` para generar `modelo_fraude.pkl`
2. Copiar `modelo_fraude.pkl` en la carpeta `app/`
3. Instalar dependencias:
   ```
   pip install -r requirements.txt
   ```
4. Correr la app:
   ```
   python app.py
   ```
5. Abrir en el navegador: `http://localhost:5000`

## Despliegue en Render

1. Subir esta carpeta a un repositorio de GitHub
2. Entrar a [render.com](https://render.com) y crear cuenta gratuita
3. Hacer click en **New > Web Service**
4. Conectar el repositorio de GitHub
5. Render detecta automaticamente el `render.yaml` y configura todo
6. El enlace al servicio queda disponible en el dashboard de Render

**Enlace al servicio:** https://detector-fraude.onrender.com *(actualizar con el enlace real)*

## Recursos utilizados

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Scikit-learn](https://scikit-learn.org/)
- [Render Docs](https://render.com/docs)
- Dataset: [Credit Card Fraud Detection - Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
