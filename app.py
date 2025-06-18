from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the Keras model
model = load_model('weather_predict.keras')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        tmin = float(request.form['tmin'])
        tmax = float(request.form['tmax'])
        prcp = float(request.form['prcp'])
        wspd = float(request.form['wspd'])
        pres = float(request.form['pres'])

        features = np.array([[tmin, tmax, prcp, wspd, pres]])

        prediction = model.predict(features)
        result = f"Predicted Avg Temperature: {prediction[0][0]:.2f} °C"
    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
