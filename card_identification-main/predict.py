import os

from tensorflow import keras
from tensorflow.keras.applications.xception import preprocess_input

from flask import Flask
from flask import request
from flask import jsonify

test_path = 'dataset/test'

model = keras.models.load_model('vgg16_06_0.658.h5')

app = Flask('card_identificator')

@app.route('/predict', methods=['POST'])
def predict():
    
    card_data = request.get_json()

    X = preprocess_input(card_data)
    pred = model.predict(X)

    classes = os.listdir(test_path)
    
   
    result = {
        'Card': dict(zip(classes, pred[0]))
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run()
