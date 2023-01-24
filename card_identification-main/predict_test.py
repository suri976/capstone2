import numpy as np
import requests

from tensorflow.keras.preprocessing.image import load_img

url = 'http://localhost:9696/predict'

### change this path for more tests
img_test_path = 'dataset\\test\\jack of clubs\\5.jpg'

img = load_img(img_test_path, target_size=(224, 224))

card_data={}
card_data['data'] = np.array(img)

response = requests.post(url, json=card_data).json()
print(response)