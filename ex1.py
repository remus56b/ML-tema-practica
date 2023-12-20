from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
import numpy as np
from PIL import Image

# Încărcați setul de date MNIST
digits = datasets.load_digits()

# Antrenarea modelului AdaBoost
model = AdaBoostClassifier(n_estimators=50)
model.fit(digits.data, digits.target)


def predict_digit(image_path):
    # Deschideți imaginea și convertiți-o într-un vector 1D
    image = Image.open(image_path).convert('L')
    image = image.resize((8, 8), Image.LANCZOS)
    image = np.array(image).reshape(1, -1)

    # Folosiți modelul pentru a prezice cifra
    prediction = model.predict(image)
    return prediction[0]


print(predict_digit('C:\\Users\\remus\\Documents\\GitHub\\ML-tema-practica\\cifra1.jpg'))
