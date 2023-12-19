from skimage import io, color, exposure
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

def load_image(image_path):
    return io.imread(image_path)

def preprocess_image(image):
    # Convertim imaginea în alb-negru
    gray_image = color.rgb2gray(image)
    # Calculăm descriptorul HOG (Histogram of Oriented Gradients)
    features, _ = hog(gray_image, orientations=8, pixels_per_cell=(4, 4), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=True)
    return features

def train_adaboost_classifier(X_train, y_train, n_estimators=50):
    # Definim un clasificator Decision Tree de bază
    base_classifier = DecisionTreeClassifier(max_depth=1)
    # Inițializăm AdaBoost cu clasificatorul de bază și numărul de clasificatori
    adaboost_classifier = AdaBoostClassifier(base_classifier, n_estimators=n_estimators)
    # Antrenăm clasificatorul
    adaboost_classifier.fit(X_train, y_train)
    return adaboost_classifier

def predict_digit(image_path, adaboost_classifier):
    # Încărcăm imaginea și o preprocesăm
    image = load_image(image_path)
    features = preprocess_image(image)
    # Facem o predicție folosind clasificatorul AdaBoost
    prediction = adaboost_classifier.predict([features])
    return prediction[0]

# Exemplu de utilizare
if __name__ == "__main__":
    # Simulăm un set de date de antrenare (nu folosim validare în acest exemplu)
    # Acest set ar trebui să conțină caracteristici (X) și etichetele corespunzătoare (y)
    # În practică, ar trebui să aveți un set de date de antrenare etichetat corespunzător.
    X_train = np.random.rand(100, 324)  # 100 de imagini, fiecare cu 324 de caracteristici HOG
    y_train = np.random.randint(0, 10, 100)  # Etichetele (cifrele de la 0 la 9)

    # Antrenăm clasificatorul AdaBoost
    adaboost_classifier = train_adaboost_classifier(X_train, y_train)

    # Încărcăm o imagine pentru a o clasifica
    test_image_path = "C:\\Users\\remus\\Documents\\GitHub\\ML-tema-practica\\cifra.jpg"

    predicted_digit = predict_digit(test_image_path, adaboost_classifier)

    print(f"The predicted digit is: {predicted_digit}")
