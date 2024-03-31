import cv2
from tensorflow import keras
import numpy as np


def accept_img_input():
    img = input("Enter the image path: ")
    return img

def open_img(img):
    img = cv2.imread(img)
    return img

def load_model():
    predictive_model = keras.models.load_model("model.keras")
    return predictive_model

def preprocess_img(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    img = img / 255 
    p= np.reshape(img,[1,28,28]) # giving only one datapoint for prediction

    return p

def predict_img(img, model):
    prediction = model.predict(img)
    return np.argmax(prediction)

def main():
    file = accept_img_input()
    img = open_img(file)
    cv2.imshow("given",img)
    model = load_model()
    img = preprocess_img(img)
    prediction = predict_img(img, model)
    print("The digit is recognised as ",prediction)

if __name__ == "__main__":
    main()