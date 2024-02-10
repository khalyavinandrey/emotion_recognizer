from bs4 import BeautifulSoup
import requests
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import tensorflow as tf


EMOTIONS = {
    "Angry": "https://ru.pinterest.com/tylo1968/angry-faces/",
    "Surprise": "https://ru.pinterest.com/gress2247/surprise/",
    "Happy": "https://ru.pinterest.com/kristifitgirl/happy-people-smiling-faces/",
    "Neutral": "https://ru.pinterest.com/irenenguyen37/neutral-face-expression/",
    "Sad": "https://ru.pinterest.com/helenwyrich/sad-faces/"
}

BASE_FOLDER = "images"

model = load_model("my_keras_model.h5")


def create_folder(folder_path):
    try:
        os.makedirs(folder_path, exist_ok=True)
    except FileExistsError:
        print(f"Папка {folder_path} уже существует!")


def download_images(images, folder_path):
    count = 0
    print(f"Найдено {len(images)} изображений!")

    for i, image in enumerate(images):
        image_link = image.get("data-srcset") or image.get("data-src") or image.get("data-fallback-src") or image.get(
            "src")
        if not image_link:
            continue

        try:
            r = requests.get(image_link).content
            with open(os.path.join(folder_path, f"image{i + 1}.jpg"), "wb") as f:
                f.write(r)
            count += 1
        except Exception as e:
            print(f"Ошибка при скачивании изображения: {e}")

    if count == len(images):
        print("Все изображения скачаны!")
    else:
        print(f"Скачано {count} изображений из {len(images)}")


def predict_emotion(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)

    result_index = np.argmax(predictions)
    with open("emotions.txt", encoding='utf-8') as f:
        content = f.readlines()
    label = []
    for i in content:
        label.append(i[:-1])

    print(label[result_index])
    return label[result_index]


def evaluate_emotions_in_folder(folder_path):
    correct_predictions = 0
    total_images = 0

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        predicted_emotion = predict_emotion(image_path)
        actual_emotion = folder_path.split("/")[-1]

        if predicted_emotion == actual_emotion:
            correct_predictions += 1

        total_images += 1

    accuracy = correct_predictions / total_images if total_images > 0 else 0
    return accuracy


def main(url, emotion):
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    images = soup.findAll('img')

    folder_path = os.path.join(BASE_FOLDER, emotion)
    create_folder(folder_path)
    download_images(images, folder_path)

    accuracy = evaluate_emotions_in_folder(folder_path)
    print(f"Для эмоции {emotion} вероятность правильного распределения: {accuracy}")


for emotion, url in EMOTIONS.items():
    main(url, emotion)
