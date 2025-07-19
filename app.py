from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model('./Image_classify.keras')

# List of fish categories
data_cat = [
    'Bangus', 'Big Head Carp', 'Black Spotted Barb', 'Catfish', 'Climbing Perch', 'Fourfinger Threadfin',
    'Freshwater Eel', 'Glass Perchlet', 'Goby', 'Gold Fish', 'Gourami', 'Grass Carp', 'Green Spotted Puffer',
    'Indian Carp', 'Indo-Pacific Tarpon', 'Jaguar Gapote', 'Janitor Fish', 'Knifefish', 'Long-Snouted Pipefish',
    'Mosquito Fish', 'Mudfish', 'Mullet', 'Pangasius', 'Perch', 'Scat Fish', 'Silver Carp', 'Silver Perch',
    'Snakehead', 'Tenpounder', 'Tilapia'
]

# Fish descriptions
fish_descriptions = {
    'Bangus': "Bangus, also known as milkfish, is a staple in Southeast Asian cuisine, known for its tender and flavorful meat.",
    'Big Head Carp': "The Big Head Carp is a freshwater fish famous for its large head and mild-tasting meat.",
    'Black Spotted Barb': "The Black Spotted Barb is a small, hardy freshwater fish often found in aquariums.",
    'Catfish': "Catfish are bottom-dwelling fish characterized by their whisker-like barbels around the mouth.",
    'Climbing Perch': "The Climbing Perch is known for its ability to survive out of water for extended periods.",
    'Fourfinger Threadfin': "This coastal fish is recognized for its elongated pectoral fins resembling fingers.",
    'Freshwater Eel': "Freshwater Eels are elongated fish with a snake-like appearance, commonly found in rivers and lakes.",
    'Glass Perchlet': "Glass Perchlets are small, transparent fish often seen in Southeast Asian freshwater habitats.",
    'Goby': "Gobies are small, adaptable fish found in both freshwater and marine environments.",
    'Gold Fish': "Goldfish are popular ornamental fish known for their vibrant colors and adaptability.",
    'Gourami': "Gouramis are freshwater fish prized for their beauty and calm temperament in aquariums.",
    'Grass Carp': "Grass Carp are herbivorous freshwater fish commonly used for aquatic weed control.",
    'Green Spotted Puffer': "This small, brightly colored fish is known for its distinctive green spots and playful nature.",
    'Indian Carp': "Indian Carp, including Rohu and Catla, are freshwater fish widely consumed in South Asia.",
    'Indo-Pacific Tarpon': "This silvery fish is found in coastal waters and rivers, known for its agility and speed.",
    'Jaguar Gapote': "The Jaguar Gapote is a visually striking fish with a pattern resembling a jaguar's spots.",
    'Janitor Fish': "Janitor Fish, or Plecos, are bottom-dwelling fish that help keep aquariums clean by eating algae.",
    'Knifefish': "Knifefish are nocturnal fish with a blade-like body, often found in slow-moving waters.",
    'Long-Snouted Pipefish': "This slender fish has a long snout and is related to seahorses.",
    'Mosquito Fish': "Mosquito Fish are small, hardy fish known for their ability to control mosquito populations.",
    'Mudfish': "Mudfish are highly adaptable fish that can survive in low-oxygen environments.",
    'Mullet': "Mullets are schooling fish often found in coastal waters and known for their economic importance.",
    'Pangasius': "Pangasius, also called Basa, is a freshwater fish known for its mild taste and soft texture.",
    'Perch': "Perch are popular freshwater fish known for their delicious meat and vibrant coloration.",
    'Scat Fish': "Scat Fish are hardy brackish water fish often kept in aquariums for their unique patterns.",
    'Silver Carp': "Silver Carp are large freshwater fish known for their ability to leap out of the water.",
    'Silver Perch': "Silver Perch are freshwater fish prized for their delicate flavor and suitability for aquaculture.",
    'Snakehead': "Snakehead fish are predatory freshwater species known for their aggressive behavior and adaptability.",
    'Tenpounder': "Tenpounders are agile, silvery fish commonly found in tropical coastal waters.",
    'Tilapia': "Tilapia is a widely farmed freshwater fish valued for its mild flavor and versatility in cooking."
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded.", 400

    file = request.files['file']

    if file.filename == '':
        return "No file selected.", 400

    file_path = os.path.join('./static/uploads', file.filename)
    file.save(file_path)

    # Preprocess the image
    img_height, img_width = 180, 180
    image = tf.keras.utils.load_img(file_path, target_size=(img_height, img_width))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    # Predict using the model
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    predicted_class = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100
    description = fish_descriptions.get(predicted_class, "No description available.")

    return render_template(
        'result.html',
        file_path=file_path,
        predicted_class=predicted_class,
        confidence=confidence,
        description=description
    )

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
