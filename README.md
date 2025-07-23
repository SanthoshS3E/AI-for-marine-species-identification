# 🐠 AI for Marine Species Identification

<img width="1915" height="906" alt="Screenshot 2025-07-23 163429" src="https://github.com/user-attachments/assets/e50f13b5-45eb-4835-a0d1-b1f4ba0aa183" />
<img width="1917" height="898" alt="Screenshot 2025-07-23 163459" src="https://github.com/user-attachments/assets/1a80a7c8-4bb2-4259-bac5-4d67209aa0b4" />


A deep learning-powered web app that identifies marine fish species from images using a custom-trained CNN model. Built for researchers and marine enthusiasts to simplify species classification and access essential fish information.

---

## 🔍 Overview

- **Model**: Custom CNN built with Keras and TensorFlow.
- **Dataset**: 1761 labeled images across 31 fish species.
- **Accuracy**: Achieved 85% validation accuracy.
- **Deployment**: Simple HTML/CSS frontend with Flask backend.

---

## 📁 Project Structure

marine_species_classifier/
│
├── model/
│ └── fish_model.h5 # Trained model
├── data/
│ └── fish_dataset/ # Labeled images
├── web/
│ ├── index.html # Frontend interface
│ └── style.css # Styling
├── app.py # Flask backend for inference
├── model_building.ipynb # Training notebook
└── README.md


--

🚀 How to Run Locally
Clone the Repo
git clone https://github.com/your-username/marine-species-identifier.git

cd marine-species-identifier
Install Dependencies
pip install -r requirements.txt


Run the Web App
python app.py

Open in Browser
http://localhost:5000


📊 Performance
Metric	Value
Training Accuracy	~90%
Validation Accuracy	~85%
Total Classes	31 fish types
Inference Time	<1s/image

📌 Roadmap
🔄 Add more species to dataset.

⚡ Try transfer learning with EfficientNet.

📱 Build mobile-friendly UI.

🌐 Deploy via public cloud or HuggingFace Spaces.

📚 Acknowledgements
TensorFlow & Keras
Flask


📝 License
MIT License. Open to collaboration and extensions.

🙌 Contributions Welcome!
Fork, star, or raise a pull request if you'd like to improve this project.
