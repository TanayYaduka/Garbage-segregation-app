# ♻️ Smart Garbage Separation App

A deep learning-based web app that classifies uploaded waste images into categories like **cardboard**, **glass**, **metal**, **paper**, **plastic**, or **trash**. This tool promotes effective waste management and supports sustainability initiatives.

🌐 **Try the Live App Here**:  
👉 [Smart Garbage Separation App](https://garbage-segregation-app-tanay-yaduka.streamlit.app/)

---

## 🚀 Features

- Upload waste images from your device or via a URL
- Uses a Convolutional Neural Network (CNN) trained on multiple garbage classes
- Instant classification into one of 6 waste types
- Built with **TensorFlow**, **Keras**, **Streamlit**, and **Pillow**
- User-friendly interface with real-time feedback

---

## 🖼️ Garbage Classes

The model is trained to identify:

- 📦 Cardboard  
- 🍾 Glass  
- 🥫 Metal  
- 📄 Paper  
- 🧴 Plastic  
- 🚮 Trash

---

## 🛠️ How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/garbage-separation-app.git
cd garbage-separation-app/Deployment
```

2. **Install dependencies:**

Create a virtual environment (optional) and run:

```bash
pip install -r requirements.txt
```

3. **Download the model weights:**

Make sure `modelnew.h5` is either in the `weights/` folder or update the script to download it from Google Drive or another URL.

4. **Run the Streamlit app:**

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Deployment/
├── app.py                # Streamlit app
├── utils.py              # Helper functions & model architecture
├── weights/
│   └── modelnew.h5       # Trained CNN weights
├── train/                # Training data (if applicable)
└── requirements.txt
```

---

## 📦 Requirements

Example dependencies (included in `requirements.txt`):

```
streamlit
tensorflow
numpy
pillow
```

---

## 🙌 Credits

- Trained on a publicly available garbage classification dataset
- Icons and illustrations from Flaticon and Storyset

---

## 📬 Contact

Made by **Tanay Yaduka**  
🔗 [Live Demo](https://garbage-segregation-app-tanay-yaduka.streamlit.app/)
---

## 🛠️ How We Built It



These are the steps involved in making this project:

1. 📚 Importing Libraries  
2. 📁 Data Importing  
3. 🔍 Data Exploration  
4. ⚙️ Data Configuration  
5. 🧹 Preparing the Data  
6. 🔄 Creating a Generator for Training Set  
7. 🔁 Creating a Generator for Testing Set  
8. 📝 Writing the labels into a text file `Labels.txt`  
9. 🧠 Model Creation  
10. 🧪 Model Compilation  
11. 🏋️ Training the Model (`batch_size = 32`, `epochs = 10`)  
12. 🔎 Testing Predictions  
13. 💾 Saving model as `modelnew.h5`  
14. 🌐 Deploying the Model as a Web Application using Streamlit  
