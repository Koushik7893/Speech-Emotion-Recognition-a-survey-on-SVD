# **Speech Emotion Recognition using SVD with CNN, LSTM, and SVM**  

### 🎯 **Overview**  
This project focuses on **Speech Emotion Recognition (SER)** using the **Toronto Emotional Speech Set (TESS) dataset**. It applies **Singular Value Decomposition (SVD)** for feature extraction and evaluates three classification models: **CNN, LSTM, and SVM**. The project is implemented in separate Jupyter notebooks for each model, providing insights into how different architectures handle emotional speech classification.  

---

## 📂 **Project Structure**  

```
📦 Speech-Emotion-Recognition-SVD
│-- 📜 README.md
│-- 📜 requirements.txt  # Required dependencies
│-- 📂 dataset/  # Contains the TESS dataset (after preprocessing)
│   └-- tess_data/  
│-- 📜 CNN.ipynb  # CNN-based implementation
│-- 📜 LSTM.ipynb  # LSTM-based implementation
│-- 📜 SVM.ipynb  # SVM-based implementation
```

---

## 🎙️ **Dataset: TESS (Toronto Emotional Speech Set)**  
The **TESS dataset** contains recordings of **two female actors**, simulating emotions across **seven categories**:  

- **Angry**
- **Disgust**
- **Fear**
- **Happy**
- **Neutral**
- **Sad**
- **Surprised**  

Each model is trained on this dataset after applying **SVD for feature extraction**.

---

## 🔍 **Models Implemented**  
This project compares the performance of three models for Speech Emotion Recognition:  

1️⃣ **Convolutional Neural Network (CNN)** – Extracts spatial and temporal patterns in speech signals.  
2️⃣ **Long Short-Term Memory (LSTM)** – Captures long-range dependencies in time-series speech data.  
3️⃣ **Support Vector Machine (SVM)** – A classical ML model for feature-based classification.  

Each model is implemented in its respective Jupyter notebook (`CNN.ipynb`, `LSTM.ipynb`, and `SVM.ipynb`).  

---

## 🛠 **Preprocessing & Feature Extraction**  
1. **Audio Preprocessing**  
   - Convert audio files to **Mel-Frequency Cepstral Coefficients (MFCCs)**.  
   - Normalize the extracted features for consistency.  

2. **Dimensionality Reduction using SVD**  
   - Singular Value Decomposition (SVD) is applied to reduce feature dimensionality while retaining essential information.  

3. **Training & Evaluation**  
   - Models are trained on processed features.  
   - **Accuracy, Precision, Recall, and F1-score** are used for performance evaluation.  

---

## 📊 **Performance Comparison**  
- The accuracy and loss curves for CNN and LSTM are visualized using **matplotlib**.  
- **Confusion matrices** are generated for all models to analyze misclassifications.  
- The impact of **SVD on model performance** is compared.  

---

## 🚀 **How to Run the Project**  

### 1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/your-username/Speech-Emotion-Recognition-SVD.git
cd Speech-Emotion-Recognition-SVD
```

### 2️⃣ **Install Dependencies**  
```bash
pip install -r requirements.txt
```

### 3️⃣ **Run the Jupyter Notebook**  
```bash
jupyter notebook
```
- Open `CNN.ipynb`, `LSTM.ipynb`, or `SVM.ipynb` in Jupyter Notebook.  
- Run the cells to preprocess data, extract features, train models, and visualize results.  

---

## 📌 **Technologies Used**  
- **Python**  
- **Librosa** (for audio processing)  
- **NumPy, Pandas**  
- **Scikit-Learn** (for SVM and evaluation metrics)  
- **TensorFlow/Keras** (for CNN & LSTM)  
- **Matplotlib, Seaborn** (for visualization)  

---

## 🔮 **Future Enhancements**  
- Implementing **transformer-based models** (like Wav2Vec, Whisper) for improved accuracy.  
- Expanding dataset coverage for multilingual speech recognition.  
- Experimenting with **Autoencoders** for unsupervised feature learning.  

---


## 📜 **License**  
This project is open-source and available under the **Apache License**.  

---
