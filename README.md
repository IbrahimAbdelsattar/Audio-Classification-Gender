# 🎶 Audio Classification Project

## 📌 Project Overview
This project focuses on building an **Audio Classification System** using machine learning and deep learning techniques. The main goal is to develop a model that can learn from raw audio signals, extract meaningful features, and accurately classify audio samples into their respective categories.

Audio classification is increasingly important in real-world applications such as:
- Voice assistants and speech recognition.
- Music genre classification and recommendation systems.
- Anomaly detection in machinery and industrial equipment.
- Emotion recognition from speech.

By automating audio classification, we can enable intelligent systems to understand and react to sounds efficiently.

---

## 🚀 Project Pipeline

The project follows a structured **end-to-end pipeline**:

### 1. Importing Libraries
We imported essential Python libraries for:
- Audio processing: `librosa`, `noisereduce`.
- Data handling: `pandas`, `numpy`.
- Visualization: `matplotlib`, `seaborn`.
- Machine learning and deep learning: `sklearn`, `tensorflow/keras`, `xgboost`.

### 2. Audio Preprocessing
- **Load Audio**: Read audio files using `librosa`.  
- **Trim Silence**: Remove unnecessary silent parts.  
- **Noise Reduction**: Reduce background noise.  
- **Normalization**: Standardize audio amplitude.  
- **Resampling**: Unify sample rates.  
- **Padding/Truncating**: Ensure equal length across samples.  

### 3. Feature Extraction
Extracted meaningful features to represent audio signals:
- Spectrogram (STFT)
- Mel-Spectrogram
- Spectral Centroid
- Zero Crossing Rate (ZCR)
- Spectral Rolloff
- MFCC (Mel-Frequency Cepstral Coefficients)
- RMS Energy

### 4. Final Preprocessing on Data
- Combined extracted features with labels into a structured dataset.
- Handled missing values and duplicates.
- Encoded categorical labels numerically.
- Scaled features using `StandardScaler`.
- Split dataset into **Training, Validation, and Test sets**.
- Calculated **class weights** to address class imbalance.

### 5. Exploratory Data Analysis (EDA)
- Visualized class distribution.
- Detected and handled class imbalance.
- Analyzed feature correlations.

### 6. Model Building
We trained and evaluated multiple models:
- **Classical ML Models**: SVM, XGBoost  
- **Deep Learning Models**: DNN, LSTM, GRU  
- **Advanced Models** (optional for future work): CNN on spectrograms, 1D-CNN on raw audio, ensemble models

### 7. Model Evaluation
Evaluation was performed on the **test set** using:
- **Accuracy**
- **Precision, Recall, F1-score**
- **Confusion Matrix**
- **Class-wise performance analysis**

**Summary of Results (Test Set):**

| Model   | Accuracy | Macro F1 | Weighted F1 | Minority Class Recall | Majority Class Recall |
|---------|---------|-----------|-------------|---------------------|---------------------|
| SVM     | 0.90    | 0.74      | 0.91        | 0.77                | 0.91                |
| XGBoost | 0.95    | 0.73      | 0.94        | 0.32                | 1.00                |
| DNN     | 0.93    | 0.80      | 0.94        | 0.76                | 0.95                |
| LSTM    | 0.87    | 0.69      | 0.89        | 0.68                | 0.89                |
| GRU     | 0.91    | 0.72      | 0.91        | 0.57                | 0.94                |

**✅ Best Overall Model:**  
- **DNN** provides the best balance between minority and majority class performance, with **high accuracy (0.93) and weighted F1-score (0.94)**.  
- XGBoost excels in overall accuracy but underperforms on minority class.  
- LSTM and GRU are slightly less effective, but GRU performs better than LSTM.

---

## ⚡ Advanced Improvements
- **CNN on Spectrograms**: Capture spatial patterns in audio.  
- **1D-CNN on Raw Audio**: End-to-end learning from waveform.  
- **Ensemble Models**: Combine multiple models for robust predictions.  
- **Real-Time Audio Classification**: Deploy models for live prediction with Streamlit.

---

## 🛠️ Deployment
- DNN model saved as `dnn_audio_classification.h5`.
- Professional **Streamlit web app** developed for interactive predictions:
  - Upload audio files or record live audio.
  - View prediction probabilities.
  - Visualize Mel-spectrogram.
- Future deployment can include **ensemble predictions** for higher accuracy.

---

## 📁 Folder Structure (Recommended)
audio_classification_project/
│
├─ data/ # Audio datasets
├─ notebooks/ # Jupyter notebooks for analysis
├─ models/ # Saved DNN/XGBoost models
├─ streamlit_app.py # Streamlit deployment script
├─ README.md # Project explanation
└─ requirements.txt # Required Python packages


---

## 📌 References
- [Librosa Documentation](https://librosa.org/doc/latest/index.html) – Audio processing in Python  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/) – Machine learning utilities  
- [TensorFlow / Keras](https://www.tensorflow.org/) – Deep learning framework  
- [XGBoost](https://xgboost.readthedocs.io/en/stable/) – Gradient boosting for ML  

---

## 💡 Conclusion
This project demonstrates an **end-to-end pipeline** for audio classification from preprocessing to model deployment.  
The trained DNN provides **robust performance across classes**, and the Streamlit app allows **real-time predictions**.  

Future improvements can include **CNN-based models**, **ensemble learning**, and deployment on web or mobile platforms.
