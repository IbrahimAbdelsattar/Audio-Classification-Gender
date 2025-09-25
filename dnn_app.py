# -----------------------------
# streamlit_app_stable.py
# -----------------------------
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# -----------------------------
# 1️⃣ Load Model
# -----------------------------
@st.cache_resource
def load_dnn_model(path='dnn_model.h5'):
    model = load_model(path)
    return model

model = load_dnn_model()

# -----------------------------
# 2️⃣ Feature Extraction Function
# -----------------------------
def extract_features(audio, sr=22050):
    # 1️⃣ MFCCs: 13
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)  # shape (13,)

    # 2️⃣ Spectral Centroid: 12 ? (تقدر تعمل mean على frames) -> 1 value
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid.T, axis=0)  # shape (frames,)

    # 3️⃣ Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr.T, axis=0)

    # 4️⃣ RMS
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms.T, axis=0)

    # 5️⃣ Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    spectral_rolloff_mean = np.mean(spectral_rolloff.T, axis=0)

    features = np.concatenate((
        mfccs_mean, 
        spectral_centroid_mean[:12],  # ناخد اول 12 frame او نعمل padding
        zcr_mean, 
        rms_mean, 
        spectral_rolloff_mean
    ))

    # padding لو العدد مش 57
    if features.shape[0] < 57:
        features = np.pad(features, (0, 57 - features.shape[0]), mode='constant')
    elif features.shape[0] > 57:
        features = features[:57]

    return features.reshape(1, -1)


# -----------------------------
# 3️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Audio Classification App", page_icon="🎵")
st.title("🎶 Advanced Audio Classification")
st.write("Upload an audio file and the model will predict the gender based on voice.")

# -----------------------------
# Upload Audio
# -----------------------------
uploaded_file = st.file_uploader("Upload your audio file (wav, mp3, ogg)", type=["wav", "mp3", "ogg"])

if uploaded_file:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=22050)
    st.audio(uploaded_file, format='audio/wav')

    # Extract features and predict
    features = extract_features(y, sr)
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction, axis=1)[0]

    st.success(f"✅ Predicted Class: {predicted_class}")

    # Display prediction probabilities
    st.subheader("Prediction Probabilities")
    st.bar_chart(prediction.flatten())

    # Display Mel-Spectrogram
    st.subheader("Mel-Spectrogram")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    st.pyplot(plt)

