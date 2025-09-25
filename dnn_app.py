# -----------------------------
# streamlit_app_upgraded.py
# -----------------------------
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sounddevice as sd
from io import BytesIO

# -----------------------------
# 1️⃣ Load Model
# -----------------------------
@st.cache_resource
def load_dnn_model(path='dnn_audio_classification.h5'):
    model = load_model(path)
    return model

model = load_dnn_model()

# -----------------------------
# 2️⃣ Feature Extraction Function
# -----------------------------
def extract_features(audio, sr=22050):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio)
    rms = librosa.feature.rms(y=audio)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    
    features = np.concatenate((
        np.mean(mfccs.T, axis=0),
        np.mean(spectral_centroid.T, axis=0),
        np.mean(zcr.T, axis=0),
        np.mean(rms.T, axis=0),
        np.mean(spectral_rolloff.T, axis=0)
    ))
    
    return features.reshape(1, -1)

# -----------------------------
# 3️⃣ Real-time Recording Function
# -----------------------------
def record_audio(duration=3, sr=22050):
    st.info("Recording audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten(), sr

# -----------------------------
# 4️⃣ Streamlit UI
# -----------------------------
st.title("🎶 Advanced Audio Classification")
st.write("Upload an audio file or record your voice to classify the sound.")

# Tabs for upload or record
tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

with tab1:
    uploaded_file = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'ogg'])
    
    if uploaded_file:
        audio, sr = librosa.load(uploaded_file, sr=22050)
        st.audio(uploaded_file, format='audio/wav')
        
        # Features
        features = extract_features(audio, sr)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        st.success(f"✅ Predicted Class: {predicted_class}")
        
        # Probability chart
        st.subheader("Prediction Probabilities")
        st.bar_chart(prediction.flatten())
        
        # Mel-spectrogram
        st.subheader("Mel-Spectrogram")
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(8,4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(plt)

with tab2:
    if st.button("Record 3 Seconds"):
        audio, sr = record_audio(duration=3)
        st.audio(audio, format='audio/wav', sample_rate=sr)
        
        # Features
        features = extract_features(audio, sr)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        st.success(f"✅ Predicted Class: {predicted_class}")
        
        # Probability chart
        st.subheader("Prediction Probabilities")
        st.bar_chart(prediction.flatten())
        
        # Mel-spectrogram
        st.subheader("Mel-Spectrogram")
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(8,4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(plt)
