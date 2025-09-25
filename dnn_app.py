# -----------------------------
# streamlit_audio_dnn.py
# -----------------------------
import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from io import BytesIO
import sounddevice as sd
import wavio

# -----------------------------
# 1️⃣ Load DNN Model
# -----------------------------
@st.cache_resource
def load_dnn_model(path='dnn_model.h5'):
    model = load_model(path)
    return model

model = load_dnn_model()

# -----------------------------
# 2️⃣ Feature Extraction
# -----------------------------
def extract_features(audio, sr=22050, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)

    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_spec_mean = np.mean(mel_spec, axis=1)

    centroid_mean = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    rolloff_mean = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    zcr_mean = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms_mean = np.mean(librosa.feature.rms(y=audio))

    feature_vector = np.hstack([
        mfccs_mean,
        mel_spec_mean,
        centroid_mean,
        rolloff_mean,
        zcr_mean,
        rms_mean
    ])

    return feature_vector.reshape(1, -1)

# -----------------------------
# 3️⃣ Audio Recording Function
# -----------------------------
def record_audio(duration=3, sr=22050):
    st.info(f"Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    audio = audio.flatten()
    return audio, sr

# -----------------------------
# 4️⃣ Streamlit UI
# -----------------------------
st.set_page_config(page_title="Audio Classification", page_icon="🎵")
st.title("🎶 Audio Gender Classification")
st.write("Upload a file or record your voice and classify gender using DNN.")

tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])

# -----------------------------
# 5️⃣ Upload Audio
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload your audio file (wav, mp3, ogg)", type=["wav", "mp3", "ogg"])
    if uploaded_file:
        y, sr = librosa.load(uploaded_file, sr=22050)
        st.audio(uploaded_file, format='audio/wav')

        features = extract_features(y, sr)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.success(f"✅ Predicted Class: {predicted_class}")
        st.subheader("Prediction Probabilities")
        st.bar_chart(prediction.flatten())

        st.subheader("Mel-Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(8,4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(plt)

# -----------------------------
# 6️⃣ Record Audio
# -----------------------------
with tab2:
    duration = st.slider("Recording Duration (seconds)", min_value=1, max_value=10, value=3)
    if st.button("Record & Classify"):
        y, sr = record_audio(duration)
        st.audio(y, format='audio/wav', sample_rate=sr)

        features = extract_features(y, sr)
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]

        st.success(f"✅ Predicted Class: {predicted_class}")
        st.subheader("Prediction Probabilities")
        st.bar_chart(prediction.flatten())

        st.subheader("Mel-Spectrogram")
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        plt.figure(figsize=(8,4))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        st.pyplot(plt)
