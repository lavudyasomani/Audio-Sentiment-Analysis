import streamlit as st
import os
import speech_recognition as sr
import pickle
from textblob import TextBlob
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Constants
UPLOAD_FOLDER = 'uploads'
MAX_SEQUENCE_LENGTH = 100  # Maximum sequence length for padding

# Load the tokenizer
TOKENIZER_FILE = 'tokenizer.pickle'
with open(TOKENIZER_FILE, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to transcribe audio
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)
    except sr.RequestError as e:
        st.error(f"Speech recognition request failed: {e}")
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except Exception as ex:
        st.error(f"An error occurred during audio processing: {ex}")

# Function to preprocess text using the tokenizer
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
    return padded_sequence

# Function to analyze sentiment using the pre-trained model
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    if sentiment > 0:
        return "Positive"
    elif sentiment < 0:
        return "Negative"
    else:
        return "Neutral"

def main():
    st.title('Audio Sentiment Analysis')

    # File uploader widget
    uploaded_file = st.file_uploader("Upload Audio File")

    if uploaded_file is not None:
        # Save the uploaded file
        with open(os.path.join(UPLOAD_FOLDER, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getvalue())

        # Transcribe audio
        transcribed_text = transcribe_audio(os.path.join(UPLOAD_FOLDER, uploaded_file.name))

        # Preprocess transcribed text using the tokenizer
        preprocessed_text = preprocess_text(transcribed_text)

        # Analyze sentiment
        sentiment = analyze_sentiment(transcribed_text)

        # Display sentiment result with customized styling
        st.markdown(f'<p style="font-size:24px; color:yellow;">Sentiment: {sentiment}</p>', unsafe_allow_html=True)

        # Provide the option to play the audio
        st.audio(open(os.path.join(UPLOAD_FOLDER, uploaded_file.name), 'rb'), format='audio/wav')

if __name__ == '__main__':
    main()
