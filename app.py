import streamlit as st
import joblib
import nltk
from time import sleep
from nltk.tokenize import word_tokenize
import string
from gtts import gTTS
import tempfile

# Ensure the NLTK data path is set correctly to the folder where resources are stored
nltk.data.path.append(r"C:/Users/ADNAN/AppData/Roaming/nltk_data")

# Download necessary NLTK resources if they are not already downloaded
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the saved Naive Bayes model and TF-IDF vectorizer
model = joblib.load('naive_bayes_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define the preprocessing function to tokenize and clean the text
def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove punctuation and lowercase all words
    tokens = [word.lower() for word in tokens if word not in string.punctuation]
    
    return tokens

# Define a function to preprocess and predict sentiment
def predict_sentiment(review_text):
    # Preprocess the review text
    tokens = preprocess_text(review_text)
    
    # Join tokens into a string for TF-IDF vectorizer
    review_text = ' '.join(tokens)
    
    # Transform the review text using the TF-IDF vectorizer
    review_tfidf = tfidf_vectorizer.transform([review_text])
    
    # Predict sentiment using the trained model
    sentiment = model.predict(review_tfidf)
    
    # Return sentiment: Positive (1) or Negative (0)
    if sentiment == 1:
        return "🎉 Wow! This review is a **Positive** experience! Keep it up! 😃"
    else:
        return "😞 Oops! This review is **Negative**. Maybe next time! 😔"

# Function to convert text to speech using gTTS
def text_to_speech(text):
    # Use gTTS to convert text to speech
    tts = gTTS(text=text, lang='en', slow=False)
    
    # Save the speech to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    tts.save(temp_file.name)
    
    return temp_file.name

# Set page layout and title
st.set_page_config(page_title="DK's Sentiment Analysis", page_icon="💬", layout="wide")

# Sidebar for additional buttons and feedback form
with st.sidebar:
    # Sentiment analysis image in sidebar
    st.image("download.png", width=200)

    
 
    
    # Buttons in the sidebar
    about_button = st.button("About the App")
    feedback = st.text_area("We'd love to hear your feedback on the app!", height=100)
    feedback_submit = st.button("Submit Feedback")

    if feedback_submit and feedback:
        st.success("Thank you for your feedback! 🙌")
    elif feedback_submit:
        st.warning("Please write some feedback before submitting.")

# Main content area (Light Background for better contrast)
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;  /* Light background for main content */
        font-family: 'Arial', sans-serif;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #333;
    }
    .stTextInput>div>div>input {
        font-size: 18px;
        color: #333;
    }
    .stButton>button {
        background-color: #FF5733;  /* Red button */
        color: white;
        font-size: 18px;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #C70039;  /* Darker red on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Main app title and welcome message with robot emoji
st.markdown("<h1 style='text-align: center; color: #5D5D5D;'>🤖 Welcome to DK's Sentiment Analysis!</h1>", unsafe_allow_html=True)

# Add the description directly under the title in the main content
st.markdown("""
    <p style="font-size: 24px; color: #555; text-align: left;">
    <strong>Hey!</strong> It's <strong>DK's Sentiment Analysis</strong> 😎  
    Type your review in the box below and let's see if it's positive or negative!  
    We use advanced machine learning techniques to analyze reviews and give quick feedback.
    </p>
""", unsafe_allow_html=True)

# Convert the welcome message to speech and play it
welcome_text = """
Hey! It's DK's Sentiment Analysis 😎  
Type your review in the box below and let's see if it's positive or negative! 
We use advanced machine learning techniques to analyze reviews and give quick feedback.
"""
welcome_audio = text_to_speech(welcome_text)
st.audio(welcome_audio, format='audio/mp3')

# Show "About the App" content when button is clicked
if about_button:
    about_text = """
    This is a sentiment analysis app developed by Muhammad Adnan, a student at NUST with a strong interest in Natural Language Processing (NLP). 
    The purpose of this app is to help develop a strong foundation in NLP and build a machine learning model for sentiment analysis of movie reviews.  
    DK (Adnan) created this app to showcase his skills and interest in the NLP field. 🎓
    """
    st.markdown(about_text)
    
    # Convert the About text to speech and play it
    about_audio = text_to_speech(about_text)
    st.audio(about_audio, format='audio/mp3')

# Input box for the user to enter a review
user_review = st.text_area("Type your movie review below!", height=150, max_chars=500)

# Add a button for prediction
if st.button("Predict Sentiment"):
    if user_review:
        # Show a progress bar while analyzing
        with st.spinner("Analyzing sentiment..."):
            sleep(2)  # Simulating some delay for a better user experience
            
            # Get the prediction
            sentiment_output = predict_sentiment(user_review)
            
            # Display the result with styled text
            st.markdown(f"<h3 style='text-align: center; color: red;'>{sentiment_output}</h3>", unsafe_allow_html=True)
            
            # Convert the sentiment result to speech and play it
            audio_file = text_to_speech(sentiment_output)
            st.audio(audio_file, format='audio/mp3')
    else:
        st.warning("Please enter a review to analyze.")
