import streamlit as st
import requests
import base64
from dotenv import load_dotenv
from openai import OpenAI
import subprocess
import json
import singlestoredb as s2
from pydub import AudioSegment
import io  # Import the io module

# Load environment variables from .env file
load_dotenv()
LLM="gemini-2.0-flash-exp"
# LLM="gemini-1.5-flash"
# Get the API key from environment variables
subscription_key = "731bcfac-aef2-4541-88ee-1dc114b017a4"
xai_api_key = "AIzaSyA02JdZFZ3Xjj26ThJhhJQ7anhrbrI66h8"
#xai_api_key = "AIzaSyALeNep6neidGStoM1qax9GAxu1_GFNLIo"
sarvamurl = "https://api.sarvam.ai/text-to-speech"
sarvamheaders = {
    "accept": "application/json",
    "content-type": "application/json",
    "api-subscription-key": subscription_key
}

# Initialize OpenAI client
LLMclient=OpenAI(
  api_key=xai_api_key,
  base_url="https://generativelanguage.googleapis.com/v1beta/openai/"

)
Nclient = OpenAI(
  api_key="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",
  base_url="https://integrate.api.nvidia.com/v1"
)

# Load CSS from file
with open("style.css", "r") as f:
    css = f.read()

# Add background image to CSS
css += """
body {
    background-image: url('https://fillyou.in/wp-content/uploads/2024/01/service_6-scaled-1.webp'); /* Update with your image path */
    background-size: cover; /* Cover the entire page */
    background-position: center; /* Center the image */
    background-repeat: no-repeat; /* Prevent the image from repeating */
    height: 100vh; /* Ensure the body takes full height */
}
"""

# Inject CSS into Streamlit
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
def create_connection():
    return s2.connect('admin:X8MBbWxI1NiuG6RGPhyIQcr7lz4oseOY@svc-8bd4e6d7-dd92-449e-b8af-56828e3aea12-dml.aws-mumbai-1.svc.singlestore.com:3306/miniDB')

# Function to send audio to Sarvam API for speech recognition
def transcribe_audio(audio_file, subscription_key, language_code):
    try:
        audio = AudioSegment.from_file(io.BytesIO(audio_file.read()))
        duration_ms = len(audio)
        if duration_ms > 30000:  # 30 seconds limit
            full_transcript = ""
            start = 0
            while start < duration_ms:
                end = min(start + 30000, duration_ms)
                chunk = audio[start:end]
                
                # Export the chunk to a BytesIO object
                chunk_io = io.BytesIO()
                chunk.export(chunk_io, format="wav")
                chunk_io.seek(0)  # Reset the buffer position to the beginning
                
                # Prepare files for the request
                files = [('file', ('audio_chunk.wav', chunk_io, 'audio/wav'))]
                
                # Set headers including your API subscription key
                headers = {
                    'api-subscription-key': subscription_key
                }
                # Prepare the payload with model and language code
                payload = {
                    'model': 'saarika:v1',
                    'language_code': language_code,
                    'with_timesteps': 'false'
                }

                # Make the POST request to the API
                response = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, data=payload, files=files)

                if response.status_code == 200:
                    full_transcript += response.json().get('transcript', '') + " "
                else:
                    return f"Error: {response.status_code}, {response.text}"

                start += 30000  # Move to the next chunk
            
            return full_transcript.strip()
        else:
            # For audio files shorter than 30 seconds
            files = [('file', ('audio.wav', audio_file, 'audio/wav'))]
            headers = {
                'api-subscription-key': subscription_key
            }
            payload = {
                'model': 'saarika:v1',
                'language_code': language_code,
                'with_timesteps': 'false'
            }
            response = requests.post("https://api.sarvam.ai/speech-to-text", headers=headers, data=payload, files=files)

            if response.status_code == 200:
                return response.json().get('transcript', 'No transcript available.')
            else:
                return f"Error: {response.status_code}, {response.text}"
    except Exception as e:
        return f"An error occurred during transcription: {str(e)}"

# Function to get embeddings and nearest neighbors
def get_embeddings_and_neighbors(sentence):
    response = Nclient.embeddings.create(
        input=[sentence],
        model="baai/bge-m3",
        encoding_format="float",
        extra_body={"truncate": "NONE"}
    )
    embeddings_json = json.dumps(response.data[0].embedding)  # Convert to JSON

    # Insert embeddings into SingleStoreDB
    with create_connection() as conn:
        with conn.cursor() as cur:
            # Search for nearest neighbors using the new query format
            cur.execute(""" 
                SELECT id, llm_generated_idea, text, embedding, embedding <-> %s AS score 
                FROM webgen 
                ORDER BY score 
                LIMIT 5
            """, (embeddings_json,))  # Pass embeddings as a JSON array

            # Fetch the results
            results = cur.fetchall()
            with open("embeddings.txt", "w") as f:
                for row in results:
                    f.write(str(row) + "\n")  # Write each row as a string to the file
            
            return results  # Return the fetched results

# Function to process text using OpenAI API
def process_with_llm(text, html_content):
    try:
        response = LLMclient.chat.completions.create(
            model=LLM,
            messages=[
                
                    {"role": "system", "content": "You are a helpful software engineer, you will write HTML with TailWind CSS code embedded into it.Give content from html start tag to end tag. Implement proper scroll for the webpage. Even if the user has given very little description, use your creatvity and make a good complete and large webpage.You will have to write the code for the frontend. If there is a website already, the code of that website will also be given to you. Make top and side pop up navigation bars,  proper scroll functionality, back button, etc. Use animations for all photos, text, buttons, everything. Have some background theme and colors do not keep it blank white. Have a lot of content on the webpage, do not make a plain simple empty website. Include images to make the website look good wherever neceessary. Maximum 10 images. To include images use https://image.pollinations.ai/prompt/<imagedescription>.png for the image url. Write code such that the bottom 10% of the image is not visble, as that part contains watermark. You can use image for background as well, make sure it is not repeating and is rendered properly. Crop the images into different size and shapes. Your response should consist of 2 parts seperated by 2 new lines. first part will be A reply message to the user in the input language saying how you have completed the work. 2nd part will be the code"},
                    {"role": "user", "content": f"User Instruction: {text}. Reference Code: {html_content} You can refer the given code for ideas."}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {str(e)}"# Languag
# Language code mapping
language_mapping = {
    "Kannada": "kn-IN",
    "Hindi": "hi-IN",
    "Bengali": "bn-IN",

    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Odia": "od-IN",
    "Punjabi": "pa-IN",
    "Tamil": "ta-IN",
    "Telugu": "te-IN",
    "Gujarati": "gu-IN"
}

# Streamlit UI
st.title("Design Companion")

# Move language selection to sidebar
with st.sidebar:
    selected_language = st.selectbox("Select Language:", 
                                  list(language_mapping.keys()), 
                                  index=0)
    language_code = language_mapping[selected_language]
audio_value = st.audio_input("Record a voice message")

if audio_value:
    if subscription_key:
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio(audio_value, subscription_key, language_code)
            st.write(f"User 🙋 : {transcript}")

            # Fetch embeddings and nearest neighbors
            results = get_embeddings_and_neighbors(transcript)  # Get embeddings and neighbors
            if results:
                # Assuming results contain the HTML content from the nearest neighbor
                html_content = results[0][2]  # Get the HTML content from the first result
                sarvam_message = process_with_llm(transcript, html_content)  # Pass both transcript and HTML content
                with open("log.txt", "w", encoding='utf-8') as f:
                    f.write(sarvam_message)                 
                st.write(f"Model 🤖 : {sarvam_message.split('\n\n')[0]}")
                
                # Extract HTML code from sarvam_message into a new variable
                html_code = ""
                start_marker = "```html"
                end_marker = "```"

                # Find the start and end of the HTML code
                start_index = sarvam_message.find(start_marker)
                end_index = sarvam_message.find(end_marker, start_index + len(start_marker))

                if start_index != -1 and end_index != -1:
                    # Extract the HTML code substring
                    html_code = sarvam_message[start_index + len(start_marker):end_index].strip()

                # Save the extracted HTML code to an HTML file with utf-8 encoding
                with open("generated_app.html", "w", encoding='utf-8') as f:
                    f.write(html_code)  # Save the new variable containing the HTML code

                # Run the HTML file in a web browser
                subprocess.run(["start", "generated_app.html"], shell=True)  # Use 'start' to open in default browser... # Prepare payload for Sarvam AI
                payload = {
                    "inputs": [sarvam_message.split('\n\n')[0][:490]],  # Limit the message length and take content before the first newline
                    "target_language_code": language_code,
                    "speaker": "meera",
                    "pitch": 0.2,
                    "pace": 1.1,
                    "loudness": 0.8,
                    "enable_preprocessing": True,
                    "model": "bulbul:v1",
                    "speech_sample_rate": 8000
                }
                response = requests.request("POST", sarvamurl, json=payload, headers=sarvamheaders)
                audio_data = response.json()
                if "audios" in audio_data and audio_data["audios"]:
                    audio_bytes = base64.b64decode(audio_data["audios"][0])
                    st.markdown('<div class="st-ae">', unsafe_allow_html=True)
                    st.audio(audio_bytes, format="audio/wav", autoplay=True)
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No nearest neighbors found.")
    else:
        st.error("API Subscription Key not found in environment variables.")

