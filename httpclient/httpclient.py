import os
import openai
import keyboard
import sounddevice as sd
import numpy as np
import tempfile
import time
import threading
import requests  # Make sure to install requests (pip install requests)
from scipy.io.wavfile import write
from playsound import playsound

# OpenAI API Key Setup
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found! Set it as an environment variable.")
openai.api_key = OPENAI_API_KEY

# Audio Configuration
SAMPLE_RATE = 44100  # 44.1 kHz
CHANNELS = 1  # Mono
RECORDING = False
audio_data = []  # Stores recorded audio samples

def start_recording():
    global RECORDING, audio_data
    if not RECORDING:
        print("Recording started... Speak now!")
        RECORDING = True
        audio_data = []  # Reset previous audio
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=callback):
            while RECORDING:
                time.sleep(0.1)

def stop_recording():
    global RECORDING
    if RECORDING:
        RECORDING = False
        print("Recording stopped. Processing audio...")
        threading.Thread(target=process_audio, daemon=True).start()

def callback(indata, frames, time_info, status):
    if status:
        print(status)
    if RECORDING:
        audio_data.append(indata.copy())

def process_audio():
    global audio_data
    if not audio_data:
        print("No audio recorded!")
        return

    # Concatenate all recorded numpy arrays
    audio_array = np.concatenate(audio_data, axis=0)

    # Save to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_filename = temp_audio_file.name
        write(temp_filename, SAMPLE_RATE, (audio_array * 32767).astype(np.int16))

    # Transcribe audio using OpenAI Whisper
    with open(temp_filename, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    os.remove(temp_filename)
    transcribed_text = transcript.strip()
    print(f"Transcribed: {transcribed_text}")

    if transcribed_text:
        get_gpt_response(transcribed_text)

def query_http_server(identifier):
    """Query the HTTP server to retrieve additional information based on the identifier."""
    try:
        # Adjust the URL and payload as needed for your HTTP server
        response = requests.post("http://localhost:5000/data", json={"query": identifier})
        if response.status_code == 200:
            return response.json().get("response", "No additional info found")
        else:
            return "Error: HTTP server returned status code {}".format(response.status_code)
    except Exception as e:
        return f"Error connecting to HTTP server: {e}"

def get_gpt_response(text):
    """
    Two-step approach:
    1. Ask GPT‑4o whether additional info is needed.
    2. If yes, query the HTTP server and include that data in a second GPT‑4o call.
    """
    # Step 1: Determine if extra info is needed
    prompt_decision = (
        f"Based on the input: '{text}', first decide if additional data from an external HTTP server is necessary. "
        "If additional data is needed, respond with 'REQUEST: <identifier>' where <identifier> is the key to query. "
        "If not needed, respond to the input directly with 'DIRECT: <your final answer>'. This response should not mention your decision of additional data."
    )
    decision_response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_decision}]
    )
    decision_text = decision_response.choices[0].message.content.strip()
    print(f"GPT-4o decision: {decision_text}")

    if decision_text.startswith("DIRECT:"):
        # Direct response – no extra info required
        final_response = decision_text[len("DIRECT:"):].strip()
        print(f"GPT-4o (direct): {final_response}")
        text_to_speech(final_response)
    elif decision_text.startswith("REQUEST:"):
        # Extra info is required. Extract the identifier and query the HTTP server.
        identifier = decision_text[len("REQUEST:"):].strip()
        additional_info = query_http_server(identifier)
        print(f"Additional info from HTTP server: {additional_info}")

        # Step 2: Use the additional info in a second GPT‑4o call
        prompt_final = (
            f"Given the user prompt: '{text}' and the following additional data: '{additional_info}' related to {identifier}, "
            "please provide a complete and final response."
        )
        final_response_obj = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt_final}]
        )
        final_response = final_response_obj.choices[0].message.content.strip()
        print(f"GPT-4o (with additional info): {final_response}")
        text_to_speech(final_response)
    else:
        print("Unexpected response format from GPT-4o.")

def text_to_speech(text):
    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_filename = temp_audio_file.name
        temp_audio_file.write(response.content)
    playsound(temp_filename)

# Key bindings: Hold space to record, release to stop and process
KEY_TO_HOLD = "space"
keyboard.on_press_key(KEY_TO_HOLD, lambda _: threading.Thread(target=start_recording, daemon=True).start())
keyboard.on_release_key(KEY_TO_HOLD, lambda _: threading.Thread(target=stop_recording, daemon=True).start())

print(f"Hold '{KEY_TO_HOLD}' to start speaking, release to transcribe and respond!")
keyboard.wait("esc")
