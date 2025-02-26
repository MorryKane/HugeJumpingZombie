import base64
import os
import tempfile
import keyboard
import sounddevice as sd
import soundfile as sf
import numpy as np
import pyautogui  # for taking screenshots
from openai import OpenAI
from playsound import playsound
import tkinter as tk
import threading

# Verify OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found! Set it as an environment variable.")

client = OpenAI()
global iteration
iteration = 0

# Global conversation memory for chat history
conversation = [
    {
        "role": "system",
        "content": "You are a helpful assistant. You need to instruct the user on how to operate the Android phone step by step."
    }
]

def record_audio_until_key_release(sample_rate=16000):
    """Record audio until the F8 key is released."""
    print("\nRecording... Speak now!")
    recorded_frames = []

    def callback(indata, frames, time, status):
        recorded_frames.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate,
                        channels=1,
                        dtype='int16',
                        callback=callback):
        while keyboard.is_pressed("f8"):
            sd.sleep(50)  # check every 50ms

    audio_data = np.concatenate(recorded_frames, axis=0)
    return audio_data, sample_rate

def transcribe_audio(audio_file):
    """Convert speech to text using OpenAI's Whisper ASR."""
    with open(audio_file, "rb") as f:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    return transcription.text.strip()

def take_screenshot():
    """Take a screenshot and save it to a temporary PNG file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_png:
        screenshot = pyautogui.screenshot()
        screenshot.save(tmp_png.name)
        return tmp_png.name

def encode_image(image_path):
    """Encode the image file as a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def get_response_with_memory(request_text, screenshot_file):
    global conversation
    # Append only the text portion to the conversation history
    conversation.append({"role": "user", "content": request_text})
    
    # Prepare the composite message using the structured format:
    base64_image = encode_image(screenshot_file)
    message_content = [
        {
            "type": "text",
            "text": request_text,
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
        },
    ]
    
    # Build the payload: use the conversation memory (text only)
    # and then add the composite message for this turn.
    messages_payload = conversation.copy()  # all previous text-only messages
    messages_payload.append({"role": "user", "content": message_content})
    
    completion = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:personal::B56tSE3Q",
        messages=messages_payload
    )
    assistant_reply = completion.choices[0].message
    conversation.append({"role": "assistant", "content": assistant_reply.content})
    return assistant_reply.content.strip()

def text_to_speech(text):
    """Convert text to speech and save to a temporary MP3 file."""
    with client.audio.speech.with_streaming_response.create(
        model="tts-1",
        voice="alloy",
        input=text
    ) as response:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            response.stream_to_file(f.name)
            return f.name

def show_overlay(center_x, center_y, rect_length, rect_width, duration=5000):
    """
    Create an overlay window that draws a red rectangle with a totally transparent background.
    
    Arguments:
    - center_x, center_y: Center coordinates of the rectangle on the screen.
    - rect_length: The horizontal dimension (width in pixels) of the rectangle.
    - rect_width: The vertical dimension (height in pixels) of the rectangle.
    - duration: How long (in milliseconds) the overlay remains visible.
    """
    overlay = tk.Tk()
    overlay.overrideredirect(True)  # Remove window borders
    overlay.attributes("-topmost", True)
    
    # Set a unique background color and mark it as transparent.
    transparent_color = "magenta"
    overlay.config(bg=transparent_color)
    overlay.wm_attributes("-transparentcolor", transparent_color)
    
    # Calculate top-left corner so the rectangle is centered at (center_x, center_y)
    top_left_x = center_x - rect_length // 2
    top_left_y = center_y - rect_width // 2
    overlay.geometry(f"{rect_length}x{rect_width}+{top_left_x}+{top_left_y}")
    
    # Create a canvas with the same transparent background.
    canvas = tk.Canvas(overlay, width=rect_length, height=rect_width, 
                       highlightthickness=0, bg=transparent_color)
    canvas.pack()
    
    # Draw a red rectangle outline. Only the outline will be visible.
    canvas.create_rectangle(0, 0, rect_length, rect_width, outline="red", width=5)
    
    # Close the overlay after 'duration' milliseconds.
    overlay.after(duration, overlay.destroy)
    overlay.mainloop()

def process_audio():
    """Complete pipeline: record, transcribe, take screenshot, query, playback, and overlay."""
    global iteration
    try:
        # Record until F8 is released
        audio, sr = record_audio_until_key_release()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            sf.write(tmp_wav.name, audio, sr)
        
        # Transcribe the spoken request
        request_text = transcribe_audio(tmp_wav.name)
        os.unlink(tmp_wav.name)
        if not request_text:
            print("No speech detected")
            return
        print(f"Recognized request: {request_text}")
        
        # Take a screenshot
        screenshot_file = take_screenshot()
        
        # Get response from the AI using both the request and the screenshot,
        # while also updating the conversation memory.
        response_text = get_response_with_memory(request_text, screenshot_file)
        print(f"Response: {response_text}")
        os.unlink(screenshot_file)
        
        # Convert the response to speech and get the temporary MP3 file
        speech_file = text_to_speech(response_text)
        
        # Start playing the audio in a separate thread so it runs concurrently
        audio_thread = threading.Thread(target=playsound, args=(speech_file,))
        audio_thread.start()
        
        # Wait for the audio thread to finish, then delete the file
        audio_thread.join()
        os.unlink(speech_file)
        
        # Cycle the iteration variable so the next overlay is used next time.
        iteration = iteration + 1
        
    except Exception as e:
        print(f"Error: {str(e)}")

def debug_method():
    # For debugging, trigger the overlay directly.
    show_overlay(center_x=1050, center_y=350, rect_length=200, rect_width=200, duration=5000)

# Set up hotkey: hold F8 to trigger the overlay (or process the full audio pipeline).
keyboard.add_hotkey('f8', process_audio)
print("Press and hold F8 to trigger the overlay. Press ESC to quit.")
keyboard.wait('esc')
print("\nExiting program...")
