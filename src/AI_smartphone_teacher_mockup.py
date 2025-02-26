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

def get_response(request_text, screenshot_file):
    """
    Encode the screenshot and then send the spoken request along with the image
    to the language model as a structured message, including a system prompt that
    instructs the assistant to show detailed steps.
    """
    global iteration
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

    prompts = [
                [
                    (
                    "The user will ask about if a restaurant is open today. You should first ask if the user remembers the name of the restaurant. Make your answer short. "
                    )
                ],
                [
                    (
                    "In the previous session the user asked about if a restaurant is open today. You let the user confirm the name."
                    "The user will then say something about the name of the restaurant, and you should first acknowledge it, then invite the user to search for it together"
                    "Then, you need to confirm if the user has Google Maps"
                    )
                ],
                [
                    (
                    "In the previous session you confirmed if the user has Google maps installed, and the user will answer no."
                    "You should first acknowledge that, then instruct the user to open Play Store."
                    "You should also mention that a red rectangle will mark the location of the icon."
                    "Then You should provide information about the position and appearance of the Play Store icon on the image (the smartphone screen). Assume it is a smartphone and don't mention Bluestacks."
                    "Then that's it. Don't say anything about what needs to be done after entering the Play Store including how to install google maps."
                    )
                ],
                [
                    (
                    "In the previous session you asked the user to open Play Store and the user successully opened it and will ask what's next. "
                    "You should first acknowledge that the user has opened Play Store, then instruct the user to tap once on the top bar labeled 'Search apps & games'."
                    "You should mention that a red rectangle will mark the location of the bar."
                    "Then that's it. Don't say anything about what to do afterwards."
                    )
                ],
                [
                    (
                    "In the previous session you asked the user to open search bar on Play Store. The user will ask about what to do next. "
                    "You should first acknowledge that the user has opened the search bar, then instruct the user to type 'Google Maps' in the search bar and tap on the Google Maps appearing below."
                    "Then that's it. Don't say anything about what to do afterwards."
                    )
                ],
                [
                    (
                    "In the previous session you asked the user to search for Google maps. The user did that and will ask what to do next. "
                    "You should first mention that it seems Google Maps has already been installed and the user may just need to open it. "
                    "Then you should instruct the user to open Google Maps by tapping on the Open button, also mention it is marked in the red rectangle."
                    "Then that's it. Don't say anything about what to do afterwards."
                    )
                ],
                [
                    (
                    "In the previous session you asked the user to open Google maps. The user did that and will ask what to do next. "
                    "You should first acknowledge that Google Maps is opened, then instruct the user to tap once on the top bar labeled 'Search here'."
                    "You should mention that a red rectangle will mark the location of the bar."
                    "Then that's it. Don't say anything about what to do afterwards."
                    )
                ],
                [
                    (
                    "In the previous session you asked the user to open search bar on Google maps. The user did that and will ask what to do next. "
                    "You should first acknowledge that, then instruct the user to input shinjuku restaurant ban."
                    "Then that's it. Don't say anything about what to do afterwards."
                    )
                ],
                [
                    (
                    "In the previous session you asked the user to input shinjuku restaurant ban. The user did that and will start saything something like this ban thai restaurant is what "
                    "he was looking for."
                    "You should first acknowledge that, then instruct the user to tap on this ban thai restaurant."
                    "Then that's it. Don't say anything about what to do afterwards."
                    )
                ],
                [
                    (
                    "In the previous session you asked the user to enter the page of a restaurant on Google maps. The user did that and will ask what's next"
                    "You should first acknowledge that, then tell the user if the restaurant is open or not. If open, you should also tell when it will close."
                    "You should also tell that the user may check the following pictures to make sure it is the same restauarnt"
                    "Then that's it. Don't say anything about what to do afterwards."
                    )
                ],
                [
                    (
                    "The user will say thank you because you finished a task for the user in a previous session. You may just respond and say if you need help with anything else or something."
                    )
                ],
            ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
       # model="ft:gpt-4o-2024-08-06:personal::B56tSE3Q",  # Adjust this to your model that supports images
        messages=[
            {
                "role": "system",
                "content": prompts[iteration],
                #'content': ("You are a helpful assistant. You need to instruct the user on how to operate the Android phone step by step.")
            },
            {
                "role": "user",
                "content": message_content
            }
        ],
    )
    return response.choices[0].message.content.strip()

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
        
        # Get response from the AI using both the request and the screenshot
        response_text = get_response(request_text, screenshot_file)
        print(f"Response: {response_text}")
        os.unlink(screenshot_file)
        
        # Convert the response to speech and get the temporary MP3 file
        speech_file = text_to_speech(response_text)
        
        # Define your overlay sequence list
        overlay_sequence = [
            {"center_x": 1027, "center_y": 301, "rect_length": 150, "rect_width": 200, "duration": 0}, # Ask restaurant name
            {"center_x": 300, "center_y": 400, "rect_length": 100, "rect_width": 100, "duration": 0}, # Confirm restaurant name and ask for google map
            {"center_x": 1027, "center_y": 301, "rect_length": 150, "rect_width": 200, "duration": 10000}, # Ask user to open play store
            {"center_x": 1246, "center_y": 129, "rect_length": 680, "rect_width": 90, "duration": 5000}, # Ask user to open search bar
            {"center_x": 1246, "center_y": 129, "rect_length": 680, "rect_width": 90, "duration": 0}, # Ask user to input Google Maps
            {"center_x": 1608, "center_y": 630, "rect_length": 120, "rect_width": 80, "duration": 8000}, # Ask user to open Google Maps
            {"center_x": 1033, "center_y": 142, "rect_length": 300, "rect_width": 80, "duration": 5000}, # Ask user to open search bar
            {"center_x": 1033, "center_y": 142, "rect_length": 300, "rect_width": 80, "duration": 0}, # Ask user to input restaurant name
            {"center_x": 1033, "center_y": 142, "rect_length": 300, "rect_width": 80, "duration": 0}, # Ask user to select restaurant
            {"center_x": 989, "center_y": 993, "rect_length": 210, "rect_width": 40, "duration": 10000}, # Tell user restaurant is open
            {"center_x": 989, "center_y": 993, "rect_length": 210, "rect_width": 40, "duration": 0}, # Tell user you are welcome
            # Add more overlays as needed.
        ]
        
        # Start playing the audio in a separate thread so it runs concurrently
        audio_thread = threading.Thread(target=playsound, args=(speech_file,))
        audio_thread.start()
        
        # Display the overlay concurrently with the audio playback
        show_overlay(**overlay_sequence[iteration])
        
        # Wait for the audio thread to finish, then delete the file
        audio_thread.join()
        os.unlink(speech_file)
        
        # Cycle the iteration variable so the next overlay is used next time.
        iteration = (iteration + 1) % len(overlay_sequence)
        
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
