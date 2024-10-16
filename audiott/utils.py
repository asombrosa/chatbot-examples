import os
import whisper
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
  api_version="2024-06-01"
)


def save_file(text, file_name):
    """Saves content on a file"""
    # Ensure the `files` directory exists
    if not os.path.exists("transcriptions"):
        os.makedirs("transcriptions")

    # Open the file in write mode and write the content
    with open(file_name, "w") as file:
        file.write(text)

    print(f"Content saved to {file_name}")


def speech_to_text(audio_path="media/audio.mp3"):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError("File not found")

        # Initialize the Whisper ASR model
        model = whisper.load_model("base")

        # Your code to transcribe the audio
        result = model.transcribe(audio_path)

        # Extract the transcript text from the result
        return result["text"]

    except Exception as e:
        print(f"An error occurred during transcription: {e}")


def speech_to_translation(audio_path="media/audio.mp3"):
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError("File not found")

        # Initialize the Whisper ASR model
        model = whisper.load_model("base")

        # Your code to transcribe the audio
        result = model.transcribe(audio_path, language="en")

        # Extract the transcript text from the result
        return result["text"]
    except Exception as e:
        print(f"An error occurred during transcription: {e}")