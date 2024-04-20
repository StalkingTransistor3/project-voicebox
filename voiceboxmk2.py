import sounddevice as sd
import numpy as np
import whisper
import torch

def transcribe_audio(audio_data, model_size='medium'):
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the Whisper model
    model = whisper.load_model(model_size, device=device)

    # Process the audio data
    result = model.transcribe(audio_data)
    return result

if __name__ == "__main__":
    # Record audio from the microphone
    duration = 10  # seconds
    fs = 44100  # Sample rate

    print("Recording started...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    print("Recording ended.")

    # Convert the audio data to a suitable format for the Whisper model
    audio_data = np.mean(audio_data, axis=1)  # Convert stereo to mono
    audio_data = (audio_data * 32768).astype(np.int16)  # Convert float to int16

    # Convert the audio data back to float32 format
    audio_data = audio_data.astype(np.float32) / 32768.0

    # Transcribe the audio data
    transcription_result = transcribe_audio(audio_data)
    print("Transcription:", transcription_result['text'])