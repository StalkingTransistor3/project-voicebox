import whisper
import sys
import torch

def transcribe_audio(audio_path, model_size='medium'):
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the Whisper model
    model = whisper.load_model(model_size, device=device)

    # Process the audio file
    result = model.transcribe(audio_path)
    return result

if __name__ == "__main__":
    audio_file_path = 'test.mp3'
    transcription_result = transcribe_audio(audio_file_path)
    print("Transcription:", transcription_result['text'])