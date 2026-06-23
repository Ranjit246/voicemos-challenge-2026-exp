import librosa
from transformers import pipeline

MODEL_ID = "HamzaSidhu786/speech-accent-detection"
TEMP_AUDIO = "/Users/ranjitpatro/Home/Research/VoiceMOS/dataset/vmc2026_track3_train_phase_distro_v3_syn/wav/vmc2026-track3-sys001-utt002.wav"
clf = pipeline("audio-classification", model=MODEL_ID)

def predict_accent(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    clip = audio[:sr * 10]
    results = clf(clip)
    top = max(results, key=lambda x: x["score"])
    return f"Accent: {top['label']} | Confidence: {top['score']*100:.2f}%"

if __name__ == "__main__":
    result = predict_accent(TEMP_AUDIO)
    print(result)