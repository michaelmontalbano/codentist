import os
import whisper
from pyannote.audio import Pipeline
from pydub import AudioSegment
from huggingface_hub import login

# === 1. Settings ===
HUGGINGFACE_TOKEN = "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"
mp3_path = "C:\Work\Codey_LangChain_OpenAI\SPT\examples\ex3.mp3"
base_name = os.path.splitext(mp3_path)[0]
wav_path = f"{base_name}.wav"
output_txt = f"{base_name}_with_speakers.txt"

# === 2. Hugging Face Authorization ===
login(HUGGINGFACE_TOKEN)

# === 3. Convert mp3 to wav ===
audio = AudioSegment.from_mp3(mp3_path)
audio.export(wav_path, format="wav")

# === 4. Load Models ===
print("🔊 Loading Whisper and Pyannote...")
whisper_model = whisper.load_model("small")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                    use_auth_token=HUGGINGFACE_TOKEN)

# === 5. Diarization ===
print("👥 Identifying speakers...")
diarization = pipeline(wav_path, num_speakers=2)

# === 6. Speech Recognition ===
print("🧠 Recognizing text...")
result = whisper_model.transcribe(mp3_path, language="en", verbose=False)

# === 7. Write Output with Deduplication ===
print("📄 Generating final text...")

used_segments = set()

with open(output_txt, "w", encoding="utf-8") as f:
    for turn in diarization.itertracks(yield_label=True):
        start_time = turn[0].start
        end_time = turn[0].end
        speaker = turn[2]

        # Find all Whisper segments that overlap with this interval
        segments = [seg for seg in result["segments"]
                    if not (seg["end"] < start_time or seg["start"] > end_time)]

        for seg in segments:
            seg_id = (round(seg["start"], 2), round(seg["end"], 2))
            if seg_id in used_segments:
                continue  # Oops, already written
            used_segments.add(seg_id)

            minutes = int(seg["start"]) // 60
            seconds = int(seg["start"]) % 60
            timestamp = f"[{minutes:02}:{seconds:02}]"
            text = seg["text"].strip()

            if len(text) > 1:
                f.write(f"{timestamp} {speaker}: {text}\n")

print(f"\n✅ Done! File saved: {output_txt}")