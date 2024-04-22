import whisperx
import json


class Scribe:
    def __init__(self, audio_path):
        print("HERE")
        self.log = []
        self.model = whisperx.load_model(
            "small",
            device="cuda",
            compute_type="float16",
            language="en"
        )
        # self.align_model, self.metadata = whisperx.load_align_model(
        #     "en",
        #     device="cuda",
        # )
        self.audio = whisperx.load_audio(audio_path)
        self.transcription = None

    def transcribe(self):
        self.transcription = self.model.transcribe(
            self.audio,
            batch_size=16
        )
        # self.transcription = whisperx.align(
        #     self.transcription["segments"],
        #     self.align_model,
        #     self.metadata,
        #     self.audio,
        #     "cuda",
        #     return_char_alignments=True
        # )
        return self.transcription

    def save_transcription(self):
        with open("transcription.json", "w") as f:
            json.dump(self.transcription["segments"], f, indent=2)

    def erase(self):
        self.log = []

    def get_log(self):
        return self.log
