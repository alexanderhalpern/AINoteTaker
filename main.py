import numpy as np
import cv2
import argparse
from dotenv import load_dotenv
from scribe import Scribe
from slide_analyzer import SlideAnalyzer
from aligner import Aligner
import os
import base64
import json

load_dotenv()

parser = argparse.ArgumentParser(
    prog="AINoteTaker", description="AI Note Taker")
parser.add_argument("--video", help="Path to video file")
parser.add_argument("--output", help="Path to output file")

args = parser.parse_args()

video = cv2.VideoCapture(args.video)

frame_number = 0
last_frame = None


scribe = Scribe(args.video)
slide_analyzer = SlideAnalyzer()


# Create Transcription
transcription = scribe.transcribe()
# Save Transcription
scribe.save_transcription()

# # Analyze Video
# slides = slide_analyzer.analyze_video(args.video, args.output)
# # Save Slides
# slide_analyzer.save_slides()

# load transcription from transcription.json
with open("transcription.json", "r") as f:
    transcription = json.load(f)

# load slides from slides.json
with open("slides.json", "r") as f:
    slides = json.load(f)

print(transcription)

# Align Transcription and Slides
aligner = Aligner(transcription, slides)
aligned_results = aligner.align()

# Save Aligned Results
aligner.save_aligned()