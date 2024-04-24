import numpy as np
import cv2
import argparse
from dotenv import load_dotenv
from scribe import Scribe
from slide_analyzer import SlideAnalyzer
from aligner import Aligner
import os
import base64
import torch
import json

load_dotenv()

parser = argparse.ArgumentParser(
    prog="AINoteTaker", description="AI Note Taker")
parser.add_argument("--file-name", help="Name of the file")
parser.add_argument("--video", help="Path to video file")
parser.add_argument("--frames", help="Path to frames folder")
parser.add_argument("--notes", help="Path to notes folder")

args = parser.parse_args()
args.frames = os.path.join(args.frames, args.file_name)
args.notes = os.path.join(args.notes, args.file_name)


video = cv2.VideoCapture(args.video)

frame_number = 0
last_frame = None

torch.cuda.empty_cache()
scribe = Scribe(args.video)

# Create Transcription
transcription = scribe.transcribe()
# Save Transcription
scribe.save_transcription()

torch.cuda.empty_cache()
slide_analyzer = SlideAnalyzer()
# # Analyze Video
slides = slide_analyzer.analyze_video(args.video, args.frames)
# # Save Slides
slide_analyzer.save_slides()

# load transcription from transcription.json
with open("transcription.json", "r") as f:
    transcription = json.load(f)

# load slides from slides.json
with open("slides.json", "r") as f:
    slides = json.load(f)

torch.cuda.empty_cache()
# Align Transcription and Slides
aligner = Aligner(transcription, slides, args.frames, args.notes)
aligned_results = aligner.align()

# Save Aligned Results
aligner.save_aligned()
