from openai import OpenAI
import os
import base64
import cv2
import numpy as np
import json
import pytesseract
from PIL import Image
import torch
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

processor = LlavaNextProcessor.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to("cuda:0")
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class SlideAnalyzer:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.slides = []

    def analyze_video(self, video_path, output_path):
        video = cv2.VideoCapture(video_path)
        frame_number = 0
        last_frame = None
        results = []

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Process frame only if it's the 60th, 120th, 180th, etc.
            if frame_number % 60 == 0:
                if last_frame is not None:
                    if self.get_diff(frame, last_frame) > 0.3:
                        # Save slides to output
                        cv2.imwrite(
                            f"{output_path}/frame_{frame_number}.jpg", frame)

                        result = self.analyze_frame_llava(frame)
                        results.append({
                            "frame_number": frame_number,
                            "slide": result,
                            "timestamp": video.get(cv2.CAP_PROP_POS_MSEC)
                        })
                        print({
                            "frame_number": frame_number,
                            "slide": result,
                            "timestamp": video.get(cv2.CAP_PROP_POS_MSEC)
                        })
                last_frame = frame
            frame_number += 1

        self.slides = results
        video.release()
        return self.slides

    # def analyze_video_upto(self, video_path, end_frame_number):
    #     video = cv2.VideoCapture(video_path)
    #     frame_number = 0
    #     last_frame = None
    #     results = []
    #     while (video.isOpened()):
    #         ret, frame = video.read()
    #         if ret:
    #             if last_frame is not None:
    #                 if self.get_diff(frame, last_frame) > 0.3:
    #                     result = self.analyze_frame(frame)
    #                     results.append({
    #                         "frame_number": frame_number,
    #                         "slide": result,
    #                         "timestamp": video.get(cv2.CAP_PROP_POS_MSEC)
    #                     })
    #                     print({
    #                         "frame_number": frame_number,
    #                         "slide": result,
    #                         "timestamp": video.get(cv2.CAP_PROP_POS_MSEC)
    #                     })
    #             last_frame = frame
    #         frame_number += 1
    #         if frame_number >= end_frame_number:
    #             break
    #     self.slides = results
    #     return self.slides

    def analyze_frame(self, frame):
        response = self.client.chat.completions.with_raw_response.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "system",
                    "content": "This is a slide from a lecture. Give me a one sentence title for this slide. \
                        Simply state the exact piece of artwork that is featured. \ Response with nothing else but the title."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(cv2.imencode('.jpg', frame)[1]).decode("utf-8")
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        chat_completion = response.parse()
        return chat_completion.choices[0].message.content

    def analyze_frame_tesseract(self, frame):
        cv2.imwrite("temp.jpg", frame)
        text = pytesseract.image_to_string(Image.open("temp.jpg"))
        os.remove("temp.jpg")
        return text

    def analyze_frame_llava(self, frame):
        prompt = "[INST] <image>\nRespond with the text in the image.</image> [/INST]"
        inputs = processor(prompt, frame, return_tensors="pt").to("cuda:0")
        output = model.generate(**inputs, max_new_tokens=100)
        return (processor.decode(output[0], skip_special_tokens=True))

    def get_diff(self, frame, last_frame):
        diff = cv2.absdiff(last_frame, frame)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
        return np.sum(thresh) / \
            (thresh.shape[0] * thresh.shape[1] * 255)

    def save_slides(self):
        with open("slides.json", "w") as f:
            json.dump(self.slides, f, indent=2)
