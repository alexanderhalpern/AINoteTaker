import os
from openai import OpenAI
import json
import requests

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class Aligner:
    def __init__(self, transcription, slides):
        self.transcription = transcription
        self.slides = slides
        self.aligned = []

    # TRANSCRIPTION:
    #     {
    #     "text": " And he's not that weak. Yeah. The time is the bottom of it. You're smart. Yeah. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening. You're listening.",
    #     "start": 4109.411,
    #     "end": 4139.189
    # },
    # {
    #     "text": " We're behind it. Yeah.",
    #     "start": 4140.009,
    #     "end": 4158.558
    # },
    # {
    #     "text": " That was like, I don't know if you was saying that was not the next or you. It's very, very broad. Yeah. And it doesn't define such an action. But it's left to you. I think you mean the Madonna. Yeah. It's very much left to.",
    #     "start": 4159.343,
    #     "end": 4179.206
    # },
    # {
    #     "text": " The viewer to try to sort out what you wanted. Yeah. So there's postmodernism, South of it into the category. Oh, yeah. Well, which is. Yeah. OK. That fit into like that. Great. Modern arts. Oh, gosh. Yeah. Yeah. Oh, yes. Nice students. A direct like first impression has a like everything from that. Thank you. That's a follow. And what's called expressionism. And it's",
    #     "start": 4179.77,
    #     "end": 4208.012
    # }

    # SLIDES:
    # {'frame_number': 831, 'slide': 'Vincent van Gogh, Starry Night, 1889', 'timestamp': 33240.0}
    # {'frame_number': 934, 'slide': 'The Modernist Revolution: Post-Impressionism, Fauvism, Cubism', 'timestamp': 37360.0}
    # {'frame_number': 1176, 'slide': 'Vincent van Gogh, Starry Night, 1889', 'timestamp': 47040.0}
    # {'frame_number': 1217, 'slide': 'Paul Cézanne, Still Life with Fruit Dish, 1879-80', 'timestamp': 48680.0}

    # transcription has start and end times in seconds
    # slides have timestamps in milliseconds

    # for each slide, find the corresponding transcription and align them
    # join the aligned text and give to chatgpt api to generate bullet points

    def align(self):
        # First, sort slides based on timestamp
        sorted_slides = sorted(self.slides, key=lambda x: x['timestamp'])

        # Initialize an empty list for aligned data
        self.aligned = []

        for i in range(len(sorted_slides)):
            current_slide = sorted_slides[i]
            # Determine the start of the current slide
            current_start_time = current_slide['timestamp']

            # Determine the end time which is the start of the next slide or a large number if it's the last slide
            if i < len(sorted_slides) - 1:
                next_start_time = sorted_slides[i + 1]['timestamp']
            else:
                # No next slide, so use a very large number as the end time
                next_start_time = float('inf')

            # Collect transcription text that falls within the current slide's time range
            slide_text_segments = []
            for segment in self.transcription:
                if segment['start'] * 1000 <= next_start_time and segment['end'] * 1000 >= current_start_time:
                    slide_text_segments.append(segment['text'])

            # Join collected text segments for comprehensive slide information
            full_slide_text = ' '.join(slide_text_segments)

            # Call OpenAI API to generate bullet points for the collected text
            # result = client.chat.completions.create(
            #     model="gpt-3.5-turbo",
            #     messages=[
            #         {"role": "system", "content": "I will provide the notes for a slide. \
            #          Please provide bullet points for the slide. Make sure that absolutely no information is missing. \
            #          This should be just formatting. Do not lose or add any information."
            #          },
            #         {"role": "user", "content": full_slide_text}
            #     ]
            # )
            api_base = "https://api.endpoints.anyscale.com/v1"
            s = requests.Session()
            content = ""

            payload = {
                # "model": "C:\\Users\\Alexander\\.cache\\lm-studio\\models\\TheBloke\\SOLAR-10.7B-Instruct-v1.0-uncensored-GGUF\\solar-10.7b-instruct-v1.0-uncensored.Q6_K.gguf",
                "model": "meta-llama/Llama-3-70b-chat-hf",
                # "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "messages": [
                    {
                        "role": "system",
                        "content": '''
                        I will provide the title and spoken text for a slide shown in class.
                        Please provide bullet points for the slide. This should be just formatting. Do not lose or add any information.
                        You will be provided with the title of the slide and the content of the slide.
                        The title is a list of the possible titles for the slide.
                        Choose the correct title from the list.
                        Ignore references to Lawrence Goedde, he is the professor.
                        If there is no content or it is jibberish, please write 'No content' or 'Not Relevant' in the content field.


                        Use the following formatting for your response with no extra information. Do not do multiple groups of bullet points.
                        All bullet points should be in one group:

                        YOUR_TITLE_HERE\nBULLET_POINT_1\nBULLET_POINT_2\nBULLET_POINT_3\n...etc.

                        OR

                        NO CONTENT
                        '''
                    },
                    {
                        "role": "user",
                        "content": f'''
                        CHOOSE A TITLE:
                        {current_slide['slide']}
                        - --------------------------------------------

                        SLIDE CONTENT:
                        {full_slide_text}
                        '''
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 300,
                # "stop": [
                #     "[INST]"
                # ],
                "top_p": 0.95,
                "top_k": 40,
                "min_p": 0.05,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "logit_bias": {},
                "repeat_penalty": 1.1,
                # "seed": -1,
                "stream": True
            }
            # with s.post("http://halps.mynetgear.com:1234/v1/chat/completions", json=payload, stream=True) as response:
            with s.post(f"{api_base}/chat/completions", json=payload, stream=True, headers={"Authorization": f"Bearer {os.environ['ANY_SCALE_TOKEN'] }"}) as response:
                # Check if the request was successful
                # print(response.status_code)
                if response.status_code == 200:
                    # Iterate over the response
                    for line in response.iter_lines():
                        # Filter out keep-alive new lines
                        if line:
                            try:
                                # print(line)
                                # decoded_line = line.decode('utf-8')
                                json_line = json.loads(
                                    line.decode('utf-8')[6:])
                                # print(json_line)
                                # print(json_line["choices"][0]["content"])
                                # Depending on the structure of your response, you might want to access specific fields
                                if "content" in json_line["choices"][0]["delta"]:
                                    # print(repr(content))
                                    content += json_line["choices"][0]["delta"]["content"]
                            except Exception as e:
                                print(e)
            print(content)
            self.aligned.append({
                "slide": content.split("\n")[0],
                "transcription": content.split("\n")[1:]
            })
            self.save_aligned()
        return self.aligned

    def save_aligned(self):
        with open("aligned.json", "w") as f:
            json.dump(self.aligned, f, indent=2)
