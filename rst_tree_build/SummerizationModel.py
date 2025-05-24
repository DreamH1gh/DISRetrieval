import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI

class LLamaAPISummerizationModel():
    def __init__(self):
        super().__init__()

        self.model_name_or_path = '~/hfmodel/Meta-Llama-3___1-8B-Instruct'

    def summarize(self, context):
        try:
            messages = [
                {
                "role": "system",
                "content": "You are a Summarizing Text Portal"
                },
                {
                    "role": "user", 
                    "content": f'write a summary of the given sentences, keeps as more key information as possible. Only give the summary without other text. Makse sure that the summary no more than 200 words.\ngiven text: \n{context}'
                }
            ]
            client = OpenAI(api_key="0",base_url=f"http://0.0.0.0:7000/v1")
            result = client.chat.completions.create(messages=messages, model=self.model_name_or_path)
            return result.choices[0].message.content
        
        except Exception as e:
            logging.error("Summarization failed: %s", str(e))
            return "[Error] Failed to generate summary."
