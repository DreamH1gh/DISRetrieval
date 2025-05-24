from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
import json

class GPTQaModel:
    def __init__(self, model="gpt-4.1-mini"):
        self.model = model
        self.config

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question):
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    # "content": f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible. For yes or no question, only return 'yes' or 'no'. question: {question}" ### for qasper
                    "content": f"Given context: {context}\n.Answer the following multiplie-choice question:\n{question}" ### for quality
                }
            ],
            "temperature":0
        })
        pass

class DeepseekModel:
    def __init__(self, model="deepseek-reasoner"):
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context, question):
        client = OpenAI(
            api_key="sk-",
            base_url="https://api.deepseek.com",
        )

        # user_prompt = f"using the folloing information {context}. Answer the following question in less than 5-7 words, if possible. For yes or no question, only return 'yes' or 'no'. question: {question}" # for qasper
        user_prompt = f"Given context: {context}\n.Answer the following multiplie-choice question:\n{question}" # for quality

        messages = [{"role": "system", "content": 'You are a helpful assistant'},
                    {"role": "user", "content": user_prompt}]

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            timeout=30
        )

        return response.choices[0].message.content

class UnifiedQAModel:
    def __init__(self, device):
        model_path = "~/hfmodel/unifiedQA"
        self.device = device
        self.model = T5ForConditionalGeneration.from_pretrained(model_path).to(
            self.device
        )
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)

    def run_model(self, input_string, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(
            self.device
        )
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context, question):
        question = question.replace('(A)', ' \\n (A)')
        input_string = question + " \\n " + context
        output = self.run_model(input_string)
        return output[0]