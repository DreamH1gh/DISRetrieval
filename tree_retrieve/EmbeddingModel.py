from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential
import torch

class OpenAIEmbeddingModel():
    def __init__(self, model="text-embedding-3-large"):
        self.client = None
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, texts):
        pass


class SBertEmbeddingModel():
    def __init__(self, model_name="~/hfmodel/sbert", device=torch.device('cpu')):
        self.model = SentenceTransformer(model_name)
        self.device = device
        self.model.to(self.device)

    def create_embedding(self, texts):
        return torch.from_numpy(self.model.encode(texts, show_progress_bar=False))
