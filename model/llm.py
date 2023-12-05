from langchain import HuggingFaceHub
import os
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv('HF_TOKEN')



STOP_SEQUENCES = ["\nUser:", "<|endoftext|>", " User:", "###"]

repo_id = "meta-llama/Llama-2-7b-chat-hf"
llm = HuggingFaceHub(
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    repo_id=repo_id,
    model_kwargs={
        "tempearture":0.4,
        "max_new_tokens":150,
        "repetition_penalty":1.2}
)