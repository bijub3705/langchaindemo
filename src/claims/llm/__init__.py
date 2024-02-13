from src.claims.utils import get_constants
from langchain_community.llms import HuggingFaceTextGenInference
from src.claims.llm.custom_llm import CustomLLM

config = get_constants().MODELS.get("fb_aws_llm")
default_llm = HuggingFaceTextGenInference(
    inference_server_url=config.api_url,
    max_new_tokens= config.max_tokens,
    top_p= config.top_p,
    server_kwargs={
        "headers": {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_token}"
        }
    }
)

def get_llm(model_name):
    if model_name and get_constants().MODELS.get(model_name):
        return CustomLLM(get_constants().MODELS.get(model_name))
    else:
        raise ValueError(f"Invalid LLM name: {model_name}")

