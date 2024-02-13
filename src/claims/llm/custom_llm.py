from typing import Any, List, Mapping, Optional
import requests
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM

from src.claims.utils import get_constants

class CustomLLM(LLM):
    model_config: Any
    api_url: str
    api_token: str

    def __init__(self, config):
        super().__init__(model_config=config, api_url=config.api_url, api_token=config.api_token)
        self.model_config = config
        self.api_url = self.model_config.api_url
        self.api_token = self.model_config.api_token

    @property
    def _llm_type(self) -> str:
        return  self.model_config.name

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        payload = {
            "inputs": prompt,
            "parameters":{
                "max_new_tokens": self.model_config.max_tokens,
                "temperature": self.model_config.temperature,
                "top_p": self.model_config.top_p,
                "do_sample": self.model_config.do_sample,
                "return_full_text": self.model_config.return_full_text
            }
        }
        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = requests.post(self.api_url, headers=headers, json=payload)
        return response.json()[0]['generated_text']

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": ""}
    