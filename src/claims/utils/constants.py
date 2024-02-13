import yaml
import re
import os

from attrs import define, field
import cattrs
from dotenv import load_dotenv

load_dotenv()

@define
class Configs:
    models: str
    embedders: str

@define
class ModelObject:
    name: str
    api_url: str
    api_token: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    do_sample: bool
    return_full_text: bool
    frequency_penalty: int
    presence_penalty: int

@define
class EmbedderObject:
    name: str
    api_url: str
    api_token: str

@define
class EmbeddersObject:
    embedders: list[EmbedderObject]

@define
class ModelsObject:
    models: list[ModelObject]

class Constants:
    MODELS = {}
    EMBEDDERS = {}
    def __init__(self,config_path) -> None:
        self.config_path = config_path
        self.load_constants()

    def load_constants(self):
        with open(self.config_path + "\\config.yml", 'r') as configfile:
            self.data = yaml.safe_load(configfile)
        self.config = cattrs.structure(self.data, Configs)
        self.load_models(self.config.models)
        self.load_embedders(self.config.embedders)

    def load_models(self, model_config_file_name):
        with open(self.config_path + "\\models\\"+model_config_file_name+".yml", 'r') as modelsfile:
            self.models_data = yaml.safe_load(modelsfile)
        for(model_data) in self.models_data["models"]:
            self.update_env_variables(model_data)
        self.models = cattrs.structure(self.models_data, ModelsObject)
        for model_object in self.models.models:
            self.MODELS.update({model_object.name: model_object})

    def load_embedders(self, embedder_config_file_name):
        with open(self.config_path + "\\embedders\\"+embedder_config_file_name+".yml", 'r') as embeddersfile:
            self.embedders_data = yaml.safe_load(embeddersfile)
        for(embedder_data) in self.embedders_data["embedders"]:
            self.update_env_variables(embedder_data)
        self.embedders = cattrs.structure(self.embedders_data, EmbeddersObject)
        for embedder_object in self.embedders.embedders:
            self.EMBEDDERS.update({embedder_object.name: embedder_object})
       
    def update_env_variables(self, dict):
        env_variable_pattern = re.compile(r'\${(.*?)}')
        for key, value in dict.items():
            if isinstance(value, str) and env_variable_pattern.search(value):
                env_variable = env_variable_pattern.search(value).group(1)
                dict[key] = os.getenv(env_variable)


if __name__ == "__main__":
    constant = Constants("configs")
    #print(constant.MODELS)
    print(constant.EMBEDDERS.get("fb_aws_embedder").api_token)