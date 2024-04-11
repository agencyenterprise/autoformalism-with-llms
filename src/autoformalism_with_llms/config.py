from dotenv import find_dotenv
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    OPENAI_API_KEY: str
    model_config = {"env_file": find_dotenv(usecwd=True)}


config = Config()
