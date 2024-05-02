from dotenv import find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    OPENAI_API_KEY: str
    model_config = SettingsConfigDict(env_file=find_dotenv(usecwd=True), extra="allow")


config = Config()
