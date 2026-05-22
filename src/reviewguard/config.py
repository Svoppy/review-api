from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REVIEWGUARD_", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[2]
    model_name: str = "FacebookAI/xlm-roberta-base"
    checkpoint_dir: Path = Path("models/latest")
    max_length: int = 256


settings = Settings()

