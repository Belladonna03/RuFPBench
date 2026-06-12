from pydantic_settings import BaseSettings
from typing import Optional

class Stage1Config(BaseSettings):
    app_name: str = "RU FP Bench Stage 1"
    output_dir: str = "artifacts/stage1"
    llm_model: str = "mock"
    max_retries: int = 3
    batch_size: int = 10

    class Config:
        env_prefix = "RUFP_"

config = Stage1Config()
