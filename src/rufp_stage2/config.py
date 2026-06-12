from pydantic_settings import BaseSettings
from typing import List, Dict

class Stage2Config(BaseSettings):
    app_name: str = "RU FP Bench Stage 2"
    output_dir: str = "artifacts/stage2"
    
    # Gating models
    safety_judge_model: str = "gpt-4o"
    naturalness_judge_model: str = "gpt-4o"
    
    # Target models for probing
    target_models: List[str] = ["gpt-3.5-turbo", "claude-3-haiku"]
    
    batch_size: int = 20
    max_retries: int = 3

    class Config:
        env_prefix = "RUFP_S2_"

config = Stage2Config()
