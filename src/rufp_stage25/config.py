from pydantic_settings import BaseSettings
from typing import List

class Stage25Config(BaseSettings):
    app_name: str = "RU FP Bench Stage 2.5"
    output_dir: str = "artifacts/stage25"
    
    # Models
    repair_planner_model: str = "gpt-4o"
    repair_executor_model: str = "gpt-4o"
    
    max_repair_attempts: int = 1
    batch_size: int = 10

    class Config:
        env_prefix = "RUFP_S25_"

config = Stage25Config()
