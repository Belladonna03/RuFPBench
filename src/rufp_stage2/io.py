import json
from typing import List, Any
from pydantic import BaseModel

def load_jsonl(path: str, model: Any) -> List[Any]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(model.model_validate_json(line))
    return items

def save_jsonl(path: str, items: List[BaseModel]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if isinstance(data, BaseModel):
            f.write(data.model_dump_json(indent=2))
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
