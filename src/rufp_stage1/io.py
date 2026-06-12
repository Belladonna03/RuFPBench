import json
import pandas as pd
from typing import List, Any, Optional
from pydantic import BaseModel
from .schemas import BaseItemRecord, FamilyBatch

def load_raw_data(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        xl = pd.ExcelFile(path)
        sheet_names = [s for s in xl.sheet_names if s.lower() in ["base_items", "base_items"]]
        if not sheet_names:
            raise ValueError(f"Sheet 'base_items' not found in {path}")
        return pd.read_excel(path, sheet_name=sheet_names[0])
    else:
        raise ValueError(f"Unsupported file format: {path}")

def normalize_base_items(df: pd.DataFrame) -> List[BaseItemRecord]:
    # Mapping of potential column names to canonical names
    mapping = {
        "id": "base_item_id",
        "text": "base_item_text",
        "family": "family_id",
        "family_name": "family_name",
        "canonical": "canonical_form",
        "source": "source_url",
    }
    df = df.rename(columns=mapping)
    
    # Ensure all required columns exist
    required = ["base_item_id", "category", "family_id", "family_name", "canonical_form", "base_item_text"]
    for col in required:
        if col not in df.columns:
            df[col] = ""
            
    records = []
    for _, row in df.iterrows():
        # Handle NaN values for Pydantic and force string for IDs
        row_dict = row.to_dict()
        clean_dict = {}
        for k, v in row_dict.items():
            if pd.isna(v):
                clean_dict[k] = None
            elif k in ["base_item_id", "family_id"]:
                clean_dict[k] = str(v)
            else:
                clean_dict[k] = v
        records.append(BaseItemRecord(**clean_dict))
    return records

def save_jsonl(path: str, items: List[BaseModel]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        if hasattr(data, "model_dump_json"):
            f.write(data.model_dump_json(indent=2))
        else:
            json.dump(data, f, indent=2, ensure_ascii=False)
