import pytest
import pandas as pd
from src.rufp_stage1.nodes.ingestion import create_family_batches
from src.rufp_stage1.schemas import BaseItemRecord

def test_create_family_batches():
    items = [
        BaseItemRecord(
            base_item_id=str(i),
            category="test",
            family_id="fam1",
            family_name="Family 1",
            canonical_form="form",
            base_item_text=f"text {i}"
        ) for i in range(7)
    ]
    
    # 7 items should be split into 2 batches (5 + 2) if max_size=5
    batches = create_family_batches(items, max_size=5)
    assert len(batches) == 2
    assert len(batches[0].items) == 5
    assert len(batches[1].items) == 2
    assert batches[0].family_id == "fam1"

def test_small_family_warning(caplog):
    items = [
        BaseItemRecord(
            base_item_id="1",
            category="test",
            family_id="small_fam",
            family_name="Small Family",
            canonical_form="form",
            base_item_text="text"
        )
    ]
    batches = create_family_batches(items, min_size=3)
    assert len(batches) == 1
    assert "Family small_fam is small" in caplog.text
