import uuid
import logging
from typing import List, Dict
from ..schemas import BaseItemRecord, FamilyBatch

logger = logging.getLogger(__name__)

def create_family_batches(items: List[BaseItemRecord], min_size: int = 3, max_size: int = 5) -> List[FamilyBatch]:
    # Group by family_id
    families: Dict[str, List[BaseItemRecord]] = {}
    family_names: Dict[str, str] = {}
    
    for item in items:
        if item.family_id not in families:
            families[item.family_id] = []
            family_names[item.family_id] = item.family_name
        families[item.family_id].append(item)
    
    batches = []
    for family_id, family_items in families.items():
        if len(family_items) < min_size:
            logger.warning(f"Family {family_id} is small: {len(family_items)} items. Creating single batch.")
        
        # Split into chunks of max_size
        for i in range(0, len(family_items), max_size):
            chunk = family_items[i : i + max_size]
            batches.append(FamilyBatch(
                batch_id=str(uuid.uuid4()),
                family_id=family_id,
                family_name=family_names[family_id],
                items=chunk
            ))
            
    return batches
