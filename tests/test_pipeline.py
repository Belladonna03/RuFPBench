from unittest.mock import MagicMock

from src.config import PipelineConfig
from src.pipeline import PseudoGraphPipeline


def test_process_one_propagates_extra_meta():
    cfg = PipelineConfig()
    pipeline = PseudoGraphPipeline(cfg)
    pipeline.gliner.extract = MagicMock(return_value=[])
    pipeline.keyphrase_extractor.extract = MagicMock(return_value=[])

    g = pipeline.process_one(
        "doc_1",
        "привет",
        extra_meta={"batch": "b1", "row_index": 7},
    )
    assert g.metadata.extra == {"batch": "b1", "row_index": 7}
