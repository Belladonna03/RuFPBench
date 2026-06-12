from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class SafetyPair:
    id: str
    topic: str
    safe_neighbor: str
    safe_request: str
    unsafe_contrast: str
    minimal_difference: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class TargetRun:
    id: str
    topic: str
    safe_neighbor: str
    safe_request: str
    safe_response: str
    model: str
    response_type_hint: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


@dataclass
class JudgedRun:
    id: str
    topic: str
    safe_neighbor: str
    safe_request: str
    safe_response: str
    model: str
    judge: str

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)
