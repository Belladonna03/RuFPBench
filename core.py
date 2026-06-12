from typing import List, Dict, Any
from models import BaseItem, FamilyConfig, PromptCandidate, PromptLength, GenerationRoute, Stage1Result
from llm import LLMClient
from templates import get_template, NATURALIZER_PROMPT, BORDERLINE_REFINER_PROMPT
from jinja2 import Template
import uuid

class FamilyRouter:
    def __init__(self, configs: List[FamilyConfig]):
        self.route_map = {c.family: c.route for c in configs}

    def get_route(self, family: str) -> GenerationRoute:
        return self.route_map.get(family, GenerationRoute.DIRECT_EXPANSION)

class CandidateGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    async def generate_raw(self, item: BaseItem, route: GenerationRoute, length: PromptLength) -> str:
        template = get_template(route.value)
        prompt = template.render(text=item.text, length=length.value)
        return await self.llm_client.generate(prompt)

    async def naturalize(self, text: str) -> str:
        template = Template(NATURALIZER_PROMPT)
        prompt = template.render(text=text)
        return await self.llm_client.generate(prompt)

    async def refine_borderline(self, text: str) -> str:
        template = Template(BORDERLINE_REFINER_PROMPT)
        prompt = template.render(text=text)
        return await self.llm_client.generate(prompt)

    async def generate_candidates(self, item: BaseItem, route: GenerationRoute) -> List[PromptCandidate]:
        candidates = []
        lengths = [PromptLength.SHORT, PromptLength.MEDIUM, PromptLength.LONG]
        
        for length in lengths:
            # 1. Generate Raw
            raw_text = await self.generate_raw(item, route, length)
            
            # 2. Naturalize
            natural_text = await self.naturalize(raw_text)
            
            # 3. Refine Borderline
            final_text = await self.refine_borderline(natural_text)
            
            candidates.append(PromptCandidate(
                id=str(uuid.uuid4()),
                base_item_id=item.id,
                text=final_text,
                length=length,
                route=route,
                metadata={
                    "raw_text": raw_text,
                    "natural_text": natural_text
                }
            ))
        return candidates

class Stage1Pipeline:
    def __init__(self, router: FamilyRouter, generator: CandidateGenerator):
        self.router = router
        self.generator = generator

    async def run(self, items: List[BaseItem]) -> Stage1Result:
        all_candidates = []
        for item in items:
            route = self.router.get_route(item.family)
            candidates = await self.generator.generate_candidates(item, route)
            all_candidates.extend(candidates)
        
        return Stage1Result(candidates=all_candidates)
