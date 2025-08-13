import requests
from rag import settings


def load_prompts(tags: list[str] = [], limit: int = 1) -> list[str]:
    if not settings.PROMPT_BACKEND_URI:
        return []
    uri = f"{settings.PROMPT_BACKEND_URI}/v1/get_prompts"
    payload = { "limit": limit, "tags": tags }
    headers = { "Content-Type": "application/json" }
    return [ x["prompt_value"] for x in requests.post(uri, json=payload, headers=headers) ]


def get_prompt(tags: list[str], default: str) -> str:
    prompts = load_prompts(tags, 1)
    if len(prompts) == 0:
        return default
    else:
        return prompts[0]