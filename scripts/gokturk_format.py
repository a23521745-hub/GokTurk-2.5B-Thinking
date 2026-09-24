"""GökTürk-2.5B-Thinking — ortak format tanımları.

Her asistan yanıtı TAM OLARAK şu 5 aşamayı, bu sırayla içerir:

    <think> ... </think>                 # serbest iç akıl yürütme (R1 tarzı)
    <plan> ... </plan>                   # numaralı çözüm planı
    <search_query> ... </search_query>   # web sorgusu veya "YOK"
    <verify> ... </verify>               # öz-denetim / sağlama
    <output> ... </output>               # kullanıcıya gösterilecek nihai yanıt
"""
from __future__ import annotations

import re

TAGS = ["think", "plan", "search_query", "verify", "output"]
NO_SEARCH = "YOK"

SYSTEM_PROMPT = (
    "Sen GökTürk-2.5B-Thinking adlı Türkçe bir yapay zekâ asistanısın. "
    "Her yanıtında sırasıyla <think>, <plan>, <search_query>, <verify> ve <output> "
    "etiketlerini kullan. Güncel veya doğrulanması gereken bilgi gerekiyorsa "
    "<search_query> içine kısa bir web sorgusu yaz; gerekmiyorsa YOK yaz. "
    "Nihai yanıtı yalnızca <output> içinde, açık ve doğru bir Türkçeyle ver."
)

_PATTERN = re.compile(
    r"^\s*" + r"\s*".join(rf"<{t}>(?P<{t}>.*?)</{t}>" for t in TAGS) + r"\s*$",
    re.DOTALL,
)


def build_response(think: str, plan: list[str] | str, verify: str, output: str,
                   search_query: str | None = None) -> str:
    if isinstance(plan, list):
        plan = "\n".join(f"{i}. {p}" for i, p in enumerate(plan, 1))
    parts = {
        "think": think.strip(),
        "plan": plan.strip(),
        "search_query": (search_query or NO_SEARCH).strip(),
        "verify": verify.strip(),
        "output": output.strip(),
    }
    return "\n".join(f"<{t}>\n{parts[t]}\n</{t}>" for t in TAGS)


def parse_response(text: str) -> dict[str, str] | None:
    """Yanıtı parçalarına ayırır; format bozuksa None döner."""
    m = _PATTERN.match(text)
    if not m:
        return None
    parsed = {t: m.group(t).strip() for t in TAGS}
    return parsed if all(parsed.values()) else None


def is_valid(text: str) -> bool:
    return parse_response(text) is not None


def to_record(user: str, response: str, category: str, source: str,
              difficulty: str = "orta") -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": response},
        ],
        "meta": {
            "category": category,
            "source": source,
            "difficulty": difficulty,
            "needs_search": parse_response(response)["search_query"] != NO_SEARCH,
        },
    }
