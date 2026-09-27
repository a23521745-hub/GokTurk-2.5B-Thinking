"""GökTürk-2.5B-Thinking — 6 Aşamalı Hibrit CoT + Tool Calling çekirdeği.

Tüm bileşenler (veri hazırlama, eğitim, RAG çalıştırıcısı, testler) bu modülü paylaşır;
böylece eğitimde görülen biçim ile çıkarımda üretilen biçim BİREBİR aynıdır.

Asistan yanıtının biçimi
------------------------
    <thought>
    [STEP 1: Problem Sentezleme & Ayrıştırma]
    ...
    [STEP 2: İçsel Akıl Yürütme]
    ... Dış kaynak: GEREKLİ | GEREKSİZ
    [STEP 3: Web Araştırması & Kaynak Toplama]
    ...
    <tool_call>
    {"name": "tavily_search", "arguments": {"query": "...", "topic": "general"}}
    </tool_call>                         ← yalnızca arama gerekiyorsa; tur burada biter
    ── (araç turu) <tool_response> ... </tool_response> ──
    [STEP 4: Kaynak Doğrulama & Yerelde İşleme]
    ...
    [STEP 5: Yanıt Planlama & Sunum Hazırlığı]
    ...
    [STEP 6: Öz-Denetim & Eksiklik Giderme]
    ...
    </thought>

    Nihai yanıt (markdown, kaynak atıfları [1], [2] ...)

Araç çağrısı Qwen2.5'in yerel (Hermes tarzı) `<tool_call>` biçimini kullanır; taban model
bu biçimi ön eğitimden tanıdığı için az veriyle güvenilir öğrenilir. Araç sonucu ChatML'de
Qwen2.5'in resmî şablonundaki gibi `user` rolü içinde `<tool_response>` olarak işlenir.
"""
from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------
MODEL_NAME = "GökTürk-2.5B-Thinking"

STEPS: list[tuple[int, str]] = [
    (1, "Problem Sentezleme & Ayrıştırma"),
    (2, "İçsel Akıl Yürütme"),
    (3, "Web Araştırması & Kaynak Toplama"),
    (4, "Kaynak Doğrulama & Yerelde İşleme"),
    (5, "Yanıt Planlama & Sunum Hazırlığı"),
    (6, "Öz-Denetim & Eksiklik Giderme"),
]
STEP_HEADERS = {i: f"[STEP {i}: {name}]" for i, name in STEPS}

THOUGHT_OPEN, THOUGHT_CLOSE = "<thought>", "</thought>"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
NEED_SEARCH, NO_SEARCH = "Dış kaynak: GEREKLİ", "Dış kaynak: GEREKSİZ"

TAVILY_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": "Güncel/canlı web bilgisi için Tavily araması yapar.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Kısa, anahtar kelime odaklı sorgu"},
                "topic": {"type": "string", "enum": ["general", "news"]},
                "time_range": {"type": "string", "enum": ["day", "week", "month", "year"]},
            },
            "required": ["query"],
        },
    },
}

# Kısa tutuldu: her token 2.5 GB RAM bütçesinde KV-cache demektir.
SYSTEM_PROMPT = f"""Sen {MODEL_NAME}, Türkçe konuşan, dürüst ve titiz bir yapay zekâ asistanısın.
Her yanıttan önce <thought> içinde sırasıyla 6 adımı uygula:
[STEP 1] isteği, kısıtları ve beklenen formatı ayrıştır;
[STEP 2] kendi bilgini tara ve "{NEED_SEARCH}" ya da "{NO_SEARCH}" diye karar ver;
[STEP 3] gerekiyorsa en iyi arama sorgusunu kur ve tavily_search aracını çağır;
[STEP 4] araç sonuçlarını güvenilirlik ve tutarlılık açısından süz, çelişkileri ele;
[STEP 5] yanıtı yapılandır;
[STEP 6] mantık, dil ve eksiklik denetimi yap.
</thought> sonrasında nihai yanıtı açık Türkçe ile ver; web kaynaklarını [1], [2] diye an.

# Araçlar
<tools>
{json.dumps(TAVILY_TOOL, ensure_ascii=False)}
</tools>
Araç çağırmak için yalnızca şunu yaz:
<tool_call>
{{"name": "tavily_search", "arguments": {{"query": "..."}}}}
</tool_call>"""

# ---------------------------------------------------------------------------
# Kayıt oluşturma
# ---------------------------------------------------------------------------

def _step_block(i: int, text: str) -> str:
    return f"{STEP_HEADERS[i]}\n{text.strip()}\n"


def build_messages(user: str, steps: dict[int, str], answer: str,
                   tool_call: dict | None = None, tool_response: str | None = None,
                   system: str = SYSTEM_PROMPT) -> list[dict]:
    """6 adım + yanıt → ChatML mesaj listesi. `tool_call` varsa üç asistan/araç turu üretilir."""
    steps = {int(k): v for k, v in steps.items()}
    missing = [i for i, _ in STEPS if not str(steps.get(i, "")).strip()]
    if missing:
        raise ValueError(f"Eksik adımlar: {missing}")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user.strip()}]
    head = THOUGHT_OPEN + "\n" + "".join(_step_block(i, steps[i]) for i in (1, 2, 3))
    tail = "".join(_step_block(i, steps[i]) for i in (4, 5, 6)) + THOUGHT_CLOSE + "\n\n" + answer.strip()
    if tool_call:
        call = {"name": tool_call.get("name", "tavily_search"),
                "arguments": tool_call.get("arguments", tool_call)}
        msgs += [
            {"role": "assistant", "content": head, "tool_call": call},
            {"role": "tool", "content": (tool_response or "").strip() or '{"results": []}'},
            {"role": "assistant", "content": tail},
        ]
    else:
        msgs.append({"role": "assistant", "content": head + tail})
    return msgs


def format_tool_call(call: dict) -> str:
    return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"


# ---------------------------------------------------------------------------
# ChatML render (eğitim + çıkarım ortak)
# ---------------------------------------------------------------------------

def render_chatml(messages: list[dict], add_generation_prompt: bool = False) -> str:
    out = []
    for m in messages:
        role, content = m["role"], m.get("content", "")
        if role == "assistant" and m.get("tool_call"):
            content = (content.rstrip("\n") + "\n" if content else "") + format_tool_call(m["tool_call"])
        if role == "tool":
            out.append(f"{IM_START}user\n<tool_response>\n{content}\n</tool_response>{IM_END}\n")
        else:
            out.append(f"{IM_START}{role}\n{content}{IM_END}\n")
    if add_generation_prompt:
        out.append(f"{IM_START}assistant\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Ayrıştırma & doğrulama
# ---------------------------------------------------------------------------
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_STEP_RE = re.compile(r"\[STEP (\d): [^\]]+\]")


def parse_tool_call(text: str) -> dict | None:
    """Metindeki SON <tool_call> bloğunu JSON olarak döndürür (bozuksa None)."""
    found = _TOOL_CALL_RE.findall(text or "")
    if not found:
        return None
    try:
        call = json.loads(found[-1])
    except json.JSONDecodeError:
        return None
    args = call.get("arguments", {})
    if isinstance(args, str):  # bazı modeller argümanları string olarak yazar
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"query": args}
    if not isinstance(args, dict) or not str(args.get("query", "")).strip():
        return None
    return {"name": call.get("name", "tavily_search"), "arguments": args}


def split_response(text: str) -> tuple[str, str]:
    """(thought, answer) döndürür. </thought> yoksa tümü düşünce sayılır."""
    if THOUGHT_CLOSE in text:
        thought, _, answer = text.partition(THOUGHT_CLOSE)
        return thought.replace(THOUGHT_OPEN, "").strip(), answer.strip()
    return text.replace(THOUGHT_OPEN, "").strip(), ""


def extract_steps(thought: str) -> dict[int, str]:
    parts = _STEP_RE.split(thought)
    # parts: [önce, "1", metin1, "2", metin2, ...]
    return {int(parts[i]): parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


def assistant_text(messages: list[dict]) -> str:
    """Asistan + araç turlarını tek metin halinde birleştirir (doğrulama için)."""
    buf = []
    for m in messages:
        if m["role"] == "assistant":
            buf.append(m.get("content", ""))
            if m.get("tool_call"):
                buf.append("\n" + format_tool_call(m["tool_call"]) + "\n")
        elif m["role"] == "tool":
            buf.append("<tool_response>\n" + m.get("content", "") + "\n</tool_response>\n")
    return "".join(buf)


def validate_text(text: str) -> tuple[bool, str]:
    """Birleştirilmiş asistan metnini (araç turları dahil) doğrular."""
    if not text.lstrip().startswith(THOUGHT_OPEN) or text.count(THOUGHT_CLOSE) != 1:
        return False, "<thought> etiketleri hatalı"
    thought, answer = split_response(text)
    if len(answer) < 2:
        return False, "nihai yanıt boş"
    order = [int(n) for n in _STEP_RE.findall(thought)]
    if order != [1, 2, 3, 4, 5, 6]:
        return False, f"adım sırası hatalı: {order}"
    steps = extract_steps(thought)
    if any(len(steps.get(i, "")) < 3 for i in range(1, 7)):
        return False, "boş adım var"
    if "<tool_call>" in thought:
        s3, call, s4 = (text.find(STEP_HEADERS[3]), text.find("<tool_call>"), text.find(STEP_HEADERS[4]))
        if not (s3 < call < s4):
            return False, "araç çağrısı STEP 3 ile STEP 4 arasında değil"
        if parse_tool_call(text) is None:
            return False, "araç çağrısı JSON'u bozuk"
    if "<tool_call>" in answer:
        return False, "nihai yanıtta araç çağrısı var"
    return True, "ok"


def validate_messages(messages: list[dict]) -> tuple[bool, str]:
    roles = [m["role"] for m in messages]
    if "user" not in roles or roles[-1] != "assistant":
        return False, "rol sırası hatalı"
    turn = messages[len(roles) - roles[::-1].index("user"):]   # son kullanıcı mesajından sonrası
    has_call = any(m.get("tool_call") for m in turn)
    if has_call and "tool" not in [m["role"] for m in turn]:
        return False, "araç çağrısı var ama araç yanıtı yok"
    return validate_text(assistant_text(turn))


# ---------------------------------------------------------------------------
# GGUF / PocketPal / llama.cpp için gömülü sohbet şablonu (Jinja)
# Sistem mesajı verilmezse GökTürk sistem promptu OTOMATİK eklenir; böylece
# PocketPal gibi uygulamalarda ayrıca prompt yapıştırmak gerekmez.
# ---------------------------------------------------------------------------
CHAT_TEMPLATE = (
    "{%- if messages[0]['role'] != 'system' -%}"
    "{{- '<|im_start|>system\\n' -}}{% raw %}" + SYSTEM_PROMPT + "{% endraw %}{{- '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- for message in messages -%}"
    "{%- if message['role'] == 'tool' -%}"
    "{{- '<|im_start|>user\\n<tool_response>\\n' + message['content'] + '\\n</tool_response><|im_end|>\\n' -}}"
    "{%- elif message['role'] == 'assistant' and ((message.tool_call is defined and message.tool_call) or (message.tool_calls is defined and message.tool_calls)) -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- if message['content'] -%}{{- message['content'].rstrip('\\n') + '\\n' -}}{%- endif -%}"
    "{%- set calls = [message.tool_call] if (message.tool_call is defined and message.tool_call) else message.tool_calls -%}"
    "{%- for c in calls -%}{%- set f = c.function if c.function is defined else c -%}"
    "{{- '<tool_call>\\n{\"name\": \"' + f['name'] + '\", \"arguments\": ' -}}"
    "{%- if f['arguments'] is string -%}{{- f['arguments'] -}}{%- else -%}{{- f['arguments'] | tojson -}}{%- endif -%}"
    "{{- '}\\n</tool_call>' -}}"
    "{%- endfor -%}{{- '<|im_end|>\\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\\n' -}}{%- endif -%}"
)
