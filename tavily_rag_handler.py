#!/usr/bin/env python3
"""GökTürk-2.5B-Thinking — Tavily API & Tool Calling / RAG katmanı.

STEP 3'te model `<tool_call>{"name": "tavily_search", ...}</tool_call>` üretir →
bu katman çağrıyı yakalar, Tavily'ye gönderir, dönen JSON'u temizler/süzer/kısaltır ve
`<tool_response>` olarak bağlama enjekte eder → model STEP 4'ten (kaynak doğrulama)
devam eder ve atıflı nihai yanıtı yazar.

Kullanım:
    export TAVILY_API_KEY=tvly-...
    llama-server -m GokTurk-2.5B-Thinking-Q4_K_M.gguf -c 4096 -t 4 --port 8080
    python tavily_rag_handler.py "OpenSSH için son kritik CVE hangisi?" --show-thought
    python tavily_rag_handler.py -i                        # sohbet modu
    python tavily_rag_handler.py --gguf model.gguf "..."   # sunucusuz (llama-cpp-python)
    python tavily_rag_handler.py --search-only "nginx latest version"

Yalnızca standart kütüphane. TAVILY_API_KEY yoksa DuckDuckGo'ya (inference/search_executor.py) düşer.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "inference")]
from gokturk_cot import (IM_END, SYSTEM_PROMPT, parse_tool_call,  # noqa: E402
                         render_chatml, split_response)

TAVILY_URL = "https://api.tavily.com/search"

# Kaynak güvenilirliği için basit alan adı sezgileri (STEP 4'e yardımcı ön süzgeç)
TRUSTED = re.compile(r"(\.gov(\.tr)?|\.edu(\.tr)?|nvd\.nist\.gov|cve\.org|mitre\.org|owasp\.org|"
                     r"python\.org|nodejs\.org|kernel\.org|github\.com|docs\.|wikipedia\.org|"
                     r"arxiv\.org|resmigazete\.gov\.tr|tcmb\.gov\.tr|tuik\.gov\.tr|usom\.gov\.tr)", re.I)
LOW_QUALITY = re.compile(r"(pinterest\.|quora\.com|facebook\.com|tiktok\.com|instagram\.com)", re.I)


# ---------------------------------------------------------------------------
# Tavily istemcisi
# ---------------------------------------------------------------------------
class TavilyError(RuntimeError):
    pass


class TavilyClient:
    def __init__(self, api_key: str | None = None, timeout: float = 20.0, retries: int = 3):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise TavilyError("TAVILY_API_KEY tanımlı değil")
        self.timeout, self.retries = timeout, retries

    def search(self, query: str, *, topic: str = "general", time_range: str | None = None,
               max_results: int = 5, search_depth: str = "basic", include_answer: bool = True,
               country: str | None = None) -> dict:
        body = {"query": query[:400], "topic": topic if topic in ("general", "news") else "general",
                "max_results": max_results, "search_depth": search_depth,
                "include_answer": include_answer, "include_raw_content": False, "include_images": False}
        if time_range in ("day", "week", "month", "year"):
            body["time_range"] = time_range
        if country:
            body["country"] = country
        req = urllib.request.Request(TAVILY_URL, data=json.dumps(body).encode(), method="POST", headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                if e.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                detail = e.read().decode(errors="replace")[:300]
                raise TavilyError(f"Tavily HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.retries:
                    time.sleep(2 ** attempt)
                    continue
                raise TavilyError(f"Tavily bağlantı hatası: {e}") from e
        raise TavilyError("Tavily: deneme hakkı bitti")


# ---------------------------------------------------------------------------
# Yanıt temizleme → bağlam
# ---------------------------------------------------------------------------
@dataclass
class Source:
    id: int
    title: str
    url: str
    content: str
    score: float = 0.0
    date: str | None = None


def _clean_text(s: str, limit: int) -> str:
    s = re.sub(r"<[^>]+>", " ", s or "")                         # HTML kalıntıları
    s = s.replace("<", "‹").replace(">", "›")                    # etiket enjeksiyonunu engelle
    s = re.sub(r"(?i)(ignore (all )?previous instructions|önceki talimatları yok say)", "[filtrelendi]", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > limit:
        cut = s[:limit]
        s = cut[: cut.rfind(". ") + 1] if ". " in cut[limit // 2:] else cut + "…"
    return s


def clean_tavily_response(raw: dict, k: int = 3, min_score: float = 0.25,
                          max_chars: int = 500) -> list[Source]:
    """Tavily JSON → en iyi k benzersiz, temizlenmiş kaynak."""
    items, seen_domains = [], set()
    for r in raw.get("results", []):
        url, content = r.get("url", ""), r.get("content", "")
        if not url or len(content.strip()) < 40 or LOW_QUALITY.search(url):
            continue
        score = float(r.get("score") or 0) + (0.15 if TRUSTED.search(url) else 0.0)
        if score < min_score:
            continue
        domain = urllib.parse.urlparse(url).netloc.removeprefix("www.")
        if domain in seen_domains:          # alan adı çeşitliliği → çapraz doğrulama
            continue
        seen_domains.add(domain)
        items.append((score, r, url))
    items.sort(key=lambda x: -x[0])
    return [Source(i, _clean_text(r.get("title", ""), 120), url, _clean_text(r.get("content", ""), max_chars),
                   round(score, 3), r.get("published_date"))
            for i, (score, r, url) in enumerate(items[:k], 1)]


def to_tool_response(query: str, sources: list[Source], answer: str | None = None,
                     error: str | None = None) -> str:
    """Modele enjekte edilecek kompakt JSON (eğitim verisiyle aynı şema)."""
    payload: dict = {"query": query}
    if error:
        payload["error"] = error
    if answer:
        payload["summary"] = _clean_text(answer, 400)
    payload["results"] = [{k: v for k, v in {"id": s.id, "title": s.title, "url": s.url,
                                             "date": s.date, "content": s.content}.items() if v}
                          for s in sources]
    return json.dumps(payload, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Arama sağlayıcıları: query/args → (tool_response_json, sources)
# ---------------------------------------------------------------------------
class SearchProvider(Protocol):
    def __call__(self, args: dict) -> tuple[str, list[Source]]: ...


class TavilyProvider:
    def __init__(self, client: TavilyClient | None = None, k: int = 3, depth: str = "basic"):
        self.client, self.k, self.depth = client or TavilyClient(), k, depth
        self._cached = lru_cache(maxsize=256)(self._search)

    def _search(self, query: str, topic: str, time_range: str | None) -> tuple[str, tuple]:
        try:
            raw = self.client.search(query, topic=topic, time_range=time_range,
                                     max_results=self.k + 3, search_depth=self.depth)
        except TavilyError as e:
            return to_tool_response(query, [], error=str(e)[:200]), ()
        sources = clean_tavily_response(raw, self.k)
        return to_tool_response(query, sources, raw.get("answer")), tuple(sources)

    def __call__(self, args: dict) -> tuple[str, list[Source]]:
        resp, src = self._cached(str(args.get("query", "")).strip(), args.get("topic", "general"),
                                 args.get("time_range"))
        return resp, list(src)


class DuckDuckGoProvider:
    """TAVILY_API_KEY yoksa ücretsiz yedek."""

    def __init__(self, k: int = 3):
        from search_executor import DuckDuckGoBackend
        self.backend, self.k = DuckDuckGoBackend(), k

    def __call__(self, args: dict) -> tuple[str, list[Source]]:
        q = str(args.get("query", "")).strip()
        try:
            res = self.backend.search(q, self.k)
        except Exception as e:  # noqa: BLE001
            return to_tool_response(q, [], error=f"arama hatası: {e}"[:200]), []
        src = [Source(i, _clean_text(r.title, 120), r.url, _clean_text(r.snippet, 500))
               for i, r in enumerate(res, 1)]
        return to_tool_response(q, src), src


def default_provider(k: int = 3) -> SearchProvider:
    if os.environ.get("TAVILY_API_KEY"):
        return TavilyProvider(k=k)
    print("[rag] TAVILY_API_KEY yok → DuckDuckGo yedeği kullanılıyor", file=sys.stderr)
    return DuckDuckGoProvider(k=k)


# ---------------------------------------------------------------------------
# Model arka uçları
# ---------------------------------------------------------------------------
class LLM(Protocol):
    def complete(self, prompt: str, stop: list[str], max_tokens: int) -> str: ...


class OpenAICompletionsLLM:
    """llama-server / vLLM / Ollama — ham ChatML ile /v1/completions."""

    def __init__(self, base_url="http://localhost:8080", model="gokturk", temperature=0.6,
                 top_p=0.95, api_key="sk-local", timeout=600.0):
        base = base_url.rstrip("/")
        self.url = base + ("/completions" if base.endswith("/v1") else "/v1/completions")
        self.model, self.temperature, self.top_p = model, temperature, top_p
        self.api_key, self.timeout = api_key, timeout

    def complete(self, prompt: str, stop: list[str], max_tokens: int) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stop": stop, "max_tokens": max_tokens,
                           "temperature": self.temperature, "top_p": self.top_p,
                           "repeat_penalty": 1.05, "cache_prompt": True}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["text"]


class LlamaCppPythonLLM:
    """Sunucusuz yerel GGUF. 2.5 GB RAM hedefi için n_ctx=4096 ve mmap önerilir."""

    def __init__(self, gguf_path: str, n_ctx: int = 4096, temperature=0.6, top_p=0.95,
                 n_threads: int | None = None, n_gpu_layers: int = 0):
        from llama_cpp import Llama  # pip install llama-cpp-python
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers,
                         use_mmap=True, verbose=False)
        self.temperature, self.top_p = temperature, top_p

    def complete(self, prompt: str, stop: list[str], max_tokens: int) -> str:
        return self.llm(prompt, stop=stop, max_tokens=max_tokens, temperature=self.temperature,
                        top_p=self.top_p, repeat_penalty=1.05)["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Ajan döngüsü
# ---------------------------------------------------------------------------
@dataclass
class AgentResult:
    answer: str
    thought: str
    transcript: str                         # tam asistan metni (araç turları dahil)
    tool_calls: list[dict] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)


class GokturkRAGAgent:
    def __init__(self, llm: LLM, search: SearchProvider | None = None, max_tool_calls: int = 2,
                 max_tokens: int = 1536, system_prompt: str = SYSTEM_PROMPT,
                 on_event: Callable[[str, str], None] | None = None):
        self.llm, self.search = llm, search or default_provider()
        self.max_tool_calls, self.max_tokens, self.system_prompt = max_tool_calls, max_tokens, system_prompt
        self.on_event = on_event or (lambda kind, text: None)

    def run(self, user: str, history: list[dict] | None = None) -> AgentResult:
        messages = [{"role": "system", "content": self.system_prompt}, *(history or []),
                    {"role": "user", "content": user}]
        transcript, calls, sources = "", [], []
        for _ in range(self.max_tool_calls + 1):
            can_call = len(calls) < self.max_tool_calls
            stop = [IM_END, "<|endoftext|>"] + (["</tool_call>"] if can_call else [])
            chunk = self.llm.complete(render_chatml(messages, add_generation_prompt=True),
                                      stop=stop, max_tokens=self.max_tokens)
            self.on_event("model", chunk)
            call = parse_tool_call(chunk + "</tool_call>") if (can_call and "<tool_call>" in chunk) else None
            if call is None:
                # araç hakkı bittiyse / JSON bozuksa yarım kalan çağrıyı temizle
                chunk = re.sub(r"<tool_call>.*?(</tool_call>|$)", "", chunk, flags=re.DOTALL)
                messages.append({"role": "assistant", "content": chunk})
                transcript += chunk
                break
            head = chunk.split("<tool_call>")[0]
            self.on_event("search", json.dumps(call["arguments"], ensure_ascii=False))
            resp, src = self.search(call["arguments"])
            # kaynak numaralarını genel listeye göre kaydır (çoklu arama)
            offset = len(sources)
            for s in src:
                s.id += offset
            if offset:
                data = json.loads(resp)
                for r in data.get("results", []):
                    r["id"] += offset
                resp = json.dumps(data, ensure_ascii=False)
            sources += src
            calls.append(call)
            self.on_event("results", resp)
            messages += [{"role": "assistant", "content": head, "tool_call": call},
                         {"role": "tool", "content": resp}]
            transcript += head + f"\n<tool_call>{json.dumps(call, ensure_ascii=False)}</tool_call>\n" \
                                 f"<tool_response>{resp}</tool_response>\n"
        thought, answer = split_response(transcript)
        return AgentResult(answer or transcript.strip(), thought, transcript, calls, sources)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="GökTürk Tavily RAG istemcisi")
    ap.add_argument("prompt", nargs="?")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--server", default=os.environ.get("GOKTURK_SERVER", "http://localhost:8080"))
    ap.add_argument("--gguf", help="llama-cpp-python ile doğrudan GGUF yükle")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("-k", type=int, default=3, help="enjekte edilecek kaynak sayısı")
    ap.add_argument("--depth", choices=["basic", "advanced"], default="basic")
    ap.add_argument("--max-tool-calls", type=int, default=2)
    ap.add_argument("--show-thought", action="store_true")
    ap.add_argument("--search-only", metavar="SORGU")
    ap.add_argument("--json", action="store_true", help="sonucu JSON olarak yazdır")
    a = ap.parse_args()

    provider = TavilyProvider(k=a.k, depth=a.depth) if os.environ.get("TAVILY_API_KEY") else default_provider(a.k)
    if a.search_only:
        print(json.dumps(json.loads(provider({"query": a.search_only})[0]), ensure_ascii=False, indent=2))
        return

    llm = (LlamaCppPythonLLM(a.gguf, n_ctx=a.n_ctx, temperature=a.temperature) if a.gguf
           else OpenAICompletionsLLM(a.server, temperature=a.temperature))

    def on_event(kind, text):
        if kind == "search":
            print(f"\n🔎 Tavily: {text}", file=sys.stderr)
        elif a.show_thought:
            print(text if kind == "model" else f"\n📥 {text[:600]}\n", end="", file=sys.stderr, flush=True)

    agent = GokturkRAGAgent(llm, provider, max_tool_calls=a.max_tool_calls, on_event=on_event)
    history: list[dict] = []

    def ask(q: str):
        res = agent.run(q, history)
        if a.json:
            print(json.dumps({"answer": res.answer, "tool_calls": res.tool_calls,
                              "sources": [s.__dict__ for s in res.sources]}, ensure_ascii=False, indent=2))
        else:
            print(f"\n{res.answer}\n")
            for s in res.sources:
                print(f"  [{s.id}] {s.title} — {s.url}")
        # bağlamı küçük tutmak için geçmişte yalnızca nihai yanıt saklanır
        history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": res.answer}])
        del history[:-8]

    if a.prompt:
        ask(a.prompt)
    if a.interactive or not a.prompt:
        print("🐺 GökTürk-2.5B-Thinking (çıkış: Ctrl+D)", file=sys.stderr)
        try:
            while q := input("› ").strip():
                ask(q)
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
