#!/usr/bin/env python3
"""GökTürk-2.5B-Thinking — Yerel Arama Çalıştırıcısı (Search Executor).

Akış:
    1. Model ChatML istemiyle çalıştırılır; üretim `</search_query>` görüldüğünde durdurulur.
    2. Etiket Regex ile yakalanır. "YOK" ise üretim olduğu gibi devam eder.
    3. Aksi halde DuckDuckGo / SearXNG ile gerçek arama yapılır; ilk k (=3) sonuç
       <search_results>...</search_results> bloğu olarak modelin bağlamına enjekte edilir.
    4. Model <verify> ve <output> aşamalarını sonuçları görerek tamamlar.

Model sunucusu: OpenAI uyumlu /v1/completions ucu (llama.cpp `llama-server`, vLLM, Ollama)
ya da doğrudan yerel GGUF (llama-cpp-python).

Örnekler:
    llama-server -m GokTurk-2.5B-Thinking-Q4_K_M.gguf -c 4096 --port 8080
    python inference/search_executor.py "nginx için son kritik CVE'ler neler?"
    python inference/search_executor.py --search searxng --searxng-url http://localhost:8888 -i
    python inference/search_executor.py --gguf model.gguf --show-trace "Python'un son sürümü?"

Yalnızca standart kütüphane gerekir. İsteğe bağlı: `pip install ddgs` (daha sağlam DDG),
`pip install llama-cpp-python` (--gguf için).
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
try:
    from gokturk_format import NO_SEARCH, SYSTEM_PROMPT
except ImportError:  # betik tek başına kopyalandıysa
    NO_SEARCH = "YOK"
    SYSTEM_PROMPT = ("Sen GökTürk-2.5B-Thinking adlı Türkçe bir yapay zekâ asistanısın. Her yanıtında "
                     "sırasıyla <think>, <plan>, <search_query>, <verify> ve <output> etiketlerini kullan.")

SEARCH_QUERY_RE = re.compile(r"<search_query>\s*(.*?)\s*(?:</search_query>|$)", re.DOTALL)
OUTPUT_RE = re.compile(r"<output>\s*(.*?)\s*(?:</output>|$)", re.DOTALL)
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) GokTurkSearchExecutor/1.0"
IM_END = "<|im_end|>"


# ---------------------------------------------------------------------------
# Arama arka uçları
# ---------------------------------------------------------------------------
@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class SearchBackend(Protocol):
    name: str

    def search(self, query: str, k: int = 3) -> list[SearchResult]: ...


def _http(url: str, data: dict | None = None, timeout: float = 15.0) -> str:
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={
        "User-Agent": USER_AGENT, "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def _strip_tags(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


class DuckDuckGoBackend:
    """Varsa `ddgs` kütüphanesi, yoksa html.duckduckgo.com HTML sürümü."""
    name = "duckduckgo"

    def __init__(self, region: str = "tr-tr", timeout: float = 15.0):
        self.region, self.timeout = region, timeout

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS  # eski paket adı
            with DDGS() as d:
                rows = d.text(query, region=self.region, max_results=k)
            return [SearchResult(r.get("title", ""), r.get("href", ""), r.get("body", "")) for r in rows][:k]
        except ImportError:
            return self._html(query, k)

    def _html(self, query: str, k: int) -> list[SearchResult]:
        page = _http("https://html.duckduckgo.com/html/", {"q": query, "kl": self.region}, self.timeout)
        results = []
        blocks = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)',
                            page, re.DOTALL)
        for href, title, rest in blocks:
            m = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', rest, re.DOTALL)
            url = href
            if "uddg=" in href:  # DDG yönlendirme bağlantısını çöz
                url = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(href).query)["uddg"][0])
            if "duckduckgo.com/y.js" in url:  # reklam
                continue
            results.append(SearchResult(_strip_tags(title), url, _strip_tags(m.group(1)) if m else ""))
            if len(results) >= k:
                break
        return results


class SearXNGBackend:
    """Kendi barındırdığınız SearXNG (settings.yml → search.formats: [html, json])."""
    name = "searxng"

    def __init__(self, base_url: str = "http://localhost:8888", language: str = "tr", timeout: float = 15.0):
        self.base_url, self.language, self.timeout = base_url.rstrip("/"), language, timeout

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        qs = urllib.parse.urlencode({"q": query, "format": "json", "language": self.language})
        data = json.loads(_http(f"{self.base_url}/search?{qs}", timeout=self.timeout))
        return [SearchResult(r.get("title", ""), r.get("url", ""), r.get("content", ""))
                for r in data.get("results", [])[:k]]


class FallbackBackend:
    """Sırayla arka uçları dener; ilk sonuç döndüreni kullanır."""
    name = "fallback"

    def __init__(self, *backends: SearchBackend):
        self.backends = backends

    def search(self, query: str, k: int = 3) -> list[SearchResult]:
        for b in self.backends:
            try:
                res = b.search(query, k)
                if res:
                    return res
            except Exception as e:  # noqa: BLE001
                print(f"[arama] {b.name} başarısız: {e}", file=sys.stderr)
        return []


def format_results(results: list[SearchResult], max_snippet: int = 400) -> str:
    """Sonuçları modele enjekte edilecek bloğa dönüştürür. Açılı parantezler
    temizlenir ki web içeriği etiket yapısını bozamasın (prompt injection)."""
    def clean(s: str) -> str:
        s = re.sub(r"\s+", " ", s.replace("<", "‹").replace(">", "›")).strip()
        return s[:max_snippet] + ("…" if len(s) > max_snippet else "")
    if not results:
        return "<search_results>\nSonuç bulunamadı.\n</search_results>"
    lines = [f"[{i}] {clean(r.title)}\nURL: {clean(r.url)}\n{clean(r.snippet)}" for i, r in enumerate(results, 1)]
    return "<search_results>\n" + "\n\n".join(lines) + "\n</search_results>"


# ---------------------------------------------------------------------------
# Model arka uçları: (prompt, stop) -> üretilen metin
# ---------------------------------------------------------------------------
class LLM(Protocol):
    def complete(self, prompt: str, stop: list[str], max_tokens: int) -> str: ...


class OpenAICompletionsLLM:
    """llama-server / vLLM / Ollama gibi OpenAI uyumlu /v1/completions ucu."""

    def __init__(self, base_url="http://localhost:8080", model="gokturk", temperature=0.6,
                 top_p=0.95, api_key="sk-local", timeout=300.0):
        self.url = base_url.rstrip("/") + ("/completions" if base_url.rstrip("/").endswith("/v1") else "/v1/completions")
        self.model, self.temperature, self.top_p, self.api_key, self.timeout = model, temperature, top_p, api_key, timeout

    def complete(self, prompt: str, stop: list[str], max_tokens: int) -> str:
        body = json.dumps({"model": self.model, "prompt": prompt, "stop": stop, "max_tokens": max_tokens,
                           "temperature": self.temperature, "top_p": self.top_p}).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["choices"][0]["text"]


class LlamaCppPythonLLM:
    """Sunucusuz, doğrudan GGUF (pip install llama-cpp-python)."""

    def __init__(self, gguf_path: str, n_ctx=4096, temperature=0.6, top_p=0.95, n_gpu_layers=-1):
        from llama_cpp import Llama
        self.llm = Llama(model_path=gguf_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
        self.temperature, self.top_p = temperature, top_p

    def complete(self, prompt: str, stop: list[str], max_tokens: int) -> str:
        out = self.llm(prompt, stop=stop, max_tokens=max_tokens, temperature=self.temperature, top_p=self.top_p)
        return out["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Çalıştırıcı
# ---------------------------------------------------------------------------
def chatml(messages: list[dict]) -> str:
    """Qwen2.5 ChatML şablonu; asistan turu açık bırakılır."""
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}{IM_END}\n" for m in messages)
    return s + "<|im_start|>assistant\n"


@dataclass
class RunResult:
    response: str                      # tam asistan metni (enjekte edilen sonuçlar dahil)
    output: str                        # yalnızca <output> içeriği
    searches: list[tuple[str, list[SearchResult]]]


class SearchExecutor:
    def __init__(self, llm: LLM, search: SearchBackend, k: int = 3, max_searches: int = 2,
                 max_tokens: int = 2048, system_prompt: str = SYSTEM_PROMPT,
                 on_event: Callable[[str, str], None] | None = None):
        self.llm, self.search_backend, self.k = llm, search, k
        self.max_searches, self.max_tokens, self.system_prompt = max_searches, max_tokens, system_prompt
        self.on_event = on_event or (lambda kind, text: None)
        self._cached_search = lru_cache(maxsize=128)(lambda q: tuple(self.search_backend.search(q, self.k)))

    def run(self, user: str, history: list[dict] | None = None) -> RunResult:
        messages = [{"role": "system", "content": self.system_prompt}, *(history or []),
                    {"role": "user", "content": user}]
        prompt = chatml(messages)
        response, searches = "", []
        budget = self.max_tokens
        while budget > 0:
            allow_search = len(searches) < self.max_searches
            stop = [IM_END, "<|endoftext|>"] + (["</search_query>"] if allow_search else [])
            chunk = self.llm.complete(prompt + response, stop=stop, max_tokens=budget)
            budget -= max(1, len(chunk) // 3)  # kaba token tahmini
            response += chunk
            self.on_event("model", chunk)

            query = self._pending_query(response) if allow_search else None
            if query is None:
                break  # model turu bitirdi
            response += "</search_query>\n"
            if query.upper() == NO_SEARCH or not query:
                continue
            self.on_event("search", query)
            try:
                results = list(self._cached_search(query))
            except Exception as e:  # noqa: BLE001 — arama hatası üretimi durdurmasın
                print(f"[arama] hata: {e}", file=sys.stderr)
                results = []
            searches.append((query, results))
            block = format_results(results)
            response += block + "\n"
            self.on_event("results", block)

        m = OUTPUT_RE.search(response)
        return RunResult(response, m.group(1).strip() if m else response.strip(), searches)

    @staticmethod
    def _pending_query(text: str) -> str | None:
        """Metin kapanmamış bir <search_query> ile bitiyorsa sorguyu döndürür."""
        last_open = text.rfind("<search_query>")
        if last_open == -1 or "</search_query>" in text[last_open:]:
            return None
        m = SEARCH_QUERY_RE.search(text, last_open)
        return m.group(1).strip() if m else None


def build_search_backend(args) -> SearchBackend:
    ddg = DuckDuckGoBackend(region=args.region)
    if args.search == "duckduckgo":
        return ddg
    sx = SearXNGBackend(args.searxng_url)
    return sx if args.search == "searxng" else FallbackBackend(sx, ddg)


def main():
    ap = argparse.ArgumentParser(description="GökTürk arama çalıştırıcısı")
    ap.add_argument("prompt", nargs="?", help="Soru (boşsa -i ile etkileşimli mod)")
    ap.add_argument("-i", "--interactive", action="store_true")
    ap.add_argument("--server", default="http://localhost:8080", help="OpenAI uyumlu sunucu")
    ap.add_argument("--model", default="gokturk")
    ap.add_argument("--gguf", help="Sunucu yerine doğrudan GGUF yükle (llama-cpp-python)")
    ap.add_argument("--search", choices=["duckduckgo", "searxng", "auto"], default="duckduckgo")
    ap.add_argument("--searxng-url", default="http://localhost:8888")
    ap.add_argument("--region", default="tr-tr")
    ap.add_argument("-k", type=int, default=3, help="Enjekte edilecek sonuç sayısı")
    ap.add_argument("--max-searches", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--show-trace", action="store_true", help="Tüm düşünce zincirini göster")
    ap.add_argument("--search-only", metavar="SORGU", help="Yalnızca aramayı test et")
    args = ap.parse_args()

    backend = build_search_backend(args)
    if args.search_only:
        print(format_results(backend.search(args.search_only, args.k)))
        return

    llm = (LlamaCppPythonLLM(args.gguf, temperature=args.temperature) if args.gguf
           else OpenAICompletionsLLM(args.server, args.model, temperature=args.temperature))

    def on_event(kind, text):
        if kind == "search":
            print(f"\n🔎 Aranıyor: {text}", file=sys.stderr)
        elif args.show_trace:
            print(text, end="" if kind == "model" else "\n", file=sys.stderr, flush=True)

    ex = SearchExecutor(llm, backend, k=args.k, max_searches=args.max_searches, on_event=on_event)
    history: list[dict] = []

    def ask(q):
        res = ex.run(q, history)
        print(f"\n{res.output}\n")
        if res.searches:
            print("Kaynaklar: " + ", ".join(r.url for _, rs in res.searches for r in rs), file=sys.stderr)
        # geçmişte yalnızca nihai yanıt tutulur (bağlamı küçük tutmak için)
        history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": res.output}])

    if args.prompt:
        ask(args.prompt)
    if args.interactive or not args.prompt:
        print("GökTürk-2.5B-Thinking 🐺  (çıkmak için Ctrl+D)", file=sys.stderr)
        try:
            while q := input("› ").strip():
                ask(q)
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
