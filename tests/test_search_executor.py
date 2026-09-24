"""Çevrimdışı birim testleri: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "inference"), str(ROOT / "scripts")]

from gokturk_format import is_valid  # noqa: E402
from search_executor import (SearchExecutor, SearchResult, format_results,  # noqa: E402
                             DuckDuckGoBackend)


class FakeLLM:
    """Stop dizilerine uyan, senaryolu sahte model."""

    def __init__(self, query):
        self.query, self.prompts = query, []

    def complete(self, prompt, stop, max_tokens):
        self.prompts.append(prompt)
        if "<search_results>" in prompt or (self.query == "YOK" and "</search_query>" in prompt):
            return "<verify>\nSonuçlarla tutarlı.\n</verify>\n<output>\nnginx 1.27 güncel.\n</output>"
        text = f"<think>\nGüncel bilgi lazım.\n</think>\n<plan>\n1. Ara\n</plan>\n<search_query>\n{self.query}\n</search_query>"
        for s in stop:
            if s in text:
                return text[:text.index(s)]
        return text


class FakeSearch:
    name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, q, k=3):
        self.calls.append(q)
        return [SearchResult(f"Başlık {i} <script>", f"https://ornek.com/{i}", "özet " * 5) for i in range(5)][:k]


class TestExecutor(unittest.TestCase):
    def test_search_injected(self):
        llm, s = FakeLLM("nginx son sürüm"), FakeSearch()
        res = SearchExecutor(llm, s).run("nginx son sürüm?")
        self.assertEqual(s.calls, ["nginx son sürüm"])
        self.assertEqual(res.output, "nginx 1.27 güncel.")
        self.assertIn("[3]", res.response)
        self.assertNotIn("[4]", res.response)
        self.assertTrue(is_valid(res.response), res.response)
        self.assertNotIn("<script>", res.response)

    def test_no_search(self):
        llm, s = FakeLLM("YOK"), FakeSearch()
        res = SearchExecutor(llm, s).run("2+2?")
        self.assertEqual(s.calls, [])
        self.assertNotIn("<search_results>", res.response)
        self.assertTrue(is_valid(res.response))

    def test_empty_results(self):
        self.assertIn("Sonuç bulunamadı", format_results([]))

    def test_ddg_html_parser(self):
        page = ('<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fnginx.org%2F&rut=x">'
                'nginx <b>news</b></a><a class="result__snippet" href="#">Stable &amp; mainline</a>'
                '<a class="result__a" href="https://b.com">B</a>')
        import search_executor as se
        orig = se._http
        se._http = lambda *a, **k: page
        try:
            r = DuckDuckGoBackend()._html("q", 3)
        finally:
            se._http = orig
        self.assertEqual(r[0].url, "https://nginx.org/")
        self.assertEqual(r[0].title, "nginx news")
        self.assertEqual(r[0].snippet, "Stable & mainline")
        self.assertEqual(r[1].url, "https://b.com")


if __name__ == "__main__":
    unittest.main()
