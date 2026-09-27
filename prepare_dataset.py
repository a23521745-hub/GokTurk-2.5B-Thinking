#!/usr/bin/env python3
"""GökTürk-2.5B-Thinking — 6 Aşamalı CoT veri hazırlama hattı.

Alt komutlar (tipik sıra):

  1) synthetic : API'siz, cevabı programatik doğrulanmış Türkçe mantık/matematik/kod/siber
                 örnekleri → 6 adımlı biçim. (Arama gerektirenlerde "araç sonuç vermedi"
                 senaryosu: model uydurmamayı öğrenir.)
       python prepare_dataset.py synthetic -n 3000 -o data/synth6.jsonl

  2) seeds     : Açık veri kümelerinden tohum soru + referans cevap çeker
                 (GSM8K, MATH, OpenR1-Math [DeepSeek-R1 CoT], UltraFeedback, ShareGPT, Türkçe set).
       pip install datasets
       python prepare_dataset.py seeds --source gsm8k:800 openr1:800 math:400 ultrafeedback:600 -o data/seeds.jsonl

  3) distill   : Öğretmen LLM (varsayılan Gemini, OpenAI uyumlu uç) + GERÇEK Tavily araması ile
                 tohumları ve teknik Türkçe konu havuzunu 6 adımlı Türkçe örneklere dönüştürür.
                 Matematikte referans cevapla otomatik doğrulama, kaldığı yerden devam (resume).
       export GEMINI_API_KEY=... TAVILY_API_KEY=tvly-...
       python prepare_dataset.py distill --seeds data/seeds.jsonl --topics 600 -o data/distill6.jsonl

  4) merge     : Birleştir, doğrula, tekrarları at, uzunluk süz, train/val ayır.
       python prepare_dataset.py merge data/synth6.jsonl data/distill6.jsonl -o data/gokturk6

Kayıt şeması (JSONL):
  {"messages": [{"role": "system"|"user"|"assistant"|"tool", "content": str, "tool_call"?: {...}}],
   "meta": {"category", "source", "needs_search", "difficulty"?}}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "inference")]
from gokturk_cot import (NEED_SEARCH, NO_SEARCH, SYSTEM_PROMPT, VERIFIED_PREFIX,  # noqa: E402
                         build_messages, render_chatml, validate_messages)


def write_jsonl(records, path: Path, mode="w"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def record(messages, category, source, needs_search, **extra):
    return {"messages": messages, "meta": {"category": category, "source": source,
                                           "needs_search": needs_search, **extra}}


# ===========================================================================
# 1) SYNTHETIC — mevcut doğrulanmış üreticileri 6 adıma dönüştür
# ===========================================================================
CATEGORY_HINTS = {
    "matematik": ("matematik", "adım adım çözüm ve net sayısal sonuç"),
    "mantik": ("mantık", "gerekçeli ve kesin bir sonuç"),
    "kodlama": ("yazılım/kodlama", "çalışan, tip ipuçlu ve açıklamalı kod"),
    "siber_guvenlik": ("siber güvenlik (savunma)", "uygulanabilir, önceliklendirilmiş savunma önerileri"),
}


def _verified_line(output: str) -> str:
    """Nihai yanıttan kısa 'Doğrulanan sonuç' özeti çıkarır (yanıtla birebir tutarlı)."""
    text = re.sub(r"```.*?```", " ", output, flags=re.DOTALL)
    bold = [b.strip() for b in re.findall(r"\*\*(.+?)\*\*", text) if not b.strip().endswith(":")]
    if bold:
        return f"{VERIFIED_PREFIX} " + "; ".join(bold[:3])
    first = re.split(r"(?<=[.!?])\s", re.sub(r"[*`#]", "", text).strip())[0]
    return f"{VERIFIED_PREFIX} " + first[:200]


def _synthetic_one(q: str, p: dict, cat: str, rng: random.Random):
    area, fmt = CATEGORY_HINTS[cat]
    if p["search_query"].strip().upper() != "YOK":
        fmt = "güncel ve doğrulanmış bilgi; doğrulanamıyorsa bunun açıkça belirtilmesi"
    short_q = re.sub(r"\s+", " ", q)[:160]
    s1 = (f"Alan: {area}. Kullanıcı şunu istiyor: «{short_q}». "
          f"Beklenen çıktı: {fmt}. Kısıt: Türkçe, doğru ve doğrulanabilir olmalı.")
    needs = p["search_query"].strip().upper() != "YOK"
    if not needs:
        steps = {
            1: s1,
            2: f"{p['think']}\n{NO_SEARCH} — konu kalıcı bilgi ve hesaplama ile çözülebiliyor.",
            3: "Arama yapılmayacak; içsel bilgi ve hesaplama yeterli.",
            4: "Dış kaynak kullanılmadı. Kullandığım kurallar/formüller standart ve iç tutarlılıkları kontrol edildi.",
            5: p["plan"],
            6: f"{p['verify']}\nDil ve biçim kontrolü: yanıt Türkçe, eksiksiz ve istenen formatta.\n"
               + _verified_line(p["output"]),
        }
        return build_messages(q, steps, p["output"]), False
    # Arama gerektiren: araç hata/boş döndürür → model dürüstçe sınırı belirtmeyi öğrenir
    query = p["search_query"].strip()
    failure = rng.choice([
        ('{"query": %s, "error": "Tavily HTTP 429: rate limit", "results": []}', "Araç hız sınırına takıldı"),
        ('{"query": %s, "results": []}', "Araç hiç sonuç döndürmedi"),
    ])
    tool_resp = failure[0] % json.dumps(query, ensure_ascii=False)
    subject = short_q.split(" için")[0].split(" güncel")[0]
    steps = {
        1: s1,
        2: f"{p['think']}\n{NEED_SEARCH} — bilgi zamana duyarlı; ezberden yanıt yanıltıcı olabilir.",
        3: f"Kısa, anahtar kelime odaklı ve güncelliği vurgulayan bir sorgu kuruyorum: «{query}».",
        4: f"{failure[1]}; doğrulanmış güncel veri yok. Tahmini sürüm/CVE numarası UYDURMAMALIYIM. "
           "Yalnızca zamandan bağımsız, her durumda geçerli önerileri verebilirim.",
        5: "1. Güncel veriyi doğrulayamadığımı açıkça belirt.\n2. Kullanıcının kendisinin kontrol "
           "edebileceği resmi kaynakları ve komutları ver.\n3. Genel güvenlik önerilerini ekle.",
        6: "Yanıtta doğrulanmamış hiçbir sayı, sürüm veya tarih yok; öneriler genel geçer. Dil ve biçim uygun.\n"
           f"{VERIFIED_PREFIX} güncel bilgi doğrulanamadı; tahmini sürüm/CVE verilmeyecek.",
    }
    answer = (f"Şu anda {subject} hakkında **güncel bilgiyi doğrulayamadım** (arama sonuç vermedi), "
              "bu yüzden tahmini sürüm veya CVE numarası vermiyorum.\n\n"
              "**Kendiniz kontrol etmek için:**\n"
              "- Resmi proje/üretici güvenlik duyuruları ve sürüm notları\n"
              "- NVD (nvd.nist.gov) ve cve.org üzerinde ürün adıyla arama\n"
              "- Türkiye için USOM (usom.gov.tr) duyuruları\n"
              "- Kurulu sürüm: `<program> --version` veya paket yöneticiniz (`apt list --upgradable`)\n\n"
              "**Genel öneri:** Sistemlerinizi paket yöneticisiyle güncel tutun ve kritik yamaları "
              "test ortamında doğrulayıp hızla uygulayın. Tekrar sorarsanız aramayı yeniden denerim.")
    return build_messages(q, steps, answer, {"name": "tavily_search", "arguments": {"query": query}},
                          tool_resp), True


def cmd_synthetic(a):
    from generate_dataset import GENERATORS  # scripts/generate_dataset.py
    from gokturk_format import parse_response
    rng = random.Random(a.seed)
    seen, out = set(), []
    counts = {c: 0 for c in GENERATORS}
    stale = {c: 0 for c in GENERATORS}
    while len(out) < a.n and stale:
        cat = min(stale, key=lambda c: counts[c])
        if counts[cat] >= a.per_category_cap:
            del stale[cat]
            continue
        q, resp, diff = rng.choice(GENERATORS[cat])(rng)
        h = hashlib.md5((q + resp).encode()).hexdigest()
        p = parse_response(resp)
        if h in seen or not p:
            stale[cat] += 1
            if stale[cat] > 300:
                del stale[cat]
            continue
        stale[cat] = 0
        seen.add(h)
        msgs, needs = _synthetic_one(q, p, cat, rng)
        ok, why = validate_messages(msgs)
        if not ok:
            raise AssertionError(f"Sentetik örnek geçersiz ({why}): {q}")
        counts[cat] += 1
        out.append(record(msgs, cat, "synthetic", needs, difficulty=diff))
    write_jsonl(out, a.output)
    print(f"✓ {len(out)} örnek → {a.output} | {counts}")
    if len(out) < a.n:
        print("ℹ️  Kod/siber şablon havuzu sınırlı; çeşitlilik için `distill` kullanın.", file=sys.stderr)


# ===========================================================================
# 2) SEEDS — açık veri kümelerinden tohum soru/cevap
# ===========================================================================
def _boxed(sol: str) -> str:
    i = sol.rfind("\\boxed{")
    if i < 0:
        return ""
    depth, j, start = 0, i + 7, i + 7
    while j < len(sol):
        depth += {"{": 1, "}": -1}.get(sol[j], 0)
        if depth < 0:
            return sol[start:j]
        j += 1
    return ""


# ad: (hf_repo, config, split, dönüştürücü, kategori). Depo adları değişirse --source ile özel verin.
SEED_SOURCES = {
    "gsm8k": ("openai/gsm8k", "main", "train",
              lambda r: (r["question"], r["answer"].split("####")[-1].strip().replace(",", ""), r["answer"]),
              "matematik"),
    "math": ("EleutherAI/hendrycks_math", "algebra", "train",
             lambda r: (r["problem"], _boxed(r["solution"]), r["solution"]), "matematik"),
    "openr1": ("open-r1/OpenR1-Math-220k", "default", "train",
               lambda r: (r["problem"], str(r.get("answer", "")),
                          (r.get("generations") or [r.get("solution", "")])[0]), "matematik"),
    "ultrafeedback": ("HuggingFaceH4/ultrafeedback_binarized", None, "train_prefs",
                      lambda r: (r["prompt"], "", r["chosen"][-1]["content"]), "genel"),
    "sharegpt": ("anon8231489123/ShareGPT_Vicuna_unfiltered", None, "train",
                 lambda r: next(((c["value"], "", nxt["value"]) for c, nxt in
                                 zip(r["conversations"], r["conversations"][1:])
                                 if c.get("from") == "human" and nxt.get("from") == "gpt"), ("", "", "")),
                 "genel"),
}


def cmd_seeds(a):
    from datasets import load_dataset  # pip install datasets
    rng = random.Random(a.seed)
    out = []
    for spec in a.source:
        # biçimler: "gsm8k:800"  veya özel: "repo/ad|config|split|soru_alanı|cevap_alanı|kategori:500"
        name, _, n = spec.rpartition(":") if spec.rsplit(":", 1)[-1].isdigit() else (spec, "", "500")
        n = int(n)
        if "|" in name:
            repo, cfg, split, qf, af, cat = (name.split("|") + [""] * 6)[:6]
            conv = (lambda qf, af: lambda r: (r[qf], "", r.get(af, "") if af else ""))(qf, af)
            src = (repo, cfg or None, split or "train", conv, cat or "genel")
        else:
            src = SEED_SOURCES[name]
        repo, cfg, split, conv, cat = src
        print(f"⇣ {repo} [{cfg or '-'}:{split}] × {n}", file=sys.stderr)
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True).shuffle(seed=a.seed, buffer_size=5000)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️  yüklenemedi ({e}); atlanıyor", file=sys.stderr)
            continue
        got = 0
        for r in ds:
            try:
                q, ref, hint = conv(r)
            except (KeyError, TypeError, IndexError):
                continue
            if not q or len(q) > a.max_q_chars or len(q) < 15:
                continue
            out.append({"id": hashlib.md5(q.encode()).hexdigest()[:12], "source": repo, "category": cat,
                        "question": q.strip(), "reference": (ref or "").strip(),
                        "reasoning_hint": (hint or "")[: a.max_hint_chars]})
            got += 1
            if got >= n:
                break
        print(f"  ✓ {got}", file=sys.stderr)
    rng.shuffle(out)
    write_jsonl(out, a.output)
    print(f"✓ {len(out)} tohum → {a.output}")


# ===========================================================================
# 3) DISTILL — Öğretmen LLM + gerçek Tavily araması
# ===========================================================================
TOPIC_POOL = {
    "siber_guvenlik": [
        "OWASP Top 10 zafiyetlerinin tespiti ve önlenmesi", "güvenli kod incelemesi (Python/JS/Go)",
        "Linux sunucu sıkılaştırma", "log analizi ve olay müdahalesi (SIEM, Sigma kuralları)",
        "TLS/PKI ve kriptografi temelleri", "Active Directory güvenliği (savunma)",
        "bulut (AWS/Azure) IAM yanlış yapılandırmaları", "KVKK uyumlu veri güvenliği",
        "{guncel} güncel kritik CVE ve yamalar", "{guncel} fidye yazılımı kampanyaları ve IOC takibi"],
    "kodlama": [
        "Python algoritma ve veri yapıları", "asenkron programlama (asyncio)", "SQL sorgu optimizasyonu",
        "Rust sahiplik ve ödünç alma", "REST/GraphQL API tasarımı", "Docker ve Kubernetes hata ayıklama",
        "hata ayıklama: verilen hatalı kodu düzelt", "{guncel} bir kütüphanenin son sürümündeki değişiklikler"],
    "mantik": ["çok adımlı mantık bulmacaları", "önermeler mantığı ve doğruluk tabloları",
               "olasılık paradoksları", "karar verme ve tümdengelim"],
    "matematik": ["olasılık ve kombinatorik", "sayı teorisi", "lineer cebir", "türev/integral uygulamaları"],
    "genel_teknik": ["{guncel} yapay zekâ/LLM gelişmeleri", "{guncel} Türkiye teknoloji ve regülasyon haberleri",
                     "bilgisayar ağları (TCP/IP, DNS, BGP)", "işletim sistemleri (bellek, zamanlayıcı)"],
}

DECIDE_PROMPT = """Aşağıdaki kullanıcı sorusunu yanıtlamak için GÜNCEL/CANLI web bilgisi gerekiyor mu?
(Güncel sürümler, CVE'ler, haberler, fiyatlar, tarihler → evet. Kalıcı bilgi, matematik, kod yazma → hayır.)
Yalnızca JSON döndür:
{{"needs_search": true|false, "query": "kısa anahtar kelime sorgusu (gerekmiyorsa boş)",
  "topic": "general"|"news", "time_range": null|"week"|"month"|"year"}}

SORU: {q}"""

QUESTION_PROMPT = """Türkçe bir yapay zekâ eğitim veri seti için "{topic}" konusunda, {diff} zorlukta,
özgün, net ve tek bir kullanıcı sorusu yaz. Siber güvenlikte yalnızca savunma/eğitim odaklı ol.
{extra}Yalnızca soruyu yaz."""

GEN_PROMPT = """Sen GökTürk-2.5B-Thinking modeli için öğretmen modelsin. Aşağıdaki soruya, 6 adımlı düşünme
zinciri ve nihai yanıt üret. TÜM metin kusursuz, doğal TÜRKÇE olmalı (soru İngilizce olsa bile soruyu da
Türkçeye çevir).

Adımlar:
step1 Problem Sentezleme & Ayrıştırma: istek, kısıtlar, beklenen format (2-4 cümle)
step2 İçsel Akıl Yürütme: kendi bilginle çözüm fikri/çekirdek akıl yürütme; SONUNDA tam olarak
      "{marker}" ve kısa gerekçe
step3 Web Araştırması & Kaynak Toplama: {step3_rule}
step4 Kaynak Doğrulama & Yerelde İşleme: {step4_rule}
step5 Yanıt Planlama & Sunum Hazırlığı: numaralı plan
step6 Öz-Denetim & Eksiklik Giderme: sonucu bağımsız yoldan sağla, hataları düzelt, dil/biçim kontrolü;
      SON SATIR tam olarak "{verified} <kısa nihai sonuç>" olmalı
answer: kullanıcıya nihai yanıt (markdown; kod gerekiyorsa kod bloğu). Yanıttaki sonuç, step6'daki
      doğrulanan sonuçla BİREBİR aynı olmalı (aynı sayılar/birimler). {answer_rule}

Kurallar: Düşünce adımları öz olsun (toplam ~150-450 kelime). Uydurma bilgi, sürüm, tarih YOK.
{ref_rule}
Yalnızca şu JSON'u döndür:
{{"question_tr": "...", "step1": "...", "step2": "...", "step3": "...", "step4": "...",
  "step5": "...", "step6": "...", "answer": "..."}}

SORU:
{q}
{hint}{tool}"""


class Teacher:
    def __init__(self, base_url, model, key_env, temperature=0.7):
        from openai import OpenAI  # pip install openai
        key = os.environ.get(key_env)
        if not key:
            sys.exit(f"[hata] {key_env} tanımlı değil")
        self.client, self.model, self.temperature = OpenAI(base_url=base_url, api_key=key), model, temperature

    def ask(self, prompt: str, json_mode=True, temperature=None) -> str:
        kw = {"response_format": {"type": "json_object"}} if json_mode else {}
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature if temperature is None else temperature,
            messages=[{"role": "user", "content": prompt}], **kw)
        return (r.choices[0].message.content or "").strip()

    def ask_json(self, prompt: str) -> dict:
        txt = self.ask(prompt)
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        if not m:
            raise ValueError("JSON bulunamadı")
        return json.loads(m.group(0))


def _norm_num(s: str) -> str:
    s = s.replace("\\!", "").replace("$", "").replace(" ", "").replace(",", ".").rstrip(".")
    try:
        f = float(s)
        return str(int(f)) if f.is_integer() else f"{f:.4f}".rstrip("0")
    except ValueError:
        return s


def answer_matches(answer: str, ref: str) -> bool:
    if not ref:
        return True
    r = _norm_num(ref)
    nums = {_norm_num(x) for x in re.findall(r"-?\d+(?:[.,]\d+)?", answer.replace("\u202f", ""))}
    return r in nums or ref.replace(" ", "") in answer.replace(" ", "")


def distill_one(item: dict, teacher: Teacher, search, a, rng: random.Random):
    q, ref, cat = item["question"], item.get("reference", ""), item.get("category", "genel")
    # (a) arama kararı — matematik tohumlarında gereksiz
    decision = {"needs_search": False}
    if cat not in ("matematik",):
        decision = teacher.ask_json(DECIDE_PROMPT.format(q=q))
    needs = bool(decision.get("needs_search")) and bool(str(decision.get("query", "")).strip())
    tool_resp, call = None, None
    if needs:
        if search is None:
            return None  # gerçek arama yoksa uydurma sonuç üretme
        args = {"query": decision["query"].strip(), "topic": decision.get("topic") or "general"}
        if decision.get("time_range"):
            args["time_range"] = decision["time_range"]
        tool_resp, sources = search(args)
        call = {"name": "tavily_search", "arguments": args}
    # (b) 6 adımlı üretim
    prompt = GEN_PROMPT.format(
        q=q, marker=NEED_SEARCH if needs else NO_SEARCH, verified=VERIFIED_PREFIX,
        step3_rule=(f'neden aranması gerektiğini ve kurulan sorguyu ("{call["arguments"]["query"]}") anlat'
                    if needs else "aramanın neden gereksiz olduğunu tek cümleyle belirt"),
        step4_rule=("ARAÇ SONUÇLARINI süz: hangi kaynak güvenilir, hangisi eski/çelişkili; yalnızca "
                    "doğrulanan bilgileri tut. Sonuç yoksa/hatalıysa bunu açıkça söyle" if needs else
                    "dış kaynak kullanılmadığını belirt ve kullanılan kural/bilgilerin iç tutarlılığını denetle"),
        answer_rule=("Web bilgisini [1], [2] biçiminde yalnızca var olan kaynak id'leriyle an; sonuç yoksa "
                     "güncel bilgiyi doğrulayamadığını dürüstçe söyle." if needs else ""),
        ref_rule=(f"Referans (doğru) nihai cevap: {ref} — yanıtın bununla AYNI olmalı." if ref else ""),
        hint=(f"\nREFERANS AKIL YÜRÜTME (İngilizce/ham olabilir, yalnızca mantık için kullan):\n"
              f"{item['reasoning_hint'][:a.max_hint_chars]}\n" if item.get("reasoning_hint") else ""),
        tool=(f"\nARAÇ YANITI (tavily_search):\n{tool_resp}\n" if needs else ""),
    )
    for _ in range(a.retries):
        try:
            d = teacher.ask_json(prompt)
            steps = {i: str(d[f"step{i}"]) for i in range(1, 7)}
            if (NEED_SEARCH if needs else NO_SEARCH) not in steps[2]:
                steps[2] = steps[2].rstrip() + "\n" + (NEED_SEARCH if needs else NO_SEARCH)
            answer = str(d["answer"])
            if not answer_matches(answer, ref):
                continue
            user_q = str(d.get("question_tr") or q) if a.translate_questions else q
            msgs = build_messages(user_q, steps, answer, call, tool_resp)
            ok, _ = validate_messages(msgs)
            if ok:
                return record(msgs, cat, f"distill:{teacher.model}" + ("+tavily" if needs else ""), needs,
                              seed_source=item.get("source", "topic"))
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return None


def cmd_distill(a):
    teacher = Teacher(a.teacher_base_url, a.teacher_model, a.teacher_key_env)
    search = None
    if os.environ.get("TAVILY_API_KEY"):
        from tavily_rag_handler import TavilyProvider
        search = TavilyProvider(k=3, depth=a.search_depth)
    else:
        print("⚠️  TAVILY_API_KEY yok → arama gerektiren örnekler ATLANACAK.", file=sys.stderr)
    rng = random.Random(a.seed)

    items = []
    for f in a.seeds or []:
        items += list(read_jsonl(Path(f)))
    if a.max_seeds:
        items = items[: a.max_seeds]
    # konu havuzundan soru üret (search_ratio kadarı güncel bilgi gerektirsin)
    if a.topics:
        def gen_q(i):
            r = random.Random(a.seed + i)
            cat = r.choice(list(TOPIC_POOL))
            topic = r.choice(TOPIC_POOL[cat])
            current = "{guncel}" in topic or r.random() < a.search_ratio
            topic = topic.replace("{guncel} ", "")
            extra = "Soru güncel/zamana duyarlı bilgi gerektirsin. " if current else \
                "Soru güncel bilgi gerektirmesin. "
            q = teacher.ask(QUESTION_PROMPT.format(topic=topic, diff=r.choice(["orta", "zor"]), extra=extra),
                            json_mode=False, temperature=1.0)
            return {"id": hashlib.md5(q.encode()).hexdigest()[:12], "source": "topic", "category": cat,
                    "question": q.strip().strip('"')}
        with ThreadPoolExecutor(a.workers) as ex:
            for fut in as_completed([ex.submit(gen_q, i) for i in range(a.topics)]):
                try:
                    items.append(fut.result())
                except Exception as e:  # noqa: BLE001
                    print(f"[soru] {e}", file=sys.stderr)

    done = set()
    if a.output.exists():  # resume
        done = {hashlib.md5(r["messages"][1]["content"].encode()).hexdigest()[:12] for r in read_jsonl(a.output)}
        done |= {r["meta"].get("seed_id") for r in read_jsonl(a.output)}
    items = [it for it in items if it["id"] not in done]
    print(f"▶ {len(items)} öğe işlenecek ({len(done)} zaten tamam)", file=sys.stderr)

    lock, ok, fail = threading.Lock(), 0, 0
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(distill_one, it, teacher, search, a, random.Random(rng.random())): it for it in items}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001
                rec = None
                print(f"[hata] {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
            with lock:
                if rec:
                    rec["meta"]["seed_id"] = futs[fut]["id"]
                    write_jsonl([rec], a.output, mode="a")  # anında diske → çökse de kaybolmaz
                    ok += 1
                else:
                    fail += 1
                if i % 10 == 0:
                    print(f"  {i}/{len(items)}  ✓{ok} ✗{fail}", file=sys.stderr)
    print(f"✓ distill bitti: {ok} geçerli, {fail} reddedildi → {a.output}")


# ===========================================================================
# 4) MERGE
# ===========================================================================
def cmd_merge(a):
    rng = random.Random(a.seed)
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    seen, recs, bad, long_ = set(), [], 0, 0
    for f in a.inputs:
        for r in read_jsonl(Path(f)):
            msgs = r["messages"]
            if msgs[0]["role"] != "system":
                msgs.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
            else:
                msgs[0]["content"] = SYSTEM_PROMPT  # tek tip sistem promptu
            ok, _ = validate_messages(msgs)
            user = next(m["content"] for m in msgs if m["role"] == "user")
            h = hashlib.md5(re.sub(r"\W+", "", user.lower()).encode()).hexdigest()
            if not ok:
                bad += 1
                continue
            if h in seen:
                continue
            text = render_chatml(msgs)
            n_tok = len(tok(text).input_ids) if tok else len(text) // 3
            if n_tok > a.max_tokens:
                long_ += 1
                continue
            seen.add(h)
            r["meta"]["approx_tokens"] = n_tok
            recs.append(r)
    rng.shuffle(recs)
    n_val = max(1, int(len(recs) * a.val_ratio)) if len(recs) > 50 else 0
    a.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(recs[n_val:], a.output / "train.jsonl")
    write_jsonl(recs[:n_val], a.output / "val.jsonl")
    from collections import Counter
    cats = Counter(r["meta"]["category"] for r in recs)
    srch = sum(r["meta"]["needs_search"] for r in recs)
    print(f"✓ train={len(recs) - n_val} val={n_val} | geçersiz={bad} uzun={long_} | "
          f"arama={srch} | {dict(cats)} → {a.output}/")


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("synthetic")
    s.add_argument("-n", type=int, default=3000)
    s.add_argument("--per-category-cap", type=int, default=600,
                   help="şablon ezberini önlemek için kategori başına üst sınır")
    s.add_argument("-o", "--output", type=Path, default=Path("data/synth6.jsonl"))
    s.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("seeds")
    s.add_argument("--source", nargs="+", default=["gsm8k:800", "openr1:800", "math:300", "ultrafeedback:500"],
                   help=f"Hazır: {list(SEED_SOURCES)} (ad:adet) veya 'repo|config|split|soru|cevap|kategori:adet'")
    s.add_argument("-o", "--output", type=Path, default=Path("data/seeds.jsonl"))
    s.add_argument("--max-q-chars", type=int, default=1500)
    s.add_argument("--max-hint-chars", type=int, default=4000)
    s.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("distill")
    s.add_argument("--seeds", nargs="*", help="seeds çıktısı JSONL dosyaları")
    s.add_argument("--max-seeds", type=int, default=0)
    s.add_argument("--topics", type=int, default=0, help="konu havuzundan üretilecek soru sayısı")
    s.add_argument("--search-ratio", type=float, default=0.3)
    s.add_argument("-o", "--output", type=Path, default=Path("data/distill6.jsonl"))
    s.add_argument("--workers", type=int, default=6)
    s.add_argument("--retries", type=int, default=3)
    s.add_argument("--max-hint-chars", type=int, default=3000)
    s.add_argument("--search-depth", choices=["basic", "advanced"], default="basic")
    s.add_argument("--no-translate-questions", dest="translate_questions", action="store_false")
    s.add_argument("--teacher-base-url", default=os.environ.get(
        "TEACHER_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"))
    s.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "gemini-2.5-flash"))
    s.add_argument("--teacher-key-env", default="GEMINI_API_KEY")
    s.add_argument("--seed", type=int, default=42)

    s = sub.add_parser("merge")
    s.add_argument("inputs", nargs="+")
    s.add_argument("-o", "--output", type=Path, default=Path("data/gokturk6"))
    s.add_argument("--val-ratio", type=float, default=0.02)
    s.add_argument("--max-tokens", type=int, default=2048)
    s.add_argument("--tokenizer", help="kesin token sayımı için (ör. Qwen/Qwen2.5-3B-Instruct)")
    s.add_argument("--seed", type=int, default=42)

    a = ap.parse_args()
    {"synthetic": cmd_synthetic, "seeds": cmd_seeds, "distill": cmd_distill, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
