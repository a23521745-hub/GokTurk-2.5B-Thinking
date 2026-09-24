#!/usr/bin/env python3
"""GökTürk-2.5B-Thinking sentetik veri seti üreticisi.

İki mod:
  synthetic : API gerektirmez. Mantık, siber güvenlik, kodlama ve matematik
              için prosedürel olarak üretilen, cevabı PROGRAMATİK OLARAK
              doğrulanmış örnekler (kod örnekleri gerçekten çalıştırılır).
  distill   : Hibrit damıtma. (1) DeepSeek-R1'den mantık adımları (reasoning)
              alınır, (2) Qwen-Max/72B bu adımları yüksek kaliteli Türkçe ile
              5 aşamalı GökTürk formatına dönüştürür. OpenAI uyumlu uç noktalar
              kullanılır (DashScope, DeepSeek, vLLM, OpenRouter ...).

Örnekler:
  python scripts/generate_dataset.py synthetic -n 5000 -o data/gokturk_synth.jsonl
  export DEEPSEEK_API_KEY=... DASHSCOPE_API_KEY=...
  python scripts/generate_dataset.py distill -n 500 -o data/gokturk_distill.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gokturk_format import (build_response,  # noqa: E402
                            is_valid, parse_response, to_record)

# ---------------------------------------------------------------------------
# MATEMATİK
# ---------------------------------------------------------------------------

def math_linear(r: random.Random):
    a, x = r.randint(2, 12), r.randint(-20, 30)
    b = r.randint(-50, 50)
    c = a * x + b
    sb = f"+ {b}" if b >= 0 else f"- {-b}"
    q = f"{a}x {sb} = {c} denklemini çöz."
    think = (f"Birinci dereceden bir denklem. x'i yalnız bırakmam gerekiyor. "
             f"Önce {b}'yi karşıya atarım: {a}x = {c} - ({b}) = {c - b}. "
             f"Sonra {a}'ya bölerim: x = {c - b}/{a} = {x}.")
    plan = [f"Sabit terimi ({b}) eşitliğin sağına taşı.", f"Her iki tarafı {a}'ya böl.",
            "Bulunan değeri denklemde yerine koyarak sağlama yap."]
    verify = f"Sağlama: {a}·({x}) {sb} = {a * x} {sb} = {c}. Sol taraf sağ tarafa eşit, çözüm doğru."
    out = f"{a}x = {c - b} olur, buradan **x = {x}** bulunur."
    return q, build_response(think, plan, verify, out), "kolay"


def math_percent(r: random.Random):
    price = r.choice(range(100, 5001, 50))
    p1, p2 = r.choice([10, 15, 20, 25, 30]), r.choice([5, 10, 20])
    after1 = price * (100 + p1) / 100
    final = after1 * (100 - p2) / 100
    q = (f"Bir ürünün fiyatı {price} TL. Önce %{p1} zam, ardından %{p2} indirim yapılıyor. "
         f"Son fiyat kaç TL olur? Başlangıca göre net değişim yüzde kaçtır?")
    net = (final - price) / price * 100
    think = (f"Ardışık yüzde değişimleri toplanmaz, çarpılır. Zam çarpanı {1 + p1/100:.2f}, "
             f"indirim çarpanı {1 - p2/100:.2f}. {price} × {1 + p1/100:.2f} = {after1:g}; "
             f"{after1:g} × {1 - p2/100:.2f} = {final:g}. Yaygın hata %{p1}-%{p2}=%{p1-p2} demek olur.")
    plan = ["Zam sonrası fiyatı hesapla.", "İndirimi yeni fiyat üzerinden uygula.",
            "Net yüzde değişimi başlangıç fiyatına göre bul."]
    verify = (f"Toplam çarpan {(1 + p1/100) * (1 - p2/100):.4f}; {price} × bu çarpan = {final:g}. "
              f"Net değişim ({final:g} - {price}) / {price} = %{net:.2f}. Tutarlı.")
    out = (f"Zam sonrası fiyat {after1:g} TL, indirim sonrası **{final:g} TL** olur. "
           f"Net değişim **%{net:.2f}**'dir ({'artış' if net >= 0 else 'azalış'}). "
           f"Not: Yüzdeler doğrudan toplanıp çıkarılamaz.")
    return q, build_response(think, plan, verify, out), "orta"


def math_combinatorics(r: random.Random):
    n, k = r.randint(5, 12), r.randint(2, 4)
    comb, perm = math.comb(n, k), math.perm(n, k)
    q = (f"{n} kişilik bir takımdan {k} kişilik bir komite seçilecek. Kaç farklı komite "
         f"oluşturulabilir? Görevler (başkan, yardımcı, ...) farklı olsaydı sonuç ne olurdu?")
    think = (f"Komitede sıra önemli değil → kombinasyon C({n},{k}). Görevler farklıysa sıra "
             f"önemlidir → permütasyon P({n},{k}). C = {n}!/({k}!·{n-k}!) = {comb}, P = {n}!/{n-k}! = {perm}.")
    plan = ["Sıranın önemli olup olmadığını belirle.", "Kombinasyon formülünü uygula.",
            "Permütasyon formülünü uygula.", "P = C × k! ilişkisiyle sağla."]
    verify = f"P({n},{k}) = C({n},{k}) × {k}! = {comb} × {math.factorial(k)} = {comb * math.factorial(k)} = {perm}. Doğru."
    out = f"Görevsiz komite: **C({n},{k}) = {comb}**. Görevler farklıysa: **P({n},{k}) = {perm}**."
    return q, build_response(think, plan, verify, out), "orta"


def math_motion(r: random.Random):
    v1, v2 = r.choice(range(40, 101, 10)), r.choice(range(50, 121, 10))
    t = r.choice([1, 1.5, 2, 2.5, 3, 4])
    d = int((v1 + v2) * t)
    q = (f"Aralarında {d} km bulunan iki şehirden iki araç aynı anda birbirine doğru hareket ediyor. "
         f"Hızları {v1} km/sa ve {v2} km/sa. Kaç saat sonra karşılaşırlar?")
    think = (f"Birbirine doğru hareket → hızlar toplanır: {v1}+{v2}={v1+v2} km/sa. "
             f"Süre = yol / bağıl hız = {d}/{v1+v2} = {t:g} saat.")
    plan = ["Bağıl hızı bul.", "t = x / v formülünü uygula.", "Her aracın aldığı yolu toplayarak sağla."]
    verify = f"{v1}×{t:g} = {v1*t:g} km ve {v2}×{t:g} = {v2*t:g} km; toplam {v1*t + v2*t:g} km = {d} km. Doğru."
    out = f"Araçlar **{t:g} saat** sonra karşılaşır."
    return q, build_response(think, plan, verify, out), "kolay"


# ---------------------------------------------------------------------------
# MANTIK
# ---------------------------------------------------------------------------
NAMES = ["Ayşe", "Mehmet", "Zeynep", "Can", "Elif", "Burak", "Deniz", "Kaan", "Selin", "Emre"]


def logic_ordering(r: random.Random):
    people = r.sample(NAMES, 4)
    ages = sorted(r.sample(range(18, 60), 4), reverse=True)  # people[0] en yaşlı
    clues = [f"{people[0]}, {people[1]}'den büyüktür.",
             f"{people[2]}, {people[1]}'den küçüktür.",
             f"{people[3]} en genç kişidir.",]
    r.shuffle(clues)
    q = ("Dört arkadaş hakkında şunlar biliniyor:\n- " + "\n- ".join(clues) +
         "\nEn yaşlıdan en gence doğru sıralama nedir?")
    order = " > ".join(people)
    think = (f"{people[3]} en genç, en sonda. Kalan üç kişi: {people[0]} > {people[1]} ve "
             f"{people[1]} > {people[2]}. Geçişlilikten {people[0]} > {people[1]} > {people[2]}. "
             f"{people[3]} hepsinden küçük olduğuna göre sıra: {order}.")
    plan = ["Kesin konum veren ipucunu (en genç) yerleştir.",
            "Karşılaştırmalı ipuçlarını zincirle (geçişlilik).", "Her ipucunu sıralamaya karşı test et."]
    verify = "; ".join(f"'{c}' → sağlanıyor" for c in clues) + ". Çelişki yok, sıralama tektir."
    out = f"Sıralama (en yaşlıdan en gence): **{order}**."
    return q, build_response(think, plan, verify, out), "orta"


def logic_knights(r: random.Random):
    a, b = r.sample(NAMES, 2)
    q = (f"Bir adada doğrucular hep doğru, yalancılar hep yalan söyler. {a} şöyle diyor: "
         f"\"{b} ve ben farklı türdeniz.\" {b} ise: \"{a} bir yalancıdır.\" diyor. Kim nedir?")
    think = (f"Durumları deneyelim. {a} doğrucuysa ifadesi doğru → {b} yalancı. Ama o zaman {b}'nin "
             f"'{a} yalancı' ifadesi yalan olmalı → {a} doğrucu. Tutarlı. {a} yalancıysa ifadesi yanlış "
             f"→ ikisi aynı tür → {b} de yalancı. O zaman {b}'nin '{a} yalancı' ifadesi yalan olmalı, "
             f"ama {a} gerçekten yalancı, yani ifade doğru olur → çelişki.")
    plan = ["Olası 4 durumu belirle.", f"{a} doğrucu varsayımını test et.",
            f"{a} yalancı varsayımını test et.", "Çelişkisiz tek durumu seç."]
    verify = (f"({a}=D, {b}=Y): {a}'nın ifadesi doğru ✓, {b}'nin ifadesi yanlış ✓. "
              f"({a}=Y, {b}=Y) çelişkili; ({a}=Y,{b}=D) {a}'nın ifadesini doğru yapar ✗; "
              f"({a}=D,{b}=D) {a}'nın ifadesini yanlış yapar ✗. Tek çözüm var.")
    out = f"**{a} doğrucu, {b} yalancıdır.**"
    return q, build_response(think, plan, verify, out), "zor"


def logic_sequence(r: random.Random):
    kind = r.choice(["arith", "geom", "square"])
    if kind == "arith":
        s, d = r.randint(1, 20), r.randint(2, 9)
        seq = [s + i * d for i in range(6)]
        rule = f"her terim bir öncekinden {d} fazla"
    elif kind == "geom":
        s, k = r.randint(1, 5), r.randint(2, 3)
        seq = [s * k ** i for i in range(6)]
        rule = f"her terim bir öncekinin {k} katı"
    else:
        o = r.randint(0, 5)
        seq = [(i + o) ** 2 + 1 for i in range(1, 7)]
        rule = "n² + 1 biçiminde (ardışık kareler + 1)"
    shown, ans = seq[:5], seq[5]
    q = f"{', '.join(map(str, shown))}, ? — dizideki bir sonraki sayı nedir?"
    diffs = [shown[i + 1] - shown[i] for i in range(4)]
    think = f"Farklara bakayım: {diffs}. Kural: {rule}. Buna göre sonraki terim {ans}."
    plan = ["Ardışık farkları/oranları incele.", "Kuralı genelle.", "Kuralı tüm terimlerde test et."]
    verify = f"Kural ilk 5 terimin hepsini üretiyor ({shown}); 6. terim {ans}."
    out = f"Sonraki sayı **{ans}**. Kural: {rule}."
    return q, build_response(think, plan, verify, out), "kolay"


# ---------------------------------------------------------------------------
# KODLAMA (çözümler gerçekten çalıştırılarak doğrulanır)
# ---------------------------------------------------------------------------
CODE_TASKS = [
    dict(q="Bir metnin palindrom olup olmadığını (boşluk ve büyük/küçük harf yok sayılarak) kontrol eden Python fonksiyonu yaz.",
         fn="palindrom_mu",
         code='''def palindrom_mu(metin: str) -> bool:
    temiz = "".join(ch.lower() for ch in metin if ch.isalnum())
    return temiz == temiz[::-1]''',
         tests=[("'Ey Edip Adanada pide ye'", True), ("'GökTürk'", False), ("''", True)],
         idea="Metni normalize edip (yalnızca harf/rakam, küçük harf) tersiyle karşılaştırmak O(n) çözümdür."),
    dict(q="n'inci Fibonacci sayısını O(n) zamanda ve O(1) bellekle hesaplayan Python fonksiyonu yaz.",
         fn="fibonacci",
         code='''def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n negatif olamaz")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a''',
         tests=[("0", 0), ("1", 1), ("10", 55), ("30", 832040)],
         idea="Özyinelemeli çözüm üsteldir; iki değişkenle iteratif ilerlemek O(n) zaman, O(1) bellek verir."),
    dict(q="Verilen bir sayının asal olup olmadığını verimli şekilde kontrol eden Python fonksiyonu yaz.",
         fn="asal_mi",
         code='''def asal_mi(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True''',
         tests=[("1", False), ("2", True), ("97", True), ("91", False), ("7919", True)],
         idea="Bir bileşik sayının √n'den küçük bir böleni vardır; yalnızca tek sayıları √n'e kadar denemek yeterli."),
    dict(q="Bir metindeki kelimelerin frekansını hesaplayıp en sık geçen k kelimeyi döndüren Python fonksiyonu yaz.",
         fn="en_sik_kelimeler",
         code='''from collections import Counter
import re

def en_sik_kelimeler(metin: str, k: int) -> list[tuple[str, int]]:
    kelimeler = re.findall(r"\\w+", metin.lower())
    return Counter(kelimeler).most_common(k)''',
         tests=[("'a b a c b a', 2", [("a", 3), ("b", 2)]), ("'', 3", [])],
         idea="Regex ile Unicode kelimeleri ayırıp collections.Counter.most_common kullanmak hem okunur hem verimli."),
    dict(q="Sıralı bir listede ikili arama (binary search) yapan, bulamazsa -1 döndüren Python fonksiyonu yaz.",
         fn="ikili_arama",
         code='''def ikili_arama(dizi: list[int], hedef: int) -> int:
    lo, hi = 0, len(dizi) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if dizi[mid] == hedef:
            return mid
        if dizi[mid] < hedef:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1''',
         tests=[("[1, 3, 5, 7, 9], 7", 3), ("[1, 3, 5], 4", -1), ("[], 1", -1)],
         idea="Arama aralığını her adımda yarıya indirerek O(log n) karmaşıklık elde edilir; lo<=hi koşulu sınır hatasını önler."),
    dict(q="İç içe geçmiş parantezlerin ( (), [], {} ) dengeli olup olmadığını kontrol eden Python fonksiyonu yaz.",
         fn="dengeli_mi",
         code='''def dengeli_mi(s: str) -> bool:
    eslesme = {")": "(", "]": "[", "}": "{"}
    yigin = []
    for ch in s:
        if ch in "([{":
            yigin.append(ch)
        elif ch in eslesme:
            if not yigin or yigin.pop() != eslesme[ch]:
                return False
    return not yigin''',
         tests=[("'([]{})'", True), ("'([)]'", False), ("'(('", False)],
         idea="Açılışları yığına it, kapanışta tepedekiyle eşleştir; sonda yığın boş olmalı."),
]


def _run_tests(code: str, fn: str, tests) -> list[str]:
    ns: dict = {}
    exec(code, ns)  # noqa: S102 - yalnızca yukarıdaki sabit, güvenilir kod
    lines = []
    for args, expected in tests:
        got = eval(f"{fn}({args})", ns)  # noqa: S307
        assert got == expected, f"{fn}({args}) = {got!r}, beklenen {expected!r}"
        lines.append(f"{fn}({args}) → {got!r} ✓")
    return lines


def code_task(r: random.Random):
    t = r.choice(CODE_TASKS)
    results = _run_tests(t["code"], t["fn"], t["tests"])
    think = (f"İstenen: {t['q']} Temel fikir: {t['idea']} Kenar durumlarını (boş girdi, "
             f"sınır değerler) da düşünmeliyim.")
    plan = ["Girdi/çıktı sözleşmesini ve kenar durumlarını belirle.", "Algoritmayı seç ve karmaşıklığını değerlendir.",
            "Kodu tip ipuçlarıyla yaz.", "Test senaryolarıyla doğrula."]
    verify = "Test sonuçları:\n" + "\n".join(results)
    out = f"```python\n{t['code']}\n```\n\n**Açıklama:** {t['idea']}"
    return t["q"], build_response(think, plan, verify, out), "orta"


# ---------------------------------------------------------------------------
# SİBER GÜVENLİK (savunma odaklı)
# ---------------------------------------------------------------------------
CYBER_STATIC = [
    dict(q="Aşağıdaki Python kodunda hangi güvenlik açığı var ve nasıl düzeltilir?\n```python\ncur.execute(f\"SELECT * FROM users WHERE name = '{ad}'\")\n```",
         think="Kullanıcı girdisi doğrudan SQL metnine f-string ile gömülüyor. ad = \"' OR '1'='1\" verilirse WHERE koşulu her zaman doğru olur: klasik SQL Injection (CWE-89). Çözüm parametreli sorgu.",
         plan=["Girdinin sorguya nasıl girdiğini incele.", "Saldırı vektörünü örnekle.", "Parametreli sorgu ile düzelt.", "Ek savunmaları listele."],
         verify="Parametreli sorguda sürücü değeri veri olarak bağlar; ' OR '1'='1 artık literal bir isim olarak aranır, sorgu yapısı değişmez. Düzeltme açığı kapatıyor.",
         out="**Açık:** SQL Injection (CWE-89).\n\n**Düzeltme:**\n```python\ncur.execute(\"SELECT * FROM users WHERE name = %s\", (ad,))\n```\nEk önlemler: en az yetkili DB kullanıcısı, ORM kullanımı, girdi doğrulama ve WAF."),
    dict(q="Kullanıcı parolalarını veritabanında nasıl saklamalıyım? MD5 yeterli mi?",
         think="MD5 hızlı bir özet fonksiyonu; GPU ile saniyede milyarlarca deneme yapılabilir, ayrıca tuzsuz kullanımda rainbow table saldırısına açık. Parola için yavaş ve bellek-yoğun KDF gerekir: Argon2id (OWASP önerisi), bcrypt veya scrypt.",
         plan=["MD5'in neden uygunsuz olduğunu açıkla.", "Önerilen algoritmaları sırala.", "Örnek kod ver.", "Operasyonel önerileri ekle."],
         verify="OWASP Password Storage Cheat Sheet Argon2id'yi birinci tercih olarak önerir; bcrypt yaygın ve kabul görür. Örnek kod argon2-cffi API'siyle uyumlu.",
         out="**Hayır, MD5 kesinlikle yeterli değil.** Parolaları **Argon2id** (veya bcrypt) ile tuzlayarak özetleyin:\n```python\nfrom argon2 import PasswordHasher\nph = PasswordHasher()\nh = ph.hash(parola)\nph.verify(h, girilen_parola)\n```\nAyrıca: benzersiz tuz (kütüphane otomatik ekler), gerekirse pepper, giriş denemelerine hız sınırı ve MFA."),
    dict(q="Stored XSS nedir ve bir web uygulamasında nasıl önlenir?",
         think="Stored XSS, saldırganın zararlı betiği sunucuya kalıcı kaydettiği (ör. yorum alanı) ve diğer kullanıcıların tarayıcısında çalıştığı açıktır. Önlem: bağlama uygun çıktı kodlama, CSP, HttpOnly çerezler, HTML sanitizasyonu.",
         plan=["Tanımı ve reflected XSS'ten farkını ver.", "Saldırı akışını özetle.", "Katmanlı savunmaları listele."],
         verify="Önlemler OWASP XSS Prevention Cheat Sheet ile uyumlu: output encoding temel savunma, CSP ikincil katman, HttpOnly çerez çalınmasını zorlaştırır.",
         out="**Stored XSS**, zararlı JavaScript'in sunucuda kalıcı saklanıp sayfayı görüntüleyen herkesin tarayıcısında çalışmasıdır.\n\n**Önleme:**\n1. Çıktıyı bağlama göre kodlayın (HTML, attribute, JS).\n2. Zengin metinde DOMPurify gibi sanitizer kullanın.\n3. Sıkı bir Content-Security-Policy tanımlayın.\n4. Oturum çerezlerini HttpOnly, Secure, SameSite yapın.\n5. Otomatik kaçışlı şablon motorları (Jinja2, React) kullanın."),
    dict(q="Bir sunucuda 22, 80, 443 ve 3389 portları açık görünüyor. Bunlar hangi servislerdir ve hangi sıkılaştırmaları önerirsin?",
         think="22 SSH, 80 HTTP, 443 HTTPS, 3389 RDP. RDP'nin internete açık olması ciddi risk (brute-force, BlueKeep gibi açıklar). SSH için anahtar tabanlı kimlik doğrulama, HTTP'yi HTTPS'e yönlendirme.",
         plan=["Portları servislere eşle.", "Her birinin risklerini değerlendir.", "Önceliklendirilmiş sıkılaştırma listesi ver."],
         verify="Port atamaları IANA standart atamalarıyla uyumlu. En kritik öneri (RDP'yi internetten kapatmak) doğru önceliklendirilmiş.",
         out="- **22/SSH:** Parola girişini kapatın, anahtar kullanın, fail2ban ekleyin.\n- **80/HTTP:** Tüm trafiği 443'e yönlendirin, HSTS açın.\n- **443/HTTPS:** TLS 1.2+/1.3, güçlü şifre takımları.\n- **3389/RDP:** ⚠️ En kritik risk — internete kapatın, yalnızca VPN/bastion üzerinden NLA ile erişim verin."),
]

CYBER_SEARCH = [
    dict(q="{p} için en son yayımlanan kritik CVE'ler hangileri? Sistemimi güncellemem gerekiyor mu?",
         items=["OpenSSH", "Apache HTTP Server", "nginx", "OpenSSL", "Linux çekirdeği", "Microsoft Exchange"],
         query="{p} kritik CVE {year} güvenlik güncellemesi",
         think="Kullanıcı en güncel CVE'leri soruyor. Güvenlik açıkları sürekli yayımlanıyor ve eğitim verim belirli bir tarihte kesiliyor; ezbere liste verirsem eksik veya yanlış olabilir. Web araması yapmalıyım."),
    dict(q="{p} güncel kararlı sürümü nedir?",
         items=["Python", "Node.js", "PostgreSQL", "Kubernetes", "Rust", "Django"],
         query="{p} latest stable release",
         think="Sürüm numaraları sık değişir, bilgim güncel olmayabilir. Doğru cevap için arama gerekir."),
]


def cyber_static(r: random.Random):
    t = r.choice(CYBER_STATIC)
    return t["q"], build_response(t["think"], t["plan"], t["verify"], t["out"]), "orta"


def search_needed(r: random.Random):
    t = r.choice(CYBER_SEARCH)
    p = r.choice(t["items"])
    year = r.choice([2025, 2026])
    q = t["q"].format(p=p)
    sq = t["query"].format(p=p, year=year)
    plan = ["Bilginin zamana duyarlı olduğunu tespit et.", "Kısa ve hedefli bir arama sorgusu oluştur.",
            "Arama sonuçlarını resmi kaynaklarla (NVD, vendor duyuruları) doğrula.", "Özet ve eylem önerisi sun."]
    verify = ("Bu soruya ezberden kesin yanıt vermek hatalı olabilir; yanıtı arama sonuçlarına dayandırmalıyım. "
              "Arama sonuçları gelene kadar yalnızca genel ve her zaman geçerli önerileri verebilirim.")
    out = (f"{p} hakkında güncel bilgiyi doğrulamak için web araması yapıyorum. Genel öneri: resmi güvenlik "
           f"duyurularını (vendor advisory, NVD) takip edin, paket yöneticinizle en son yamaları uygulayın ve "
           f"kullandığınız sürümü `--version` ile kontrol edin. Arama sonuçları geldiğinde ayrıntılı özet sunacağım.")
    return q, build_response(t["think"], plan, verify, out, search_query=sq), "orta"


GENERATORS = {
    "matematik": [math_linear, math_percent, math_combinatorics, math_motion],
    "mantik": [logic_ordering, logic_knights, logic_sequence],
    "kodlama": [code_task],
    "siber_guvenlik": [cyber_static, search_needed],
}


def run_synthetic(n: int, seed: int, out_path: Path):
    r = random.Random(seed)
    seen, records = set(), []
    active = {c: 0 for c in GENERATORS}  # kategori -> art arda tekrar sayısı
    counts = {c: 0 for c in GENERATORS}
    while len(records) < n and active:
        # dengeli dağılım: en az örneği olan aktif kategoriyi seç
        cat = min(active, key=lambda c: counts[c])
        q, resp, diff = r.choice(GENERATORS[cat])(r)
        h = hashlib.md5((q + resp).encode()).hexdigest()
        if h in seen or not is_valid(resp):
            active[cat] += 1
            if active[cat] > 300:  # bu kategorinin benzersiz havuzu tükendi
                del active[cat]
            continue
        active[cat] = 0
        seen.add(h)
        counts[cat] += 1
        records.append(to_record(q, resp, cat, "synthetic", diff))
    write_jsonl(records, out_path)
    if len(records) < n:
        print(f"[uyarı] Benzersiz örnek havuzu tükendi: {len(records)}/{n}", file=sys.stderr)


# ---------------------------------------------------------------------------
# DISTILL: DeepSeek-R1 (mantık) + Qwen-Max (Türkçe biçimlendirme)
# ---------------------------------------------------------------------------
SEED_TOPICS = {
    "matematik": ["olasılık", "sayı teorisi", "geometri", "cebirsel denklemler", "yüzde ve oran problemleri", "limit ve türev"],
    "mantik": ["doğrucu-yalancı bulmacaları", "sıralama bulmacaları", "önermeler mantığı", "kıyas (syllogism)", "Einstein tipi bulmacalar"],
    "kodlama": ["Python algoritmaları", "veri yapıları", "SQL sorguları", "hata ayıklama", "asenkron programlama", "Rust sahiplik modeli"],
    "siber_guvenlik": ["OWASP Top 10 (savunma)", "güvenli kod incelemesi", "kriptografi temelleri", "log analizi ve olay müdahalesi", "ağ sıkılaştırma", "güncel CVE takibi"],
}

QUESTION_GEN_PROMPT = """Türkçe bir yapay zekâ eğitim veri seti için {cat} alanında, "{topic}" konusunda,
{diff} zorlukta, özgün ve tek bir kullanıcı sorusu yaz. Soru net ve çözülebilir olsun.
Siber güvenlikte yalnızca savunma/eğitim amaçlı sorular yaz. {search_hint}
YALNIZCA soruyu yaz, başka hiçbir şey ekleme."""

FORMAT_PROMPT = """Aşağıda bir kullanıcı sorusu ve bir akıl yürütme modelinin (DeepSeek-R1) ham düşünce adımları
ile cevabı var. Bunu kusursuz, akıcı ve doğal TÜRKÇE ile şu 5 etiketli formata dönüştür:

<think>
(ham düşünceyi Türkçe, öz ve mantıksal akışı koruyarak yeniden yaz)
</think>
<plan>
1. ...
2. ...
</plan>
<search_query>
(güncel/zamana duyarlı bilgi gerekiyorsa kısa bir web arama sorgusu, gerekmiyorsa yalnızca: YOK)
</search_query>
<verify>
(sonucu bağımsız bir yolla sağla, hataları düzelt)
</verify>
<output>
(kullanıcıya yönelik nihai, net Türkçe yanıt; gerekiyorsa markdown/kod bloğu)
</output>

Etiketlerin dışında HİÇBİR metin yazma. Ham akıl yürütmede hata varsa <verify> içinde yakala ve düzelt.

SORU:
{question}

HAM DÜŞÜNCE (R1):
{reasoning}

R1 CEVABI:
{answer}"""


def _client(base_url: str, key_env: str):
    from openai import OpenAI  # pip install openai
    key = os.environ.get(key_env)
    if not key:
        sys.exit(f"[hata] {key_env} ortam değişkeni tanımlı değil.")
    return OpenAI(base_url=base_url, api_key=key)


def distill_one(args, reasoner, formatter, r: random.Random):
    cat = r.choice(list(SEED_TOPICS))
    topic = r.choice(SEED_TOPICS[cat])
    diff = r.choice(["kolay", "orta", "zor"])
    search_hint = ("Soru, güncel/zamana duyarlı bilgi gerektirsin." if r.random() < args.search_ratio
                   else "Soru, güncel bilgi gerektirmeyen kalıcı bilgiye dayansın.")
    # 1) Soru üretimi (Qwen-Max)
    question = formatter.chat.completions.create(
        model=args.formatter_model, temperature=1.0,
        messages=[{"role": "user", "content": QUESTION_GEN_PROMPT.format(
            cat=cat, topic=topic, diff=diff, search_hint=search_hint)}],
    ).choices[0].message.content.strip()
    # 2) Mantık adımları (DeepSeek-R1)
    rr = reasoner.chat.completions.create(
        model=args.reasoner_model, messages=[{"role": "user", "content": question}])
    msg = rr.choices[0].message
    reasoning = getattr(msg, "reasoning_content", None) or ""
    answer = msg.content or ""
    if not reasoning and "<think>" in answer:  # açık kaynak R1 sunucuları
        reasoning, _, answer = answer.partition("</think>")
        reasoning = reasoning.replace("<think>", "")
    # 3) Türkçe 5-aşamalı biçimlendirme (Qwen-Max)
    for _ in range(args.retries):
        resp = formatter.chat.completions.create(
            model=args.formatter_model, temperature=0.3,
            messages=[{"role": "user", "content": FORMAT_PROMPT.format(
                question=question, reasoning=reasoning[:12000], answer=answer[:4000])}],
        ).choices[0].message.content.strip()
        resp = resp.removeprefix("```").removesuffix("```").strip()
        parsed = parse_response(resp)
        if parsed:
            return to_record(question, build_response(
                parsed["think"], parsed["plan"], parsed["verify"], parsed["output"],
                parsed["search_query"]), cat, "distill:r1+qwen-max", diff)
    return None


def run_distill(args):
    reasoner = _client(args.reasoner_base_url, args.reasoner_key_env)
    formatter = _client(args.formatter_base_url, args.formatter_key_env)
    records, seen = [], set()
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(distill_one, args, reasoner, formatter, random.Random(args.seed + i))
                for i in range(args.n)]
        for i, f in enumerate(as_completed(futs), 1):
            try:
                rec = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"[hata] {e}", file=sys.stderr)
                continue
            if rec:
                h = hashlib.md5(rec["messages"][1]["content"].encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    records.append(rec)
            if i % 10 == 0:
                print(f"  {i}/{args.n} işlendi, {len(records)} geçerli", file=sys.stderr)
                write_jsonl(records, args.output, quiet=True)  # ara kayıt
    write_jsonl(records, args.output)


def write_jsonl(records, path: Path, quiet=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if not quiet:
        from collections import Counter
        c = Counter(r["meta"]["category"] for r in records)
        s = sum(r["meta"]["needs_search"] for r in records)
        print(f"✓ {len(records)} örnek → {path}  | kategoriler: {dict(c)} | arama gerektiren: {s}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("-n", type=int, default=2000)
    s.add_argument("-o", "--output", type=Path, default=Path("data/gokturk_synth.jsonl"))
    s.add_argument("--seed", type=int, default=42)
    d = sub.add_parser("distill")
    d.add_argument("-n", type=int, default=100)
    d.add_argument("-o", "--output", type=Path, default=Path("data/gokturk_distill.jsonl"))
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--retries", type=int, default=2)
    d.add_argument("--search-ratio", type=float, default=0.15)
    d.add_argument("--reasoner-base-url", default="https://api.deepseek.com")
    d.add_argument("--reasoner-model", default="deepseek-reasoner")
    d.add_argument("--reasoner-key-env", default="DEEPSEEK_API_KEY")
    d.add_argument("--formatter-base-url", default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
    d.add_argument("--formatter-model", default="qwen-max")
    d.add_argument("--formatter-key-env", default="DASHSCOPE_API_KEY")
    args = ap.parse_args()
    if args.mode == "synthetic":
        run_synthetic(args.n, args.seed, args.output)
    else:
        run_distill(args)


if __name__ == "__main__":
    main()
