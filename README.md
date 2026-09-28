# 🐺 GökTürk2.5-3B-Thinking
Hafif, yerelde çalışan (≤ 2.5 GB RAM, Q4_K_M), **6 aşamalı hibrit Chain-of-Thought** ve **Tavily tool calling / RAG** yeteneğine sahip Türkçe LLM.

| | |
|---|---|
| Taban model | Qwen2.5-3B-Instruct (varsayılan) · Qwen2.5-1.5B-Instruct (daha hafif) |
| Eğitim | Unsloth QLoRA (r=16, α=16, 7 projeksiyon katmanı), Kaggle/Colab T4 |
| Dağıtım | GGUF Q4_K_M (≈1.9 GB / 3B, ≈1.0 GB / 1.5B) · Q8_0 |
| Arama | Tavily API (yedek: DuckDuckGo / SearXNG) |

> Qwen2.5 ailesinde 2.5B boyutunda bir model yoktur; “2.5B” proje adıdır. 3B Q4_K_M, ctx ≤ 4096 ile 2.5 GB bütçesine sığar (tahmini ~2.3 GB). Daha fazla pay için `--size 1.5b` kullanın.

## 6 aşamalı düşünme biçimi
```text
<thought>
[STEP 1: Problem Sentezleme & Ayrıştırma]
[STEP 2: İçsel Akıl Yürütme]            … Dış kaynak: GEREKLİ | GEREKSİZ
[STEP 3: Web Araştırması & Kaynak Toplama]
<tool_call>{"name": "tavily_search", "arguments": {"query": "…"}}</tool_call>   ← yalnızca gerekirse
   ── <tool_response>{temizlenmiş Tavily JSON}</tool_response> ──
[STEP 4: Kaynak Doğrulama & Yerelde İşleme]
[STEP 5: Yanıt Planlama & Sunum Hazırlığı]
[STEP 6: Öz-Denetim & Eksiklik Giderme]
</thought>

Nihai yanıt (kaynaklar [1], [2] …)
```
Araç çağrısı Qwen2.5'in yerel `<tool_call>` biçimidir. Biçimin tek kaynağı `gokturk_cot.py` dosyasıdır; veri, eğitim, GGUF sohbet şablonu ve RAG istemcisi bu dosyayı kullanır. Örnekler: `data/template6.jsonl`.

## Dosyalar
| Dosya | Görev |
|---|---|
| `gokturk_cot.py` | Format, sistem promptu, araç şeması, ChatML render, doğrulayıcı, GGUF Jinja şablonu |
| `prepare_dataset.py` | `synthetic` · `seeds` (GSM8K, MATH, OpenR1/DeepSeek-R1, UltraFeedback, ShareGPT) · `distill` (Gemini + gerçek Tavily) · `merge` |
| `train_unsloth.py` | Unsloth QLoRA eğitimi (Kaggle/Colab/yerel otomatik algılama) |
| `push_to_hf.py` | LoRA → merged 16-bit (CPU) → GGUF Q4_K_M/Q8_0 (llama.cpp) → RAM tablosu → Hugging Face |
| `tavily_rag_handler.py` | Tavily istemcisi, sonuç temizleme ve süzme, ajan döngüsü, CLI |
| `gokturk_env.py` | Kaggle/Colab secrets, GPU kontrolü, sürüm kilitli kurulum, yeniden deneme (retry/backoff) |
| `requirements-train.txt` + `constraints.txt` | Eğitim bağımlılıkları; `datasets<4.4.0` (unsloth-zoo), `huggingface_hub<1.0` kilitli |
| `requirements.txt` | Veri hazırlama / RAG istemcisi (GPU gerekmez) |
| `notebooks/GokTurk6_Kaggle_T4.ipynb` | Kaggle'da uçtan uca çalışan notebook |

## Hızlı başlangıç
```bash
# 1) Veri
python prepare_dataset.py synthetic -o data/synth6.jsonl                    # API gerekmez
pip install datasets openai
python prepare_dataset.py seeds --source gsm8k:800 openr1:800 ultrafeedback:500 -o data/seeds.jsonl
export GEMINI_API_KEY=... TAVILY_API_KEY=tvly-...
python prepare_dataset.py distill --seeds data/seeds.jsonl --topics 600 -o data/distill6.jsonl
python prepare_dataset.py merge data/synth6.jsonl data/distill6.jsonl -o data/gokturk6

# 2) Eğitim (T4) — LoRA eğitim biter bitmez HF'e yüklenir
pip install -r requirements-train.txt -c constraints.txt
export HF_TOKEN=hf_...  HF_REPO=kullanici/GokTurk2.5-3B-Thinking   # HF_REPO isteğe bağlı
python train_unsloth.py --size 3b --epochs 2 --push-lora

# 3) Merge (CPU) → GGUF → Hugging Face (başarısız olursa yalnızca bu adımı tekrarlayın)
python push_to_hf.py --quants q4_k_m q8_0 --push

# 4) Yerelde çalıştırma + Tavily RAG
llama-server -m outputs/gokturk/gguf/GokTurk2.5-3B-Thinking-Q4_K_M.gguf -c 4096 -ctk q8_0 -ctv q8_0 -fa on --port 8080
python tavily_rag_handler.py "OpenSSH için son kritik CVE hangisi?" --show-thought
```

Hugging Face'te oluşan depolar: `<kullanıcı>/GokTurk2.5-3B-Thinking` (merged 16-bit) · `…-GGUF` (Q4_K_M, Q8_0) · `…-LoRA`.

### GGUF'ta 6 aşama yok, model kendini "Qwen" sanıyor mu?
28 Eylül'den önce üretilen GGUF'lara yanlışlıkla orijinal Qwen sohbet şablonu gömüldü. Bu şablon "You are Qwen, created by Alibaba Cloud" ekler. Model ağırlıkları sağlamdır; yalnızca şablonu değiştirmek yeterli, yaklaşık 1 dakika sürer:
```bash
pip install gguf
python fix_gguf_template.py GokTurk2.5-3B-Thinking-Q4_K_M.gguf   # → ...-fixed.gguf
```
PocketPal'da **Sistem İstemi (system prompt) boş** olmalı. Şablon, eğitimdeki GökTürk sistem promptunu otomatik ekler.

### Eğitim bitti, yalnızca GGUF üretmek
**En kolay yol:** `notebooks/GokTurk_GGUF_Export.ipynb` dosyası tek başına çalışır, repoya bağımlı değildir. Kaggle'a yükleyin, Accelerator None ve Internet On seçin, `HF_TOKEN` secret'ını bağlayın, Save & Run All yapın.
Notebook LoRA'yı indirir, 16-bit tabanla birleştirir, Q4_K_M ve Q8_0 GGUF üretir, doğrular ve yükler.

Alternatif:
LoRA hazırsa yeniden eğitmeye gerek yok. Birleştirme ve dönüştürme **CPU'da** çalışır:
```bash
python push_to_hf.py --lora-repo kullanici/GokTurk2.5-3B-Thinking-LoRA --base Qwen/Qwen2.5-3B-Instruct --push
```
Kaggle notebook'unda ayarlardaki `LORA_SOURCE` alanını doldurmanız yeterli; Accelerator `None` olabilir.

### Düşünce ↔ yanıt tutarlılığı
STEP 6 her zaman `Doğrulanan sonuç: …` satırıyla biter. Nihai yanıt bu sonuçla çelişen örnekler eğitim verisinden atılır. Çıkarımda `tavily_rag_handler.py` tutarsızlık görürse düşünceyi aynen tutar ve yalnızca nihai yanıtı temperature=0 ile yeniden üretir.

### Kaggle
`notebooks/GokTurk6_Kaggle_T4.ipynb` dosyasını yükleyin. **Accelerator mutlaka `GPU T4 x2` olmalı.** CPU seçiliyse notebook ilk hücrede durup uyarır. **GPU T4 · Internet On · Secrets: `HF_TOKEN`** (isteğe bağlı: `GEMINI_API_KEY`, `TAVILY_API_KEY`). Ardından **Save Version → Save & Run All**.
Kendi verinizi kullanmak için `train.jsonl`/`val.jsonl` dosyalarını Kaggle Dataset olarak ekleyin; notebook bunları otomatik bulur.

### PocketPal AI / LM Studio
GGUF'a GökTürk sohbet şablonu gömülüdür. Sistem promptu boş bırakılırsa 6 adımlı prompt otomatik eklenir. Bu uygulamalar araç çalıştırmaz; model STEP 3'te arama isterse `tool_call` görünür. Canlı arama için `tavily_rag_handler.py` kullanın.

### Diğer
- `.github/workflows/build-gguf.yml`: Hugging Face'teki merged modelden CI ile GGUF üretir ve GitHub Release'e yükler.
- `inference/search_executor.py` (DuckDuckGo/SearXNG yedeği) ve `scripts/` (sentetik üreticiler) önceki sürümden kalma yardımcılardır.
- Testler: `python -m unittest discover tests`
