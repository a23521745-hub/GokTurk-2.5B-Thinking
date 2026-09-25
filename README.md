# GokTurk-2.5B-Thinking
Lightweight, local-executable 2.5B Turkish LLM with multi-stage thinking (CoT) &amp; search capabilities.

## Yanıt formatı (5 aşama)
```
<think>iç akıl yürütme</think>
<plan>1. ... 2. ...</plan>
<search_query>web sorgusu | YOK</search_query>
<verify>sağlama / öz-denetim</verify>
<output>nihai Türkçe yanıt</output>
```
Örnek kayıtlar: `data/template.jsonl` (her satır: `messages` [system/user/assistant] + `meta`).

## Veri üretimi
```bash
# API gerektirmez; cevaplar programatik doğrulanır (kod örnekleri çalıştırılır)
python scripts/generate_dataset.py synthetic -n 3000 -o data/gokturk_synth.jsonl

# Hibrit damıtma: DeepSeek-R1 (mantık adımları) + Qwen-Max (Türkçe 5-aşama biçim)
pip install openai
export DEEPSEEK_API_KEY=... DASHSCOPE_API_KEY=...
python scripts/generate_dataset.py distill -n 1000 --workers 8 -o data/gokturk_distill.jsonl
```
Yerel R1/Qwen-72B için `--reasoner-base-url`, `--formatter-base-url` ve `--*-model` ile herhangi bir OpenAI uyumlu uç nokta (vLLM, Ollama, OpenRouter) kullanılabilir.

## Eğitim (Colab T4)
`train/GokTurk_Colab_T4.ipynb` → T4 GPU seç → Tümünü çalıştır. Ya da:
```bash
python train/train_unsloth.py --model unsloth/Qwen2.5-3B-Instruct-bnb-4bit --data data/*.jsonl
```
Çıktılar: `outputs/gokturk-2.5b-thinking/{lora,merged_16bit,gguf}`.

## GGUF derleme (GitHub Actions)
`.github/workflows/build-gguf.yml` → **Actions → Build GGUF → Run workflow**
- `source=hf`: HF repo id (merged 16-bit ağırlıklar; özel repo için `HF_TOKEN` secret'ı)
- `source=release`: bu repodaki bir Release'e yüklenmiş `merged_16bit.tar.gz`
- Ya da etiketle: `git tag gguf-v1 && git push origin gguf-v1` (repo değişkeni `GGUF_DEFAULT_HF_REPO`)

llama.cpp ile F16 → Q4_K_M + Q8_0 üretir, duman testi yapar, 2 GiB'ı aşan dosyaları `llama-gguf-split` ile böler ve SHA256SUMS ile birlikte GitHub Releases'e yükler.

## Arama çalıştırıcısı
```bash
llama-server -m GokTurk-2.5B-Thinking-Q4_K_M.gguf -c 4096 --port 8080
python inference/search_executor.py "nginx için son kritik CVE'ler neler?" --show-trace
python inference/search_executor.py -i --search searxng --searxng-url http://localhost:8888
python inference/search_executor.py --search-only "python latest version"   # yalnızca arama testi
```
Model `</search_query>` ürettiğinde üretim durur, sorgu DuckDuckGo/SearXNG'de aranır, ilk 3 sonuç `<search_results>` olarak enjekte edilir ve model `<verify>`/`<output>` ile devam eder. Testler: `python -m unittest discover tests`.

## Eğitim (Kaggle T4)
`train/kaggle/GokTurk_Kaggle_T4.ipynb` (veya `gokturk_kaggle_t4.py`) → Kaggle'a yükleyin:
GPU T4 · Internet On · Secrets'a `HF_TOKEN` → **Save Version → Save & Run All**.
Eğitim → Q4_K_M GGUF → `<kullanıcı>/GokTurk-2.5B-Thinking-GGUF` HF deposuna otomatik yükleme.
