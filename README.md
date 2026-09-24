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
