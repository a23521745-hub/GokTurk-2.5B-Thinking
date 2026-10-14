# GökTürk-3.3B-Thinking: Teacher → Student → GGUF

Bu dizin, soru bankasından öğretmen verisi üretmek, doğrulamak, H200 üzerinde QLoRA/SFT yapmak ve tam bir HF `safetensors` modelini `Q4_K_M` GGUF'e çevirmek için ayrı, tekrarlanabilir bir iş akışı ekler. Mevcut GökTürk-2.5B kodu ve veri biçimi değiştirilmez.

## 1. EVREN ile uyumluluk ve teyit edilen sınırlar

EVREN'in resmi LLM çıkarım sayfası sohbet tamamlama için OpenAI uyumlu API, model seçimi, JSON yanıt ve SSE akışını açıklıyor. Resmi platform sayfası H200 kümesi ve Slurm/Apptainer tabanlı eğitim iş akışını belirtiyor. Bu nedenle bu repo:

- Öğretmen API isteklerinde OpenAI Python SDK'nın OpenAI-uyumlu Chat Completions biçimini kullanır.
- H200 eğitimini `scripts/train_qlora_h200.py` ile çalıştırır; `hpc/evren/train_gokturk_h200.sbatch` ise Slurm için düzenlenebilir örnek şablondur.
- EVREN'in kimlik doğrulaması gerektiren URL'si, model kimlikleri, Slurm partition/account değerleri veya platforma dosya gönderme API'si hakkında tahminde bulunmaz. Bunları EVREN hesabındaki güncel API/iş ayrıntılarından alın.
- Soru bankası/JSONL veri aktarımı için EVREN'de sunulan onaylı veri yükleme yolunu kullanır; burada belgelenmeyen bir platform endpoint'i uydurulmamıştır.

**Veri egemenliği:** varsayılan üç doğrudan öğretmen (Gemini, DeepSeek, Qwen) harici servislerdir. Soru metinleri bu servis sağlayıcılarına gönderilir. Üretici, `--allow-external-teachers` açık onayı olmadan dış çağrı yapmaz. Bu kipte yalnızca açık, sentetik veya kurum politikasınca paylaşılmasına izin verilmiş/sanitize edilmiş soru kullanın; sınıflı, kişisel, ihracat kontrollü veya hassas savunma verisini göndermeyin.

EVREN'in OpenAI uyumlu LLM API'si üzerinden kurum içinde kullanılabilen iki ayrı model varsa (örneğin katalogda görünen bir DeepSeek ve Qwen), `evren_deepseek` ve `evren_qwen` config girişleri kullanılabilir. `EVREN_BASE_URL`, `EVREN_LLM_API_KEY`, `EVREN_DEEPSEEK_MODEL`, `EVREN_QWEN_MODEL` değerlerini yalnızca EVREN portalından alın; iki model ID'sinin gerçekten farklı ve erişilebilir olduğunu doğrulayın. Sonra `configs/distillation.yaml` içinde bu iki girişi `enabled: true` yapıp `--teachers evren_deepseek,evren_qwen` kullanın. Bu yol `--allow-external-teachers` istemez. EVREN model kataloğunda uygun iki model yoksa bu örneği çalıştırmayın.

## 2. API ve model kimliği notları (14 Ekim 2026)

- **Gemini 1.5 Pro artık çalışan bir varsayılan değildir.** Google, `gemini-1.5-pro` modelini Gemini API'de 29 Eylül 2025'te kapattığını duyurdu. Bu yüzden config, eski talebi `legacy_requested_model` olarak kaydeder fakat çalışan varsayılanı `gemini-3.8-flash` yapar. `GEMINI_MODEL` ve `GEMINI_BASE_URL` ile hesapta gerçekten erişilebilir bir model/uyumlu gateway seçilebilir; kapatılmış `gemini-1.5-pro` değerini ayarlamak servisi geri getirmez.
- DeepSeek varsayılanı OpenAI uyumlu `https://api.deepseek.com` ve `deepseek-v4-pro`'dur; model/endpoint zamanla değişebilir, `DEEPSEEK_MODEL` ile güncellenebilir.
- Qwen/DashScope anahtarları ve endpoint'ler bölgeye göre değişir. `QWEN_BASE_URL` boş bırakılmıştır; Alibaba Model Studio konsolunda hesabınıza ait tam OpenAI uyumlu URL'yi (genellikle `/compatible-mode/v1` son ekiyle) girin.
- EVREN sayfasında görülen OpenAI uyumluluğu, Gemini/DeepSeek/Qwen'in dış API anahtarlarını EVREN anahtarına dönüştürmez. EVREN gateway kullanırken kendi EVREN anahtarınızı ve katalogdaki model ID'lerini kullanın.

## 3. Kurulum ve güvenli anahtar yönetimi

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

cp .env.example .env
chmod 600 .env
$EDITOR .env                 # Anahtarları kendi makinenizde girin; bu dosyayı commit etmeyin.
```

`--env-file .env` yalnızca açıkça çağrıldığında okunur ve POSIX sistemlerde grup/diğer kullanıcılarca okunabilir dosyayı reddeder. Süreç ortam değişkenleri önceliklidir. Loglarda anahtar veya prompt basılmaz. API anahtarlarını notebook'a, Git'e, komut satırı argümanına veya bu sohbete koymayın.

## 4. Soru bankası ve JSON sözleşmesi

Bir satır bir JSON nesnesi olacak şekilde JSONL kullanılır. Kategoriler: `matematik`, `mantik`, `turkce_talimat`, `planlama`, `kodlama` (İngilizce eş anlamlılar da kabul edilir). Her satırda `id`, `category`, `question` bulunur. Matematik için doğruluk filtresinin gerçek bir referansı olmalıdır:

```json
{"id":"m-001","category":"matematik","question":"17 + 25 kaçtır?","expected_answer":42,"verifier":{"type":"numeric","tolerance":0}}
```

`verifier.type` değerleri `numeric`, `exact`, `contains`, `consensus`, `none` olabilir. Matematik ifadeleri `verifier.expression` ile verilebilir; Python `eval` kullanılmaz, yalnızca sabit sayılar ve sınırlı aritmetik AST işleçleri kabul edilir. Kod cevaplarında isteğe bağlı `"python_syntax": true` yalnızca `ast.parse` ile sözdizimi kontrolü yapar; modelin ürettiği kod **asla çalıştırılmaz**.

Örnek banka: `data/questions.sample.jsonl`. Bu dosyada matematik/mantık için referans, planlama/Türkçe/kodlama için en az iki öğretmenin normalleştirilmiş aynı kısa cevabı gerekir.

Her öğretmenin üretmesi gereken tek JSON nesnesinin schema'sı `configs/reasoning_steps.schema.json` dosyasındadır:

```json
{"understand":"...","plan":"...","solve":"...","verify":"...","answer":"...","confidence":0.92}
```

Üretici her öğretmenden kısa Türkçe alanlar ister ve `response_format: {"type":"json_object"}` kullanır; ardından bağımsız olarak duplicate-key/NaN/fence/ek alan kontrolleri ve yerel şema kontrolü uygular. Kayıtlar bir JSONL satırında geçerli JSON objesi olarak yazılır. Altı alanın dışında teacher payload'ı kabul edilmez.

## 5. Teacher ensemble, rejection sampling ve verifier

Önce soru bankasını kontrol edin, sonra örnekleri üretin:

```bash
python scripts/generate_teacher_data.py \
  --config configs/distillation.yaml \
  --input data/questions.jsonl \
  --accepted data/teacher.accepted.jsonl \
  --rejected data/teacher.rejected.jsonl \
  --dry-run

python scripts/generate_teacher_data.py \
  --config configs/distillation.yaml \
  --input data/questions.jsonl \
  --accepted data/teacher.accepted.jsonl \
  --rejected data/teacher.rejected.jsonl \
  --env-file .env \
  --allow-external-teachers

python scripts/verify_teacher_data.py \
  --config configs/distillation.yaml \
  --input data/teacher.accepted.jsonl \
  --accepted data/verified.accepted.jsonl \
  --rejected data/verified.rejected.jsonl
```

- `ACCEPTED` yalnızca schema/alan limitleri, güven eşiği, referans kontrolü (varsa) ve bağımsız teacher quorum'u geçince yazılır.
- Referanslı matematik cevaplarında `solve`, `verify` ve `answer` içindeki son sayısal sonuç Python `Decimal` ile beklenen değere karşı kontrol edilir. `numeric` toleransını yalnızca bilerek toleranslı ölçümler için artırın.
- Referanssız planlama/kodlama/doğal dil kayıtlarında en az iki ayrı provider'ın tam normalleştirilmiş cevap fingerprint'i aynı olmalıdır. Bu kasıtlı olarak muhafazakâr bir rejection-sampling kuralıdır; eşleşmeyen ama doğru alternatif yanıtlar `REJECTED` olabilir.
- Öğretmen mutabakatı genel halüsinasyonların kanıtı değildir. Arbitrary doğal dil iddialarını otomatik olarak ispat etmek mümkün değildir; önemli veriler için domain uzmanı örneklemesi ve bağımsız test seti kullanın. `teacher.rejected.jsonl` soru ve denetim adaylarını içerebilir; erişimini de veri politikasıyla koruyun.
- `--resume` daha önce accepted/rejected olarak işlenen ID'leri atlar. Reddedilen örnekleri yeniden denemek için ayrı bir çıktı dosyası/kopya kullanın; aynı ID'yi sessizce tekrar yazmaz.

## 6. H200 QLoRA/SFT

Config: `configs/train_h200.yaml` — **r=64, alpha=128, 4-bit NF4 + double quant, BF16 compute/train, max sequence length 4096**. Eğitim script'i tek bir allocated H200 için tasarlanmıştır (EVREN'in 64 GPU kümesinden bir GPU ayırmak yeterlidir); multi-GPU `torchrun` değildir. `report_to: none`; otomatik W&B/üçüncü taraf telemetry kapalıdır. Loss yalnızca assistant JSON completion üzerinde hesaplanır; user/system prompt maskelenir.

```bash
# Önce EVREN'in sağladığı H200 uyumlu ortamı etkinleştirin; torch/CUDA'yı körlemesine değiştirmeyin.
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
python -m pip install --dry-run --report /tmp/gokturk-h200-resolve.json \
  -r requirements-h200.txt -c constraints.txt
# Çözüm mevcut torch'u değiştirmiyorsa gerçek kurulum:
python -m pip install -r requirements-h200.txt -c constraints.txt
# EVREN portalının istediği partition/account/container seçeneklerini ekleyin.
mkdir -p logs
sbatch hpc/evren/train_gokturk_h200.sbatch
```

Yeni H200 script'i native Transformers/PEFT kullanır; `requirements-h200.txt` yalnızca bu akışın küçük runtime listesidir ve PyTorch/CUDA'yı bilerek pinlemez. `requirements-train.txt` ise eski Kaggle/Colab Unsloth eğitim yolunun daha geniş bağımlılıklarını tutar. 14 Ekim 2026 tarihli resolver dry-run, genel `requirements-train.txt` için Torch 2.12.1 + CUDA 13 paketlerini seçti; bu EVREN'in driver/container sürümüyle doğrulanmış değildir. Kurulum mevcut site Torch'unu değiştirecekse durun; portalın H200 image/module sürümüyle eşleşen bağımlılık setini EVREN yöneticisinden alın. Dry-run çözümlemesi gerçek GPU eğitim/ABI testi değildir.

Container kullanılıyorsa:

```bash
export EVREN_APPTAINER_IMAGE=/portalin-gosterdigi/yol/h200-training.sif
sbatch hpc/evren/train_gokturk_h200.sbatch
```

SBATCH header'ındaki GPU/partition/memory değerlerini EVREN projenizin gerçek politikasıyla eşleyin. `--gres=gpu:1` genel bir Slurm örneğidir; partition/constraint adı tahmin edilmemiştir. Eğitim betiği cihaz adının `H200` olmasını varsayılan olarak zorunlu kılar. Ağırlık lisansı, erişim izni ve tokenizer/chat template uygunluğu eğitim başlamadan önce doğrulanmalıdır.

**3.3B adlandırma uyarısı:** Depodaki mevcut model kodu varsayılan olarak Qwen2.5-3B kullanıyor; bu checkpoint doğrulanmış bir 3.3B parametreli model değildir. YAML'deki `GokTurk-3.3B-Thinking` artifact adı tek başına parametre sayısını değiştirmez. Gerçek 3.3B base ağırlığınız varsa `STUDENT_BASE_MODEL`/`model.base_model_id` değerini ona ayarlayın ve model lisansını kontrol edin. Qwen2.5-3B örneği yalnızca repo ile uyumlu başlangıç noktasıdır.

## 7. Adapter merge ve GGUF Q4_K_M

QLoRA eğitimi bir adapter üretir. Önce adapter'ı base model ile BF16 safetensors full model olarak birleştirin:

```bash
python scripts/merge_lora.py \
  --config configs/train_h200.yaml \
  --adapter outputs/gokturk-3.3b-h200 \
  --output outputs/gokturk-3.3b-merged \
  --device cuda --dtype bfloat16
```

Ardından reviewed llama.cpp tag/commit'ini pinleyip dönüştürün:

```bash
export LLAMA_CPP_REF=<llama.cpp-tag-veya-commit-sha>
python scripts/export_gguf.py \
  --model-dir outputs/gokturk-3.3b-merged \
  --output-dir outputs/gokturk-3.3b-gguf \
  --llama-cpp-ref "$LLAMA_CPP_REF" \
  --install-converter-deps
```

Script `convert_hf_to_gguf.py --outtype f16` ile ara F16 GGUF üretir, `llama-quantize ... Q4_K_M` çalıştırır, açılabilirlik/üretim smoke test'i yapar, SHA-256 ve sürüm manifest'i yazar ve varsayılan olarak ara F16 dosyasını siler. `--keep-f16` ile saklayın. Yaklaşık **1.8–1.95 GB** sadece hedef aralıktır; gerçek boyut mimari, embedding/vocabulary ve tensörlere bağlıdır. `--strict-size` bunu zorunlu hale getirir; varsayılan olarak gerçek boyut raporlanıp aralık dışıysa uyarılır.

`export_gguf.py` S25 FE için 4K context, Q8_0 KV-cache ve runtime rezervi ile yaklaşık RAM hesabı yazar. Bu bir gerçek cihaz benchmark'ı veya uyumluluk garantisi değildir. GGUF'nin yanında Android'de llama.cpp destekleyen bir runtime gerekir. 3.6 GB boş RAM için cihazda model yüklemesi, context/batch ayarları ve diğer süreçler açıkken ölçüm yapın; uygulama çöküyorsa context/batch'i azaltın. Güvenli başlangıç runtime ayarları: context 4096 veya daha düşük, K/V cache `q8_0`, CPU fallback'te 2–4 thread ve küçük batch.

## 8. Resmi teknik kaynaklar

- EVREN LLM çıkarımı ve OpenAI uyumlu Chat Completions: <https://evren.ssyz.org.tr/llm-inference/>
- EVREN platformu, HPC/Slurm/Apptainer ve H200 iş akışı: <https://evren.ssyz.org.tr/explore/>
- Gemini API değişiklikleri (Gemini 1.5 kapanış duyurusu): <https://ai.google.dev/gemini-api/docs/changelog>
- Gemini API güncel model kataloğu: <https://ai.google.dev/gemini-api/docs/models>
- DeepSeek resmi API ilk çağrı / base URL / model kimlikleri: <https://api-docs.deepseek.com/news/news>
- Alibaba Cloud Model Studio OpenAI-uyumlu Qwen Chat API: <https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions>
- Alibaba Cloud Qwen JSON/structured output: <https://www.alibabacloud.com/help/en/model-studio/qwen-structured-output>
- llama.cpp HF → GGUF dönüştürme: <https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py>
