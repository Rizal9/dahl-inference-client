# Dahl Inference Client

Client Python untuk API Dahl Inference (OpenAI-compatible) di
`https://inference.dahl.global/v1` — dengan rotasi key dan fallback model.

## Fitur

- Baca key dari file lokal (satu key per baris), rotasi round-robin.
- Key yang kena 401 (mati) otomatis dilewati dan dicatat.
- Kalau model penuh (`model_concurrency` / 429), otomatis pindah ke model berikut.
- Mode interaktif (REPL) dengan history.

## Cara pakai

1. Buat file key kamu sendiri, misal `keys.txt` (satu key per baris).
2. Arahkan `key_file` di `config.json` ke file itu.
3. Jalankan:

```bash
# Satu prompt
python3 dahl.py "jelaskan black hole dengan bahasa sederhana"

# Model lain
python3 dahl.py -m "zai-org/GLM-5.3-Flash" "tulis haiku tentang hujan"

# Dari stdin / pipe
echo "apa itu fotosintesis?" | python3 dahl.py

# Ngobrol interaktif (REPL)
python3 dahl.py

# Tes semua key, laporkan yang hidup
python3 dahl.py --check
```

## Model default

| Model | Keterangan |
|---|---|
| `MiniMaxAI/MiniMax-M2.7` | Default |
| `zai-org/GLM-5.3-Flash` | Cadangan 1 |
| `deepseek-ai/DeepSeek-V4-Flash-0731` | Cadangan 2 |

## Konfigurasi

Lihat `config.json`: `base_url`, `default_model`, `key_file`,
`max_retries_per_key`, `max_tokens`, `temperature`, `timeout_seconds`.
