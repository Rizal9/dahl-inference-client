#!/usr/bin/env python3
"""Dahl Inference client — OpenAI-compatible, key rotation + model fallback.

Usage:
    python3 dahl.py "tulis puisi tentang hujan"     # satu prompt, jawab sekali
    python3 dahl.py -m "zai-org/GLM-5.3-Flash" "prompt"   # pakai model lain
    echo "prompt" | python3 dahl.py                  # baca dari stdin
    python3 dahl.py                                  # mode interaktif (REPL)
    python3 dahl.py --check                          # tes semua key, laporkan yang hidup

Cara kerja:
- Baca 148 key dari keys-round6.txt, rotasi round-robin.
- Kalau key kena 401 (mati) -> tandai mati, lanjut key berikut.
- Kalau kena 429 / model_concurrency (model penuh) -> ganti model berikut
  (MiniMax -> GLM -> DeepSeek), lalu ganti key.
- Key yang mati dicatat di .dead_keys.json supaya tidak dipakai lagi di sesi
  berikutnya (file ini hanya berisi nomor urut key, bukan key-nya).
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
DEAD_PATH = os.path.join(HERE, ".dead_keys.json")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def load_keys(path):
    keys = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("dahl_"):
                keys.append(line)
    return keys


def load_dead():
    try:
        with open(DEAD_PATH) as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_dead(dead):
    with open(DEAD_PATH, "w") as f:
        json.dump(sorted(dead), f)


def clean_answer(text):
    """Buang blok <think>...</think> (jejak reasoning) dari jawaban."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()


class DahlClient:
    def __init__(self, cfg):
        self.cfg = cfg
        self.keys = load_keys(cfg["key_file"])
        if not self.keys:
            raise SystemExit("Tidak ada key di " + cfg["key_file"])
        self.dead = load_dead()
        self.cursor = 0  # round-robin position

    def _next_key(self):
        """Ambil key hidup berikutnya (round-robin), None kalau semua mati."""
        for _ in range(len(self.keys)):
            idx = self.cursor % len(self.keys)
            self.cursor += 1
            if idx not in self.dead:
                return idx, self.keys[idx]
        return None, None

    def _call(self, key, model, messages, stream=False):
        body = {
            "model": model,
            "messages": messages,
            "temperature": self.cfg.get("temperature", 0.7),
            "max_tokens": self.cfg.get("max_tokens", 2048),
            "stream": stream,
        }
        req = urllib.request.Request(
            self.cfg["base_url"] + "/chat/completions",
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": UA,
            },
            data=json.dumps(body).encode(),
        )
        return urllib.request.urlopen(req, timeout=self.cfg.get("timeout_seconds", 60))

    def chat(self, messages, model=None, stream=False):
        """Kirim chat, otomatis rotasi key + fallback model. Return teks jawaban."""
        # Model yang diminta dicoba dulu, lalu sisanya dari config sebagai fallback.
        models = [model] if model else list(self.cfg["models"])
        if model:
            models += [m for m in self.cfg["models"] if m != model]
        tried_keys = set()
        last_err = None
        # coba tiap model, tiap model coba sampai semua key hidup terpakai
        for m in models:
            attempts = 0
            while attempts < len(self.keys):
                idx, key = self._next_key()
                if key is None:
                    break
                if idx in tried_keys:
                    break
                tried_keys.add(idx)
                attempts += 1
                try:
                    with self._call(key, m, messages, stream=False) as r:
                        data = json.loads(r.read().decode())
                    if "choices" in data and data["choices"]:
                        content = data["choices"][0]["message"]["content"]
                        return clean_answer(content), m, idx
                    last_err = f"respons tanpa choices dari model {m}"
                except urllib.error.HTTPError as e:
                    try:
                        err = json.loads(e.read().decode())
                        code = err.get("error", {}).get("code", "")
                        msg = err.get("error", {}).get("message", "")[:120]
                    except Exception:
                        code, msg = f"HTTP{e.code}", ""
                    if e.code == 401 or code == "invalid_api_key":
                        self.dead.add(idx)
                        save_dead(self.dead)
                        last_err = f"key #{idx} mati (401)"
                        continue
                    if e.code == 429 or code in ("rate_limit_error", "model_concurrency"):
                        last_err = f"model {m} penuh/limit ({code})"
                        break  # ganti model
                    last_err = f"HTTP {e.code} {code} {msg}"
                except Exception as e:
                    last_err = f"{type(e).__name__}: {str(e)[:100]}"
            # model berikut
        raise RuntimeError(f"Semua key/model gagal. Terakhir: {last_err}")

    def check_all(self):
        """Tes semua key dengan 1 chat kecil, laporkan ringkasan (tanpa print key)."""
        from concurrent.futures import ThreadPoolExecutor

        def probe(item):
            idx, key = item
            if idx in self.dead:
                return idx, "dead-sebelumnya"
            try:
                with self._call(key, self.cfg["models"][0],
                                [{"role": "user", "content": "OK"}]) as r:
                    data = json.loads(r.read().decode())
                return idx, "OK" if "choices" in data else "no-choices"
            except urllib.error.HTTPError as e:
                if e.code == 401:
                    return idx, "401-mati"
                return idx, f"HTTP{e.code}"
            except Exception as e:
                return idx, "ERR:" + type(e).__name__

        with ThreadPoolExecutor(max_workers=5) as ex:
            results = list(ex.map(probe, enumerate(self.keys)))
        for idx, st in results:
            if st == "401-mati":
                self.dead.add(idx)
        save_dead(self.dead)
        ok = [i for i, s in results if s == "OK"]
        dead401 = [i for i, s in results if s == "401-mati"]
        other = [(i, s) for i, s in results if s not in ("OK", "401-mati")]
        print(f"Total key: {len(self.keys)} | Hidup: {len(ok)} | Mati (401): {len(dead401)} | Gagal sementara: {len(other)}")
        if dead401:
            print("Key mati (nomor urut):", dead401)
        if other:
            print("Gagal sementara (kemungkinan rate limit, coba lagi nanti):", other[:10])
        return results


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="Dahl Inference client (key rotation + fallback)")
    ap.add_argument("prompt", nargs="*", help="prompt (kosong = baca stdin / REPL)")
    ap.add_argument("-m", "--model", default=None, help="model (default: urutan di config)")
    ap.add_argument("--check", action="store_true", help="tes semua key")
    ap.add_argument("--temp", type=float, default=None, help="temperature")
    args = ap.parse_args()

    if args.temp is not None:
        cfg["temperature"] = args.temp
    client = DahlClient(cfg)

    if args.check:
        client.check_all()
        return

    prompt = " ".join(args.prompt).strip()
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()

    if prompt:
        try:
            answer, model_used, key_idx = client.chat(
                [{"role": "user", "content": prompt}], model=args.model)
            print(answer)
            print(f"\n[model: {model_used} | key #{key_idx}]", file=sys.stderr)
        except RuntimeError as e:
            print(f"Gagal: {e}", file=sys.stderr)
            sys.exit(1)
        return

    # REPL interaktif
    print("Dahl REPL — ketik prompt, kosongkan untuk keluar. Model: "
          + (args.model or cfg["default_model"]))
    history = []
    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDadah.")
            break
        if not line:
            break
        history.append({"role": "user", "content": line})
        try:
            answer, model_used, key_idx = client.chat(history, model=args.model)
        except RuntimeError as e:
            print(f"Gagal: {e}")
            history.pop()
            continue
        print(answer)
        history.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
