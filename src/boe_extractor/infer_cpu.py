"""Inferencia en CPU con el modelo GGUF (llama.cpp) y salida restringida al esquema JSON.

Uso (en tu PC):
  uv run --with llama-cpp-python==0.3.19 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    boe-infer-cpu --modelo data/modelos/qwen2.5-1.5b-qlora-v3.Q4_K_M.gguf \
    --salida data/preds/qwen2.5-1.5b-qlora-v3-cpu.jsonl [--limite 30]

Un documento cada vez, así que latencia_ms es la latencia real por documento (comparable con la
API). Reanudable: salta los ids ya escritos. Con --limite procesa primero los 30 documentos
anotados desde cero y luego el resto.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from boe_extractor.evaluate import GOLD_FINAL, leer
from boe_extractor.fetch import DATA_DIR
from boe_extractor.infer import MAX_LEN
from boe_extractor.label.silver import validar
from boe_extractor.schemas import ESQUEMAS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--datos", type=Path, default=DATA_DIR / "sft" / "gold.jsonl")
    ap.add_argument("--limite", type=int)
    ap.add_argument("--hilos", type=int, default=os.cpu_count())
    args = ap.parse_args()

    from llama_cpp import Llama

    gold = leer(GOLD_FINAL)
    filas = [json.loads(x) for x in args.datos.read_text(encoding="utf-8").splitlines()]
    filas.sort(key=lambda r: gold.get(r["id"], {}).get("prellenado", True))  # desde cero primero
    hechos = set(leer(args.salida)) if args.salida.exists() else set()
    filas = [r for r in filas if r["id"] not in hechos][: args.limite]
    llm = Llama(str(args.modelo), n_ctx=MAX_LEN, n_threads=args.hilos, verbose=False)
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("a", encoding="utf-8") as f:
        for n, r in enumerate(filas, 1):
            t0 = time.perf_counter()
            o = llm.create_chat_completion(
                messages=r["messages"],
                response_format={
                    "type": "json_object",
                    "schema": ESQUEMAS[r["tipo"]].model_json_schema(),
                },
                temperature=0,
                max_tokens=1024,
            )
            ms = (time.perf_counter() - t0) * 1000
            datos, error = validar(r["tipo"], o["choices"][0]["message"]["content"])
            fila = {
                "id": r["id"], "json": datos, "valido": datos is not None, "error": error,
                "latencia_ms": round(ms), "tokens_in": o["usage"]["prompt_tokens"],
                "tokens_out": o["usage"]["completion_tokens"],
            }  # fmt: skip
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{n}/{len(filas)} {r['id']} {ms / 1000:.1f} s", flush=True)


if __name__ == "__main__":
    main()
