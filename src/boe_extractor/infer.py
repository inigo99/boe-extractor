"""Inferencia con vLLM y decodificación restringida al esquema JSON de cada tipo (en Colab, GPU).

Uso:
  python -m boe_extractor.infer --datos sft/gold.jsonl --salida preds/qwen2.5-1.5b-base.jsonl
  python -m boe_extractor.infer ... --lora qlora-v1/adaptador --salida preds/<nombre>.jsonl

Escribe filas {id, json, valido, error, latencia_ms, tokens_in, tokens_out} que lee boe-eval.
latencia_ms es el tiempo del lote repartido entre sus documentos (vLLM procesa en paralelo).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from boe_extractor.label.silver import validar
from boe_extractor.schemas import ESQUEMAS

MODELO = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_LEN = 8192  # 6.144 de texto + instrucciones y esquema (~1.000) + salida


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datos", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--modelo", default=MODELO)
    ap.add_argument("--lora", type=Path, help="carpeta del adaptador LoRA")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.sampling_params import StructuredOutputsParams

    filas = [json.loads(x) for x in args.datos.read_text(encoding="utf-8").splitlines()]
    llm = LLM(
        model=args.modelo,
        dtype="half",  # la T4 de Colab no tiene bf16
        max_model_len=MAX_LEN,
        enable_lora=args.lora is not None,
        max_lora_rank=64,
        seed=0,
    )
    lora = LoRARequest("adaptador", 1, str(args.lora)) if args.lora else None
    tok = llm.get_tokenizer()
    args.salida.parent.mkdir(parents=True, exist_ok=True)
    with args.salida.open("w", encoding="utf-8") as f:
        for tipo, esquema in ESQUEMAS.items():
            lote = [r for r in filas if r["tipo"] == tipo]
            if not lote:
                continue
            prompts = [
                tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
                for r in lote
            ]
            params = SamplingParams(
                temperature=0,
                max_tokens=1024,
                structured_outputs=StructuredOutputsParams(json=esquema.model_json_schema()),
            )
            t0 = time.perf_counter()
            salidas = llm.generate(prompts, params, lora_request=lora)
            ms = (time.perf_counter() - t0) * 1000 / len(lote)
            for r, o in zip(lote, salidas, strict=True):
                datos, error = validar(tipo, o.outputs[0].text)
                fila = {
                    "id": r["id"], "json": datos, "valido": datos is not None, "error": error,
                    "latencia_ms": round(ms), "tokens_in": len(o.prompt_token_ids),
                    "tokens_out": len(o.outputs[0].token_ids),
                }  # fmt: skip
                f.write(json.dumps(fila, ensure_ascii=False) + "\n")
    print(f"{len(filas)} predicciones → {args.salida}")


if __name__ == "__main__":
    main()
