"""Fusiona el adaptador LoRA con el modelo base y lo exporta a GGUF para CPU (en Colab).

Uso: python -m boe_extractor.train.exportar --adaptador qlora-v3/adaptador --salida qlora-v3/gguf

q4_k_m: ~1 GB para 1,5B parámetros, la cuantización habitual para CPU.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from boe_extractor.infer import MAX_LEN


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adaptador", type=Path, required=True)
    ap.add_argument("--salida", type=Path, required=True)
    ap.add_argument("--cuantizacion", default="q4_k_m")
    args = ap.parse_args()

    from unsloth import FastLanguageModel

    model, tok = FastLanguageModel.from_pretrained(
        str(args.adaptador), max_seq_length=MAX_LEN, load_in_4bit=False
    )
    model.save_pretrained_gguf(str(args.salida), tok, quantization_method=args.cuantizacion)
    print(*sorted(args.salida.rglob("*.gguf")), sep="\n")


if __name__ == "__main__":
    main()
