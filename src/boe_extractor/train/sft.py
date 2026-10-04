"""Conversaciones de chat para el modelo pequeño: silver → train/val, gold → entradas de test.

Uso (en tu PC, donde está data/): uv run boe-sft → data/sft/{train,val,gold}.jsonl
Sube data/sft/ a Google Drive (MyDrive/boe-extractor/sft/) para entrenar e inferir en Colab.

El prompt es el de Gemini (mismas instrucciones y mismo formato de entrada) más el esquema JSON
del tipo, para que el modelo base sin ajustar sepa qué campos rellenar.
"""

from __future__ import annotations

import hashlib
import json

from boe_extractor.fetch import DATA_DIR
from boe_extractor.label.silver import INSTRUCCIONES, entrada
from boe_extractor.schemas import ESQUEMAS

SFT = DATA_DIR / "sft"
VAL = 0.05  # fracción del silver para la pérdida de validación


def sistema(tipo: str) -> str:
    esquema = json.dumps(
        ESQUEMAS[tipo].model_json_schema(), ensure_ascii=False, separators=(",", ":")
    )
    return (
        f"{INSTRUCCIONES}\n\nTipo de documento: {tipo}. "
        f"Responde solo con un JSON que cumpla este esquema:\n{esquema}"
    )


def ejemplo(doc: dict, texto: str, salida: dict | None = None) -> dict:
    """{id, tipo, messages}; sin salida (gold) solo lleva system y user."""
    messages = [
        {"role": "system", "content": sistema(doc["tipo"])},
        {"role": "user", "content": entrada(doc, texto)},
    ]
    if salida is not None:
        messages.append({"role": "assistant", "content": json.dumps(salida, ensure_ascii=False)})
    return {"id": doc["id"], "tipo": doc["tipo"], "messages": messages}


def es_val(doc_id: str) -> bool:
    """Partición estable: no depende del orden ni de la semilla."""
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % 1000 < VAL * 1000


def main() -> None:
    import polars as pl

    from boe_extractor.docs import recortar
    from boe_extractor.evaluate import GOLD_FINAL, leer

    silver = {i: f for i, f in leer(DATA_DIR / "labels" / "silver.jsonl").items() if f["valido"]}
    gold = leer(GOLD_FINAL)
    docs = pl.read_parquet(DATA_DIR / "documents.parquet").filter(
        pl.col("id").is_in(list(silver) + list(gold))
    )
    textos = recortar(docs["texto"].to_list())
    salidas: dict[str, list[dict]] = {"train": [], "val": [], "gold": []}
    for doc, texto in zip(docs.iter_rows(named=True), textos, strict=True):
        if doc["id"] in gold:
            salidas["gold"].append(ejemplo(doc, texto))
        else:
            parte = "val" if es_val(doc["id"]) else "train"
            salidas[parte].append(ejemplo(doc, texto, silver[doc["id"]]["json"]))
    SFT.mkdir(exist_ok=True)
    for parte, filas in salidas.items():
        (SFT / f"{parte}.jsonl").write_text(
            "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas), encoding="utf-8"
        )
    print({k: len(v) for k, v in salidas.items()}, "→", SFT)


if __name__ == "__main__":
    main()
