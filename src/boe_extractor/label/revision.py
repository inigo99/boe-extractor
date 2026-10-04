"""Aplica la revisión del gold y escribe data/gold/gold_final.jsonl.

Uso: uv run boe-gold-final

data/gold/revision.jsonl tiene una fila por propuesta de Claude:
{"n", "ids", "cambios": {campo|campo.subcampo: valor} | null, "motivo", "decision"}
decision: "aceptada" | "rechazada" (la decide Íñigo), "convencion" (regla acordada, se aplica)
o "descartar" (los ids salen del gold). Además se normaliza el formato: espacios y comas
sobrantes, y organismos EN MAYÚSCULAS pasan a tipo título.
"""

from __future__ import annotations

import copy
import json

from boe_extractor.label.annotate import GOLD, guardar, leer_jsonl
from boe_extractor.schemas import ESQUEMAS

REVISION = GOLD.with_name("revision.jsonl")
FINAL = GOLD.with_name("gold_final.jsonl")
APLICAR = {"aceptada", "convencion"}
MINUSCULAS = {
    *("de", "del", "la", "las", "los", "el", "y", "e", "en", "para", "a", "al", "con", "por")
}


def titulo(s: str) -> str:
    """'MINISTERIO DE LA PRESIDENCIA' → 'Ministerio de la Presidencia'."""
    palabras = s.lower().split()
    return " ".join(
        p if i and p in MINUSCULAS else p[:1].upper() + p[1:] for i, p in enumerate(palabras)
    )


def formato(x):
    if isinstance(x, dict):
        return {k: formato(v) for k, v in x.items()}
    if isinstance(x, list):
        return [formato(v) for v in x]
    if isinstance(x, str):
        x = x.strip().rstrip(",").strip()
        return titulo(x) if x.isupper() and len(x) > 6 else x
    return x


def aplicar(datos: dict, cambios: dict) -> dict:
    datos = copy.deepcopy(datos)
    for campo, valor in cambios.items():
        padre, _, hijo = campo.partition(".")
        if not hijo:
            datos[padre] = valor
        elif padre == "plazas":  # el valor se aplica a todas las entradas
            for p in datos["plazas"]:
                p[hijo] = valor
        elif datos.get(padre) is not None:
            datos[padre][hijo] = valor
    return datos


def construir(gold: dict[str, dict], revision: list[dict]) -> list[dict]:
    descartes = {i for r in revision if r["decision"] == "descartar" for i in r["ids"]}
    aplicadas: dict[str, list] = {}
    final = {i: copy.deepcopy(f) for i, f in gold.items() if not f["descartado"]}
    for r in revision:
        if r["decision"] not in APLICAR:
            continue
        for i in r["ids"]:
            if i in final and i not in descartes:
                final[i]["json"] = aplicar(final[i]["json"], r["cambios"])
                aplicadas.setdefault(i, []).append(r["n"] or "convención")
    filas = []
    for i, f in sorted(final.items()):
        if i in descartes:
            continue
        datos = ESQUEMAS[f["tipo"]].model_validate(formato(f["json"])).model_dump(mode="json")
        filas.append(f | {"json": datos, "revision": aplicadas.get(i, [])})
    return filas


def leer_lineas(path) -> list[dict]:
    return [json.loads(linea) for linea in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    pendientes = [r["n"] for r in leer_lineas(REVISION) if r["decision"] is None]
    if pendientes:
        raise SystemExit(f"Faltan decisiones: {pendientes}")
    FINAL.unlink(missing_ok=True)
    filas = construir(leer_jsonl(GOLD), leer_lineas(REVISION))
    for f in filas:
        guardar(f, FINAL)
    print(f"{len(filas)} documentos → {FINAL}")


if __name__ == "__main__":
    main()
