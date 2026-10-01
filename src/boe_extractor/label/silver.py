"""Etiquetas silver con Gemini (nivel gratuito) y salida restringida al esquema de cada tipo.

Uso:
  boe-silver --piloto [--modelo M]   5 docs por grupo → data/labels/piloto_<M>.jsonl
  boe-silver                         muestra completa → data/labels/silver.jsonl

Reanudable: salta los ids ya etiquetados. Respeta 15 peticiones/min; si la cuota diaria se agota
(429 persistente), para y basta con relanzar al día siguiente.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from pydantic import ValidationError

from boe_extractor.docs import recortar
from boe_extractor.fetch import DATA_DIR
from boe_extractor.schemas import ESQUEMAS, SCHEMA_VERSION

MODELO = "gemini-3.5-flash-lite"
CORTE_TEST = pl.date(2026, 8, 1)  # entrenamiento: abr–jul; ago–sep queda para el gold
MUESTRA = {"bases": None, "ayuda": None, "licitacion": 450, "anuncio_local": 300}  # None = todos
PILOTO = 5
INTERVALO_S = 60 / 15
LABELS = DATA_DIR / "labels"

INSTRUCCIONES = """Extraes datos de documentos del Boletín Oficial del Estado (BOE) a JSON.
Reglas:
- Usa solo información del documento. Si un dato no aparece ni se deduce sin dudas,
  pon null (o [] en listas).
- No inventes ni completes con conocimiento general.
- Importes en euros como número (1.250.000,50 euros → 1250000.5).
- Fechas en formato AAAA-MM-DD.
- organismo: el nombre tal como aparece, sin abreviar y sin la provincia entre paréntesis.
- puesto: solo la denominación de la plaza o del cuerpo (sin escala, subescala ni grupo).
- vinculo: docente_universitario para cuerpos docentes universitarios (catedráticos, titulares).
- titulacion: solo el título académico exigido; null si el requisito no es una titulación.
- plazas: una entrada por turno (libre, promocion_interna, discapacidad u otro) con su número.
- Plazos: días y tipo (hábiles o naturales) solo si el texto lo dice; si no, tipo null.
  'desde' indica a partir de qué publicación se cuentan.
- cpv: solo los códigos de 8 dígitos."""


def _grupo() -> pl.Expr:
    return pl.coalesce("subtipo", "tipo")


def seleccionar(docs: pl.DataFrame, piloto: bool = False, seed: int = 42) -> pl.DataFrame:
    """Muestra estratificada del periodo de entrenamiento, en orden estable."""
    train = docs.filter(pl.col("fecha") < CORTE_TEST).with_columns(grupo=_grupo())
    partes = []
    for grupo, n in MUESTRA.items():
        g = train.filter(pl.col("grupo") == grupo).sort("id")
        n = PILOTO if piloto else n
        partes.append(g if n is None or n >= g.height else g.sample(n, seed=seed))
    return pl.concat(partes)


def entrada(doc: dict, texto: str) -> str:
    """Texto que ve el modelo (el mismo formato servirá para el modelo pequeño)."""
    return f"TÍTULO: {doc['titulo']}\nDEPARTAMENTO: {doc['departamento']}\n\nTEXTO:\n{texto}"


def validar(tipo: str, salida: str) -> tuple[dict | None, str | None]:
    try:
        return ESQUEMAS[tipo].model_validate_json(salida).model_dump(mode="json"), None
    except ValidationError as e:
        return None, str(e)[:500]


def _cargar_env(path: Path = Path(".env")) -> None:
    if path.exists():
        for linea in path.read_text(encoding="utf-8").splitlines():
            clave, _, valor = linea.partition("=")
            if clave.strip() and not clave.startswith("#") and valor.strip():
                os.environ.setdefault(clave.strip(), valor.strip())


def _llamar(client, modelo: str, tipo: str, prompt: str):
    from google.genai import errors, types

    config = types.GenerateContentConfig(
        system_instruction=INSTRUCCIONES,
        temperature=0,
        response_mime_type="application/json",
        response_json_schema=ESQUEMAS[tipo].model_json_schema(),
    )
    for intento in range(3):
        try:
            return client.models.generate_content(model=modelo, contents=prompt, config=config)
        except errors.APIError as e:
            if e.code == 429 and intento < 2:
                time.sleep(65)  # cuota por minuto: esperar y reintentar
                continue
            raise


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--piloto", action="store_true")
    ap.add_argument("--modelo", default=MODELO)
    args = ap.parse_args(argv)

    from google import genai

    _cargar_env()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    LABELS.mkdir(parents=True, exist_ok=True)
    out = LABELS / (f"piloto_{args.modelo}.jsonl" if args.piloto else "silver.jsonl")
    hechos = (
        {json.loads(linea)["id"] for linea in out.read_text(encoding="utf-8").splitlines()}
        if out.exists()
        else set()
    )

    muestra = seleccionar(pl.read_parquet(DATA_DIR / "documents.parquet"), piloto=args.piloto)
    muestra = muestra.filter(~pl.col("id").is_in(list(hechos)))
    print(f"{len(hechos)} ya etiquetados, {muestra.height} pendientes → {out}", flush=True)
    textos = recortar(muestra["texto"].to_list())

    with out.open("a", encoding="utf-8") as f:
        for i, (doc, texto) in enumerate(
            zip(muestra.iter_rows(named=True), textos, strict=True), 1
        ):
            t0 = time.monotonic()
            try:
                resp = _llamar(client, args.modelo, doc["tipo"], entrada(doc, texto))
            except Exception as e:  # cuota diaria agotada u otro error persistente
                print(f"Parado en {doc['id']}: {e}. Relanza más tarde para continuar.", flush=True)
                break
            datos, error = validar(doc["tipo"], resp.text or "")
            uso = resp.usage_metadata
            fila = {
                "id": doc["id"],
                "tipo": doc["tipo"],
                "subtipo": doc["subtipo"],
                "schema_version": SCHEMA_VERSION,
                "json": datos,
                "valido": datos is not None,
                "error": error,
                "fuente": f"api:{args.modelo}",
                "tokens_in": uso.prompt_token_count if uso else None,
                "tokens_out": uso.candidates_token_count if uso else None,
                "latencia_ms": round((time.monotonic() - t0) * 1000),
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            f.write(json.dumps(fila, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{muestra.height}", flush=True)
            time.sleep(max(0.0, INTERVALO_S - (time.monotonic() - t0)))


if __name__ == "__main__":
    main()
