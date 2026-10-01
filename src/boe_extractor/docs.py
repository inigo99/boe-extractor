"""Descarga el XML de cada candidato y extrae su texto y los metadatos de <analisis>.

Uso: boe-docs  →  lee data/candidatos.parquet y escribe data/documents.parquet

La caché (data/raw/docs/ID.xml) hace que se pueda cortar y relanzar: solo baja lo que falta.
"""

from __future__ import annotations

import json
import urllib.error
import xml.etree.ElementTree as ET
from pathlib import Path

import polars as pl

from boe_extractor.fetch import DATA_DIR, _get

DOC_URL = "https://www.boe.es/diario_boe/xml.php?id={id}"
RAW_DOCS = DATA_DIR / "raw" / "docs"
BLOQUES = {"p", "dt", "dd", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}


def descargar_documento(doc_id: str, raw_dir: Path = RAW_DOCS) -> bytes | None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{doc_id}.xml"
    if path.exists():
        return path.read_bytes()
    contenido = _get(DOC_URL.format(id=doc_id))
    if contenido is not None:
        path.write_bytes(contenido)
    return contenido


def _texto_plano(texto: ET.Element) -> str:
    """Una línea por bloque (párrafo, dt/dd, fila de tabla); celdas separadas por ' | '."""
    lineas: list[str] = []

    def visitar(el: ET.Element) -> None:
        if el.tag == "tr":
            celdas = [" ".join("".join(c.itertext()).split()) for c in el]
            lineas.append(" | ".join(celdas))
            return
        if el.tag in BLOQUES and not any(h.tag in BLOQUES for h in el.iter() if h is not el):
            lineas.append(" ".join("".join(el.itertext()).split()))
            return
        for hijo in el:
            visitar(hijo)

    visitar(texto)
    return "\n".join(linea for linea in lineas if linea)


def parsear_documento(xml: bytes) -> dict:
    raiz = ET.fromstring(xml)
    analisis = raiz.find("analisis")
    meta = {}
    if analisis is not None:
        for el in analisis.iter():
            valor = " ".join("".join(el.itertext()).split())
            if len(el) == 0 and valor:  # hojas con texto: modalidad, tipo, cpv, notas...
                meta.setdefault(el.tag, []).append(valor)
    texto = raiz.find("texto")
    return {
        "rango": raiz.findtext("metadatos/rango"),
        "analisis": json.dumps(meta, ensure_ascii=False),
        "texto": _texto_plano(texto) if texto is not None else "",
    }


def main() -> None:
    candidatos = pl.read_parquet(DATA_DIR / "candidatos.parquet")
    ids = candidatos["id"].to_list()
    filas, fallos = [], []
    for i, doc_id in enumerate(ids, 1):
        try:
            xml = descargar_documento(doc_id)
        except (urllib.error.URLError, TimeoutError) as e:
            fallos.append(doc_id)
            print(f"{doc_id}: {e}", flush=True)
            continue
        if xml is not None:
            filas.append({"id": doc_id, **parsear_documento(xml)})
        if i % 200 == 0:
            print(f"{i}/{len(ids)}", flush=True)
    docs = candidatos.join(pl.DataFrame(filas), on="id", how="inner")
    out = DATA_DIR / "documents.parquet"
    docs.write_parquet(out)
    print(f"{docs.height} documentos → {out}. Fallos: {len(fallos)} (relanza para reintentar)")


MAX_TOKENS_ENTRADA = 6144
TOKENIZADOR = "Qwen/Qwen2.5-1.5B-Instruct"


def recortar(textos: list[str], max_tokens: int = MAX_TOKENS_ENTRADA) -> list[str]:
    """Recorta cada texto a max_tokens del tokenizador de Qwen (por el final). Misma regla para
    etiquetar, entrenar y servir."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(TOKENIZADOR)
    out = []
    for texto, enc in zip(textos, tok.encode_batch(textos), strict=True):
        out.append(texto if len(enc.ids) <= max_tokens else texto[: enc.offsets[max_tokens - 1][1]])
    return out


def tokens() -> None:
    """boe-tokens: añade n_tokens (tokenizador de Qwen) y muestra la distribución por tipo."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(TOKENIZADOR)
    path = DATA_DIR / "documents.parquet"
    docs = pl.read_parquet(path)
    n = [len(e.ids) for e in tok.encode_batch(docs["texto"].to_list())]
    docs = docs.with_columns(n_tokens=pl.Series(n, dtype=pl.Int32))
    docs.write_parquet(path)
    q = pl.col("n_tokens")
    print(
        docs.group_by("tipo", "subtipo")
        .agg(
            pl.len().alias("docs"),
            q.median().alias("p50"),
            q.quantile(0.95).alias("p95"),
            q.max().alias("max"),
            (q > 4096).mean().round(3).alias("frac_>4k"),
        )
        .sort("tipo", "subtipo")
    )


if __name__ == "__main__":
    main()
