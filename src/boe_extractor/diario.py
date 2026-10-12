"""Proceso diario: BOE del día → candidatos → JSON con el modelo GGUF en CPU → web estática.

Uso (lo lanza .github/workflows/diario.yml cada mañana, de lunes a sábado):
  uv run --with llama-cpp-python==0.3.19 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    boe-diario --modelo modelos/modelo.gguf --web web [--fecha AAAAMMDD]

Escribe web/datos/AAAA-MM-DD.jsonl (una fila por documento) y regenera web/index.html con el
último día y el enlace a los JSONL anteriores. Sin BOE ese día (domingos y festivos), no hace nada.
"""

from __future__ import annotations

import argparse
import html
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ETIQUETAS = {"convocatoria": "Empleo público", "ayuda": "Ayudas y subvenciones",
             "licitacion": "Licitaciones"}  # fmt: skip


def candidatos(fecha: str):
    """Sumario del día → documentos candidatos con su texto (o None si no hay BOE)."""
    import polars as pl

    from boe_extractor.classify import clasificar
    from boe_extractor.docs import descargar_documento, parsear_documento
    from boe_extractor.fetch import procesar_dia

    sumario = procesar_dia(fecha)
    if sumario is None:
        return None
    df = clasificar(pl.read_parquet(sumario))
    textos = {}
    for doc_id in df["id"]:
        xml = descargar_documento(doc_id)
        if xml is not None:
            textos[doc_id] = parsear_documento(xml)["texto"]
    return df.filter(pl.col("id").is_in(list(textos))).with_columns(
        texto=pl.col("id").replace_strict(textos)
    )


def euros(v: float | None) -> str:
    return "" if v is None else f"{v:,.0f} €".replace(",", ".")


def plazo(p: dict | None) -> str:
    if not p:
        return ""
    if p.get("dias"):
        tipo = {"habiles": " hábiles", "naturales": " naturales"}.get(p.get("tipo") or "", "")
        return f"{p['dias']} días{tipo}"
    return f"hasta el {p['fecha_fin']}" if p.get("fecha_fin") else ""


def celdas(tipo: str, j: dict) -> list[str]:
    if tipo == "convocatoria":
        plazas = sum(x["numero"] for x in j["plazas"]) or ""
        return [j["organismo"], j["puesto"] or "", str(plazas), j["sistema"] or "",
                plazo(j["plazo_solicitudes"])]  # fmt: skip
    if tipo == "ayuda":
        return [j["organismo"], j["objeto"], euros(j["cuantia_total_eur"]),
                plazo(j["plazo_solicitudes"])]  # fmt: skip
    return [j["organo_contratacion"], j["objeto"], euros(j["valor_estimado_eur"]),
            j["fecha_limite_ofertas"] or ""]  # fmt: skip


CABECERAS = {
    "convocatoria": ["Organismo", "Puesto", "Plazas", "Sistema", "Plazo"],
    "ayuda": ["Organismo", "Objeto", "Cuantía", "Plazo"],
    "licitacion": ["Órgano", "Objeto", "Valor estimado", "Fecha límite"],
}


def pagina(dia: str, filas: list[dict], anteriores: list[str]) -> str:
    """HTML autocontenido con las tablas del día y los enlaces a los días anteriores."""
    e = html.escape
    partes = [f"<h1>BOE del {e(dia)}</h1>"]
    for tipo, etiqueta in ETIQUETAS.items():
        del_tipo = [f for f in filas if f["tipo"] == tipo and f["json"]]
        if not del_tipo:
            continue
        cab = "".join(f"<th>{e(c)}</th>" for c in CABECERAS[tipo]) + "<th></th>"
        cuerpo = "".join(
            "<tr>" + "".join(f"<td>{e(c or '')}</td>" for c in celdas(tipo, f["json"]))
            + f'<td><a href="{e(f["url_html"])}">BOE</a></td></tr>'
            for f in del_tipo
        )  # fmt: skip
        partes.append(
            f"<h2>{e(etiqueta)} ({len(del_tipo)})</h2><table><tr>{cab}</tr>{cuerpo}</table>"
        )
    enlaces = " · ".join(f'<a href="datos/{e(d)}.jsonl">{e(d)}</a>' for d in anteriores)
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>boe-extractor</title>
<style>body{{font:15px system-ui,sans-serif;max-width:1100px;margin:auto;padding:16px}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #ddd;padding:6px;
text-align:left;vertical-align:top}}th{{background:#f4f4f4}}</style></head><body>
{"".join(partes)}
<p>Extraído automáticamente con un Qwen2.5-1.5B ajustado (QLoRA, GGUF en CPU); puede contener
errores, consulta siempre el BOE. <a href="https://github.com/inigo99/boe-extractor">Código</a>.
Datos en JSONL: {enlaces}</p></body></html>"""


def main() -> None:
    hoy = datetime.now(ZoneInfo("Europe/Madrid")).strftime("%Y%m%d")
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", type=Path, required=True)
    ap.add_argument("--web", type=Path, default=Path("web"))
    ap.add_argument("--fecha", default=hoy)
    args = ap.parse_args()

    from boe_extractor.docs import recortar
    from boe_extractor.infer_cpu import cargar, extraer
    from boe_extractor.train.sft import ejemplo

    df = candidatos(args.fecha)
    if df is None or df.is_empty():
        print(f"Sin documentos el {args.fecha}")
        return
    llm = cargar(args.modelo)
    dia = f"{args.fecha[:4]}-{args.fecha[4:6]}-{args.fecha[6:]}"
    datos = args.web / "datos"
    datos.mkdir(parents=True, exist_ok=True)
    filas = []
    textos = recortar(df["texto"].to_list())
    for n, (doc, texto) in enumerate(zip(df.iter_rows(named=True), textos, strict=True), 1):
        r = extraer(llm, ejemplo(doc, texto)["messages"], doc["tipo"])
        filas.append({k: doc[k] for k in ("id", "tipo", "subtipo", "titulo", "departamento",
                                          "url_html")} | {"fecha": dia} | r)  # fmt: skip
        print(f"{n}/{df.height} {doc['id']} {r['latencia_ms'] / 1000:.1f} s", flush=True)
    (datos / f"{dia}.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas), encoding="utf-8"
    )
    anteriores = sorted((p.stem for p in datos.glob("*.jsonl")), reverse=True)
    (args.web / "index.html").write_text(pagina(dia, filas, anteriores), encoding="utf-8")
    print(f"{len(filas)} documentos → {datos / f'{dia}.jsonl'}")


if __name__ == "__main__":
    main()
