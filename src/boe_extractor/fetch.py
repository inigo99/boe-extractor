"""Descarga del sumario diario del BOE con caché en disco y 1 petición por segundo.

Uso: boe-fetch AAAAMMDD [AAAAMMDD_FIN]

Cada día produce data/sumarios/AAAAMMDD.parquet (una fila por documento).
La sección II.A (nombramientos) se descarta al parsear: contiene nombres de personas.
Los días sin BOE (domingos) devuelven 404 y se marcan en caché para no repetir la petición.
"""

from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl

SUMARIO_URL = "https://www.boe.es/datosabiertos/api/boe/sumario/{fecha}"
USER_AGENT = "boe-extractor/0.1 (+https://github.com/inigo99/boe-extractor)"
SECCIONES_EXCLUIDAS = {"2A"}
MIN_INTERVALO_S = 1.0

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw" / "sumarios"
OUT_DIR = DATA_DIR / "sumarios"

SCHEMA = {
    "id": pl.String,
    "fecha": pl.Date,
    "diario_numero": pl.Int32,
    "seccion": pl.String,
    "seccion_nombre": pl.String,
    "departamento_codigo": pl.String,
    "departamento": pl.String,
    "epigrafe": pl.String,
    "titulo": pl.String,
    "pagina_inicial": pl.Int32,
    "pagina_final": pl.Int32,
    "pdf_kb": pl.Int32,
    "url_pdf": pl.String,
    "url_html": pl.String,
    "url_xml": pl.String,
}

_ultima_peticion = 0.0


def _get(url: str) -> bytes | None:
    """GET respetando 1 req/s. Devuelve None si el BOE responde 404 (día sin publicación)."""
    global _ultima_peticion
    espera = MIN_INTERVALO_S - (time.monotonic() - _ultima_peticion)
    if espera > 0:
        time.sleep(espera)
    req = urllib.request.Request(
        url, headers={"Accept": "application/xml", "User-Agent": USER_AGENT}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    finally:
        _ultima_peticion = time.monotonic()


def descargar_sumario(fecha: str, raw_dir: Path = RAW_DIR) -> bytes | None:
    """XML del sumario de `fecha` (AAAAMMDD), desde caché si existe. None si no hubo BOE."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    xml_path, sin_boe = raw_dir / f"{fecha}.xml", raw_dir / f"{fecha}.404"
    if xml_path.exists():
        return xml_path.read_bytes()
    if sin_boe.exists():
        return None
    contenido = _get(SUMARIO_URL.format(fecha=fecha))
    if contenido is None:
        sin_boe.touch()
    else:
        xml_path.write_bytes(contenido)
    return contenido


def _int(valor: str | None) -> int | None:
    return int(valor) if valor and valor.isdigit() else None


def parsear_sumario(xml: bytes) -> list[dict]:
    """Aplana el sumario en una fila por documento. Excluye SECCIONES_EXCLUIDAS."""
    raiz = ET.fromstring(xml)
    fecha = datetime.strptime(raiz.findtext(".//metadatos/fecha_publicacion"), "%Y%m%d").date()
    filas = []
    for diario in raiz.iter("diario"):
        numero = _int(diario.get("numero"))
        for seccion in diario.iter("seccion"):
            if seccion.get("codigo") in SECCIONES_EXCLUIDAS:
                continue
            for dep in seccion.iter("departamento"):
                # En la sección V los items cuelgan directamente del departamento, sin epígrafe.
                grupos = [(e.get("nombre"), e.findall("item")) for e in dep.findall("epigrafe")]
                grupos.append((None, dep.findall("item")))
                for epigrafe, items in grupos:
                    for item in items:
                        pdf = item.find("url_pdf")
                        attr = pdf.attrib if pdf is not None else {}
                        filas.append(
                            {
                                "id": item.findtext("identificador"),
                                "fecha": fecha,
                                "diario_numero": numero,
                                "seccion": seccion.get("codigo"),
                                "seccion_nombre": seccion.get("nombre"),
                                "departamento_codigo": dep.get("codigo"),
                                "departamento": dep.get("nombre"),
                                "epigrafe": epigrafe,
                                "titulo": item.findtext("titulo"),
                                "pagina_inicial": _int(attr.get("pagina_inicial")),
                                "pagina_final": _int(attr.get("pagina_final")),
                                "pdf_kb": _int(attr.get("szkbytes")),
                                "url_pdf": pdf.text if pdf is not None else None,
                                "url_html": item.findtext("url_html"),
                                "url_xml": item.findtext("url_xml"),
                            }
                        )
    return filas


def procesar_dia(fecha: str, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> Path | None:
    """Idempotente: si el Parquet del día ya existe, no hace nada."""
    out = out_dir / f"{fecha}.parquet"
    if out.exists():
        return out
    xml = descargar_sumario(fecha, raw_dir)
    if xml is None:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(parsear_sumario(xml), schema=SCHEMA).write_parquet(out)
    return out


def rango_fechas(inicio: str, fin: str) -> list[str]:
    d0, d1 = (datetime.strptime(f, "%Y%m%d").date() for f in (inicio, fin))
    return [(d0 + timedelta(days=i)).strftime("%Y%m%d") for i in range((d1 - d0).days + 1)]


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not 1 <= len(args) <= 2:
        sys.exit("Uso: boe-fetch AAAAMMDD [AAAAMMDD_FIN]")
    fechas = rango_fechas(args[0], args[-1])
    hoy = date.today().strftime("%Y%m%d")
    for fecha in fechas:
        if fecha > hoy:
            break
        out = procesar_dia(fecha)
        print(f"{fecha}: {out if out else 'sin BOE'}", flush=True)


if __name__ == "__main__":
    main()
