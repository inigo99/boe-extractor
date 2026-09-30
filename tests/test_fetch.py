"""Tests de contrato de fetch.py contra una fixture real del sumario. Sin red."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from boe_extractor import fetch

FIXTURE = Path(__file__).parent / "fixtures" / "sumario_20260929.xml"


@pytest.fixture
def xml() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture
def sin_red(monkeypatch):
    """Sustituye la petición HTTP y cuenta las llamadas."""
    llamadas = []

    def falso_get(url):
        llamadas.append(url)
        return None if url.endswith("20260927") else FIXTURE.read_bytes()

    monkeypatch.setattr(fetch, "_get", falso_get)
    return llamadas


def test_parsea_documentos_y_excluye_2a(xml):
    filas = fetch.parsear_sumario(xml)
    assert [f["id"] for f in filas] == [
        "BOE-A-2026-20159",
        "BOE-A-2026-20160",
        "BOE-A-2026-20164",
        "BOE-A-2026-20205",
        "BOE-A-2026-20211",
        "BOE-B-2026-31510",
    ]
    assert all(f["seccion"] != "2A" for f in filas)


def test_campos_de_un_documento(xml):
    doc = fetch.parsear_sumario(xml)[0]
    assert doc["fecha"] == date(2026, 9, 29)
    assert doc["diario_numero"] == 240
    assert doc["seccion"] == "2B"
    assert doc["departamento"] == "CONSEJO GENERAL DEL PODER JUDICIAL"
    assert doc["epigrafe"] == "Carrera Judicial"
    assert (doc["pagina_inicial"], doc["pagina_final"], doc["pdf_kb"]) == (127164, 127171, 240)
    assert doc["url_xml"] == "https://www.boe.es/diario_boe/xml.php?id=BOE-A-2026-20159"


def test_seccion_v_sin_epigrafe(xml):
    doc = fetch.parsear_sumario(xml)[-1]
    assert doc["seccion"] == "5A"
    assert doc["epigrafe"] is None
    assert doc["titulo"].startswith("Anuncio de formalización")


def test_entidades_xml_decodificadas(xml):
    assert "Nakhal & CIE" in fetch.parsear_sumario(xml)[4]["titulo"]


def test_procesar_dia_escribe_parquet_y_es_idempotente(tmp_path, sin_red):
    raw, out = tmp_path / "raw", tmp_path / "out"
    p1 = fetch.procesar_dia("20260929", raw, out)
    p2 = fetch.procesar_dia("20260929", raw, out)
    assert p1 == p2 == out / "20260929.parquet"
    assert len(sin_red) == 1
    df = pl.read_parquet(p1)
    assert df.height == 6
    assert df.schema == pl.Schema(fetch.SCHEMA)


def test_cache_raw_evita_red_si_se_borra_el_parquet(tmp_path, sin_red):
    raw, out = tmp_path / "raw", tmp_path / "out"
    fetch.procesar_dia("20260929", raw, out).unlink()
    fetch.procesar_dia("20260929", raw, out)
    assert len(sin_red) == 1


def test_dia_sin_boe_se_cachea(tmp_path, sin_red):
    raw, out = tmp_path / "raw", tmp_path / "out"
    assert fetch.procesar_dia("20260927", raw, out) is None
    assert fetch.procesar_dia("20260927", raw, out) is None
    assert len(sin_red) == 1
    assert (raw / "20260927.404").exists()


def test_rango_fechas():
    assert fetch.rango_fechas("20260228", "20260302") == ["20260228", "20260301", "20260302"]
