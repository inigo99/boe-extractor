"""Parseo del XML de documentos con fixtures reales (recortadas: sin firmas ni contactos)."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from boe_extractor.docs import _texto_plano, parsear_documento

FIX = Path(__file__).parent / "fixtures"


def test_licitacion_texto_y_analisis():
    doc = parsear_documento((FIX / "BOE-B-2026-30422.xml").read_bytes())
    lineas = doc["texto"].splitlines()
    assert lineas[:3] == [
        "1. Poder adjudicador:",
        "1.1) Nombre:",
        "Jefatura de la Sección Económico Administrativa 22 - Base Aérea de Torrejón (Agrupación de Base).",
    ]
    assert "Hasta las 23:00 horas del 15 de octubre de 2026." in lineas
    analisis = json.loads(doc["analisis"])
    assert analisis["tipo"] == ["Servicios"]
    assert analisis["procedimiento"] == ["Restringido"]
    assert analisis["materias_cpv"][0].startswith("55000000")
    assert "materias" not in analisis  # vacíos fuera
    assert doc["rango"] is None


def test_convocatoria_local_notas_y_rango():
    doc = parsear_documento((FIX / "BOE-A-2026-19568.xml").read_bytes())
    assert doc["rango"] == "Resolución"
    assert "quince días hábiles" in doc["texto"]
    assert len(json.loads(doc["analisis"])["nota"]) == 2


def test_tablas_una_linea_por_fila():
    texto = ET.fromstring(
        "<texto><p>Plazas:</p><table><tr><th>Turno</th><th>Plazas</th></tr>"
        "<tr><td>Libre</td><td>12</td></tr></table></texto>"
    )
    assert _texto_plano(texto) == "Plazas:\nTurno | Plazas\nLibre | 12"
