"""Partes puras de la app de anotación (la interfaz Streamlit no se testea)."""

from boe_extractor.label.annotate import guardar, leer_jsonl, plantilla
from boe_extractor.schemas import ESQUEMAS


def test_plantilla_valida_salvo_obligatorios():
    p = plantilla("convocatoria")
    assert p["plazas"] == [] and p["plazo_solicitudes"] is None and p["organismo"] is None
    ESQUEMAS["convocatoria"].model_validate(p | {"organismo": "X"})
    assert plantilla("licitacion")["cpv"] == []


def test_ultimo_guardado_gana(tmp_path):
    path = tmp_path / "gold.jsonl"
    guardar({"id": "A", "json": {"v": 1}}, path)
    guardar({"id": "B", "json": {"v": 1}}, path)
    guardar({"id": "A", "json": {"v": 2}}, path)
    gold = leer_jsonl(path)
    assert gold["A"]["json"] == {"v": 2} and len(gold) == 2
