"""Selección de la muestra y validación de respuestas (sin red)."""

from datetime import date

import polars as pl

from boe_extractor.label.silver import MUESTRA, seleccionar, validar


def _docs() -> pl.DataFrame:
    filas = []
    for grupo, tipo, subtipo in [
        ("bases", "convocatoria", "bases"),
        ("anuncio_local", "convocatoria", "anuncio_local"),
        ("ayuda", "ayuda", None),
        ("licitacion", "licitacion", None),
    ]:
        for i in range(600):
            filas.append(
                {
                    "id": f"{grupo}-{i:03d}",
                    "fecha": date(2026, 9, 1) if i % 4 == 0 else date(2026, 5, 1),
                    "tipo": tipo,
                    "subtipo": subtipo,
                }
            )
    return pl.DataFrame(filas)


def test_muestra_solo_entrenamiento_y_tamanos():
    m = seleccionar(_docs())
    assert m["fecha"].max() < date(2026, 8, 1)
    n = dict(m.group_by("grupo").len().iter_rows())
    assert (
        n["licitacion"] == MUESTRA["licitacion"] and n["anuncio_local"] == MUESTRA["anuncio_local"]
    )
    assert n["bases"] == n["ayuda"] == 450  # "todos" los de abr–jul (600 × 3/4)
    assert m.equals(seleccionar(_docs()))  # determinista: permite reanudar


def test_piloto_5_por_grupo():
    assert seleccionar(_docs(), piloto=True).height == 20


def test_validar():
    ok, err = validar(
        "licitacion", '{"organo_contratacion": "X", "objeto": "Y", "cpv": ["71310000"]}'
    )
    assert err is None and ok["cpv"] == ["71310000"] and ok["lotes"] is None
    malo, err = validar(
        "licitacion", '{"organo_contratacion": "X", "objeto": "Y", "procedimiento": "subasta"}'
    )
    assert malo is None and "procedimiento" in err
    assert validar("ayuda", "no es json")[0] is None


def test_gold_solo_test_y_sin_prellenado():
    g = seleccionar(_docs(), gold=True)
    assert g["fecha"].min() >= date(2026, 8, 1)
    n = dict(g.group_by("grupo").len().iter_rows())
    assert n == {"bases": 50, "anuncio_local": 50, "ayuda": 150, "licitacion": 100}
    assert dict(g.group_by("tipo").agg(pl.col("sin_prellenado").sum()).iter_rows()) == {
        "convocatoria": 10,
        "ayuda": 10,
        "licitacion": 10,
    }
    assert g.equals(seleccionar(_docs(), gold=True))
