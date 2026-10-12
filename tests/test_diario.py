from boe_extractor.diario import euros, pagina, plazo


def test_formatos():
    assert euros(1250000.5) == "1.250.000 €" and euros(None) == ""
    assert (
        plazo({"dias": 20, "tipo": "habiles", "desde": "publicacion_boe", "fecha_fin": None})
        == "20 días hábiles"
    )
    assert (
        plazo({"dias": None, "tipo": None, "desde": None, "fecha_fin": "2026-10-01"})
        == "hasta el 2026-10-01"
    )
    assert plazo(None) == ""


def test_pagina():
    conv = {"organismo": "Ayuntamiento de <Soria>", "puesto": "Policía Local", "vinculo": None,
            "plazas": [{"turno": "libre", "numero": 2}, {"turno": "discapacidad", "numero": 1}],
            "sistema": "oposicion", "plazo_solicitudes": None, "titulacion": None, "tasa_eur": None,
            "boletin_bases": None}  # fmt: skip
    filas = [{"tipo": "convocatoria", "json": conv, "url_html": "https://www.boe.es/x"},
             {"tipo": "ayuda", "json": None, "url_html": "https://www.boe.es/y"}]  # fmt: skip
    h = pagina("2026-10-13", filas, ["2026-10-13", "2026-10-11"])
    assert "Empleo público (1)" in h and "Ayudas" not in h  # sin JSON válido no se publica
    assert "&lt;Soria&gt;" in h and "<td>3</td>" in h and 'href="datos/2026-10-11.jsonl"' in h
