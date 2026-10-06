from boe_extractor.label.convenciones import alinear, es_provision


def test_regimen_y_bases():
    d = {"regimen": "concurrencia_competitiva",
         "bases_reguladoras": "Orden DEF/395/2019, de 1 de abril, por la que se establecen las bases"}  # fmt: skip
    r = alinear("ayuda", d, "", "Premio sin mención del procedimiento")
    assert r["regimen"] is None and r["bases_reguladoras"] == "Orden DEF/395/2019, de 1 de abril"
    assert alinear("ayuda", d, "", "en régimen de concurrencia competitiva")["regimen"]
    assert (
        alinear(
            "ayuda", {"regimen": None, "bases_reguladoras": "Bases de la convocatoria"}, "", ""
        )["bases_reguladoras"]
        == "Bases de la convocatoria"
    )


def test_provision():
    t = "Resolución de 6 de agosto de 2026, de la Subsecretaría, por la que se convoca concurso específico"
    assert es_provision(t, "")
    assert not es_provision(
        "…, por la que se convoca concurso de acceso a plazas de cuerpos docentes", ""
    )
    assert es_provision("Resolución del Ayuntamiento de X", "para proveer por concurso: Un puesto")
    assert not es_provision(
        "Resolución del Ayuntamiento de X", "por concurso-oposición, en turno libre"
    )
    d = {"plazas": [{"turno": "libre", "numero": 1}, {"turno": "discapacidad", "numero": 1}]}
    r = alinear("convocatoria", d, t, "")
    assert [p["turno"] for p in r["plazas"]] == ["otro", "discapacidad"]
