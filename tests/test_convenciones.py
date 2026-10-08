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


def test_plazo_un_mes_y_ventanas():
    p = {
        "plazo_solicitudes": {
            "dias": 1,
            "tipo": "habiles",
            "desde": "publicacion_boe",
            "fecha_fin": None,
        }
    }
    r = alinear("ayuda", p | {"regimen": None, "bases_reguladoras": None}, "",
                "Plazo de presentación de solicitudes: un mes desde la publicación en el BOE.")  # fmt: skip
    assert r["plazo_solicitudes"]["dias"] == 30 and r["plazo_solicitudes"]["tipo"] == "naturales"
    v = {"dias": None, "tipo": None, "desde": None, "fecha_fin": "2026-12-31"}
    fijo = "El plazo comenzará el día 4 de agosto de 2026 y finalizará el 31 de diciembre de 2026."
    pub = "Desde el día siguiente a la publicación del extracto hasta el 31 de diciembre de 2026."
    nada = "Podrán presentarse solicitudes hasta el 31 de diciembre de 2026."
    for texto, desde in ((fijo, "otro"), (pub, "publicacion_boe"), (nada, None)):
        r = alinear(
            "ayuda",
            {"plazo_solicitudes": dict(v), "regimen": None, "bases_reguladoras": None},
            "",
            texto,
        )
        assert r["plazo_solicitudes"]["desde"] == desde, texto


def test_plazas_sumadas_y_puestos_distintos():
    texto = "Cuatro plazas de Ingeniero/a, pertenecientes a…\nTres plazas de Arquitecto/a, pertenecientes a…"
    d = {
        "puesto": "Ingeniero/a",
        "plazas": [{"turno": "libre", "numero": 4}, {"turno": "libre", "numero": 3}],
    }
    r = alinear(
        "convocatoria", d | {"plazo_solicitudes": None}, "Resolución del Ayuntamiento", texto
    )
    assert r["plazas"] == [{"turno": "libre", "numero": 7}] and r["puesto"] is None
