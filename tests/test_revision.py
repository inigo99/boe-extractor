from boe_extractor.label.revision import aplicar, construir, formato, titulo


def fila(i, json, descartado=False):
    return {"id": i, "tipo": "convocatoria", "json": json, "descartado": descartado}


def test_titulo_y_formato():
    assert titulo("MINISTERIO DE LA PRESIDENCIA") == "Ministerio de la Presidencia"
    assert formato({"a": " Orden X, ", "b": ["UE "]}) == {"a": "Orden X", "b": ["UE"]}


def test_aplicar_subcampos():
    d = {"plazas": [{"turno": "libre", "numero": 1}], "plazo_solicitudes": None, "puesto": "x"}
    r = aplicar(d, {"plazas.turno": "otro", "plazo_solicitudes.dias": 5, "puesto": None})
    assert r == {
        "plazas": [{"turno": "otro", "numero": 1}],
        "plazo_solicitudes": None,
        "puesto": None,
    }
    assert d["puesto"] == "x"  # no muta la entrada


def test_construir():
    base = {
        "organismo": "AYUNTAMIENTO DE SORIA",
        "puesto": None,
        "vinculo": None,
        "plazas": [],
        "sistema": None,
        "plazo_solicitudes": None,
        "titulacion": None,
        "tasa_eur": None,
        "boletin_bases": None,
    }
    gold = {"a": fila("a", base), "b": fila("b", base), "c": fila("c", None, descartado=True)}
    rev = [
        {"n": 1, "ids": ["a"], "cambios": {"sistema": "oposicion"}, "decision": "aceptada"},
        {"n": 2, "ids": ["a"], "cambios": {"tasa_eur": 9}, "decision": "rechazada"},
        {"n": None, "ids": ["b"], "cambios": None, "decision": "descartar"},
    ]
    (a,) = construir(gold, rev)
    assert a["json"]["sistema"] == "oposicion" and a["json"]["tasa_eur"] is None
    assert a["json"]["organismo"] == "Ayuntamiento de Soria" and a["revision"] == [1]
