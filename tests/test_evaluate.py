from boe_extractor.evaluate import alucinado, campos_de, contar, evaluar, f1_tokens


def test_campos_aplanados():
    c = campos_de("ayuda")
    assert "plazo_solicitudes.dias" in c and "plazo_solicitudes" not in c and "bdns_id" in c
    assert "plazas" in campos_de("convocatoria")  # las listas no se aplanan


def test_contar():
    assert contar("tasa_eur", None, None) == (0, 0, 0)
    assert contar("tasa_eur", None, 5.0) == (0, 1, 0)
    assert contar("tasa_eur", 5.0, None) == (0, 0, 1)
    assert contar("tasa_eur", 5.0, 5.001) == (1, 0, 0)
    assert contar("tasa_eur", 5.0, 6.0) == (0, 1, 1)
    assert contar("cpv", ["1", "2"], ["2", "3"]) == (1, 1, 1)
    assert f1_tokens("Ayuntamiento de Marbella", "ayuntamiento de marbella") == 1.0
    assert contar("organismo", "Ayuntamiento de Marbella", "Ayuntamiento")[0] == 0.5


def test_alucinado():
    texto = "Presupuesto: 1.250.000,50 euros. Hasta el 4 de septiembre de 2026. CPV 35331000."
    assert alucinado("valor_estimado_eur", 1250000.5, texto) is False
    assert alucinado("valor_estimado_eur", 999.0, texto) is True
    assert alucinado("fecha_limite_ofertas", "2026-09-04", texto) is False
    assert alucinado("cpv", ["35331000", "11111111"], texto) is True
    assert alucinado("procedimiento", "abierto", texto) is None


def test_evaluar():
    g = {"cpv": ["1"], "procedimiento": "abierto", "tipo_contrato": None}
    gold = {"a": {"tipo": "licitacion", "json": g}, "b": {"tipo": "licitacion", "json": g}}
    preds = {"a": {"json": g, "valido": True}, "b": {"json": None, "valido": False}}
    m = evaluar(gold, preds, {})
    assert m["json_valido"] == 0.5
    assert m["por_campo"]["licitacion/procedimiento"]["tp_fp_fn"] == [1, 0, 1]
    assert round(m["f1_micro"], 3) == round(2 * 2 / (2 * 2 + 2), 3)
