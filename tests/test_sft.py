import json

from boe_extractor.train.sft import ejemplo, es_val, fila_hf


def test_ejemplo_y_particion():
    d = {"id": "X", "tipo": "ayuda", "titulo": "T", "departamento": "D"}
    e = ejemplo(d, "texto", {"bdns_id": "1"})
    assert [m["role"] for m in e["messages"]] == ["system", "user", "assistant"]
    assert '"bdns_id"' in e["messages"][0]["content"]  # el esquema va en el prompt
    assert json.loads(e["messages"][2]["content"]) == {"bdns_id": "1"}
    assert len(ejemplo(d, "texto")["messages"]) == 2
    vals = sum(es_val(f"BOE-B-2026-{i}") for i in range(2000))
    assert 50 < vals < 150 and es_val("a") == es_val("a")


def test_fila_hf():
    import datetime

    d = {"id": "X", "tipo": "ayuda", "subtipo": "bdns", "titulo": "T", "departamento": "D",
         "url_html": "u", "fecha": datetime.date(2026, 8, 3), "seccion": "5B"}  # fmt: skip
    f = fila_hf(d, "texto", {"bdns_id": "1"}, "silver")
    assert f["fecha"] == "2026-08-03" and "seccion" not in f
    assert json.loads(f["json"]) == {"bdns_id": "1"} and f["etiqueta"] == "silver"
