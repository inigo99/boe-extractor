import json

from boe_extractor.train.sft import ejemplo, es_val


def test_ejemplo_y_particion():
    d = {"id": "X", "tipo": "ayuda", "titulo": "T", "departamento": "D"}
    e = ejemplo(d, "texto", {"bdns_id": "1"})
    assert [m["role"] for m in e["messages"]] == ["system", "user", "assistant"]
    assert '"bdns_id"' in e["messages"][0]["content"]  # el esquema va en el prompt
    assert json.loads(e["messages"][2]["content"]) == {"bdns_id": "1"}
    assert len(ejemplo(d, "texto")["messages"]) == 2
    vals = sum(es_val(f"BOE-B-2026-{i}") for i in range(2000))
    assert 50 < vals < 150 and es_val("a") == es_val("a")
