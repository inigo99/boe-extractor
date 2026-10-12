import json

from fastapi.testclient import TestClient

from boe_extractor import api

AYUDA = {"organismo": "Secretaría de Estado de Educación", "objeto": "Becas", "beneficiarios": None,
         "cuantia_total_eur": 510800.0, "importe_max_beneficiario_eur": None,
         "plazo_solicitudes": None, "regimen": None, "bdns_id": "925842",
         "bases_reguladoras": None}  # fmt: skip


class LlmFalso:
    def __init__(self):
        self.messages = None

    def create_chat_completion(self, messages, **kw):
        self.messages = messages
        return {"choices": [{"message": {"content": json.dumps(AYUDA)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50}}  # fmt: skip


def test_extract(monkeypatch):
    llm = LlmFalso()
    monkeypatch.setenv("BOE_MODELO", "x.gguf")
    monkeypatch.setattr(api, "cargar", lambda ruta: llm)
    monkeypatch.setattr(api, "recortar", lambda textos: textos)  # sin descargar el tokenizador
    with TestClient(api.app) as c:
        assert c.get("/health").json() == {"ok": True}
        r = c.post("/extract", json={"tipo": "ayuda", "titulo": "Extracto", "texto": "BDNS 925842"})
        assert r.status_code == 200
        assert r.json()["valido"] and r.json()["json"]["bdns_id"] == "925842"
        assert "BDNS 925842" in llm.messages[1]["content"]
        # entrada no válida: tipo desconocido o sin texto
        for malo in ({"tipo": "x", "titulo": "t", "texto": "y"}, {"tipo": "ayuda", "titulo": "t"}):
            assert c.post("/extract", json=malo).status_code == 422
