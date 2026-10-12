"""API HTTP: POST /extract con un documento del BOE → JSON validado contra el esquema.

Uso:
  BOE_MODELO=modelos/modelo.gguf uv run --group api \
    --with llama-cpp-python==0.3.19 \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
    uvicorn boe_extractor.api:app

  curl -X POST localhost:8000/extract -H 'Content-Type: application/json' \
    -d '{"tipo": "ayuda", "titulo": "Extracto de…", "texto": "BDNS (Identif.): …"}'

Mismo prompt, recorte y salida restringida que en entrenamiento y en el proceso diario.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from boe_extractor.docs import recortar
from boe_extractor.infer_cpu import cargar, extraer
from boe_extractor.train.sft import ejemplo


class Peticion(BaseModel):
    tipo: Literal["convocatoria", "ayuda", "licitacion"]
    titulo: str = Field(min_length=1)
    departamento: str = ""
    texto: str = Field(min_length=1, max_length=500_000)


class Respuesta(BaseModel):
    json_: dict | None = Field(alias="json")
    valido: bool
    error: str | None
    latencia_ms: int
    tokens_in: int
    tokens_out: int


estado: dict = {}
# llama.cpp no admite llamadas concurrentes sobre el mismo modelo.
# ponytail: un documento cada vez; varias réplicas detrás de un balanceador si hace falta más.
_lock = threading.Lock()


@asynccontextmanager
async def vida(app: FastAPI):
    estado["llm"] = cargar(Path(os.environ["BOE_MODELO"]))
    yield
    estado.clear()


app = FastAPI(title="boe-extractor", lifespan=vida)


@app.get("/health")
def health() -> dict:
    return {"ok": "llm" in estado}


@app.post("/extract", response_model=Respuesta)
def extract(p: Peticion) -> dict:
    doc = {"id": "", "tipo": p.tipo, "titulo": p.titulo, "departamento": p.departamento}
    messages = ejemplo(doc, recortar([p.texto])[0])["messages"]
    with _lock:
        return extraer(estado["llm"], messages, p.tipo)
