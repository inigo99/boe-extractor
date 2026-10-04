"""Mini app para anotar el gold: texto a la izquierda, JSON editable y validado a la derecha.

Uso: uv run --group anotar streamlit run src/boe_extractor/label/annotate.py

Lee data/labels/gold_pre.jsonl (propuestas de Gemini) y guarda en data/gold/gold.jsonl
(una línea por guardado; vale la última de cada id). Muestra el texto RECORTADO, que es lo que
verá el modelo: si un dato no está ahí, no se anota.
"""

from __future__ import annotations

import json
import time
import typing
from datetime import UTC, datetime
from pathlib import Path

from boe_extractor.fetch import DATA_DIR
from boe_extractor.schemas import ESQUEMAS, SCHEMA_VERSION

PRE = DATA_DIR / "labels" / "gold_pre.jsonl"
GOLD = DATA_DIR / "gold" / "gold.jsonl"


def leer_jsonl(path: Path) -> dict[str, dict]:
    """id → última fila (los guardados posteriores corrigen a los anteriores)."""
    if not path.exists():
        return {}
    filas = (json.loads(linea) for linea in path.read_text(encoding="utf-8").splitlines())
    return {f["id"]: f for f in filas}


def plantilla(tipo: str) -> dict:
    """JSON vacío del esquema: listas a [], el resto a null."""
    campos = ESQUEMAS[tipo].model_fields
    return {k: [] if typing.get_origin(c.annotation) is list else None for k, c in campos.items()}


def guardar(fila: dict, path: Path = GOLD) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(fila, ensure_ascii=False) + "\n")


def app() -> None:
    import polars as pl
    import streamlit as st

    from boe_extractor.docs import recortar

    st.set_page_config(page_title="boe-extractor · gold", layout="wide")

    @st.cache_resource
    def cargar():
        pre = leer_jsonl(PRE)
        docs = pl.read_parquet(DATA_DIR / "documents.parquet").filter(pl.col("id").is_in(list(pre)))
        docs = docs.with_columns(texto=pl.Series(recortar(docs["texto"].to_list())))
        return pre, {d["id"]: d for d in docs.iter_rows(named=True)}

    pre, docs = cargar()
    ids = sorted(pre, key=lambda i: (pre[i]["tipo"], pre[i]["subtipo"] or "", i))
    gold = leer_jsonl(GOLD)
    pendientes = [i for i in ids if i not in gold]

    with st.sidebar:
        st.metric("Anotados", f"{len(gold)}/{len(ids)}")
        for tipo in ESQUEMAS:
            del_tipo = [i for i in ids if pre[i]["tipo"] == tipo]
            st.progress(
                sum(i in gold for i in del_tipo) / max(len(del_tipo), 1),
                text=f"{tipo}: {sum(i in gold for i in del_tipo)}/{len(del_tipo)}",
            )
        defecto = ids.index(pendientes[0]) if pendientes else 0
        pos = st.number_input("Documento", 1, len(ids), st.session_state.get("pos", defecto + 1))
    st.session_state["pos"] = pos
    doc_id = ids[pos - 1]
    doc, propuesta = docs[doc_id], pre[doc_id]
    tipo = propuesta["tipo"]
    if st.session_state.get("doc") != doc_id:
        st.session_state.update(doc=doc_id, t0=time.monotonic())

    izq, der = st.columns([3, 2])
    with izq:
        st.markdown(f"**{doc_id}** · {tipo} {doc['subtipo'] or ''} · {doc['fecha']}")
        st.markdown(f"**{doc['titulo']}**  \n{doc['departamento']} · [BOE]({doc['url_html']})")
        with st.container(height=700):
            st.text(doc["texto"])
    with der:
        if doc_id in gold:
            st.success("Ya anotado: estás corrigiendo tu versión.")
            inicial = gold[doc_id]["json"]
        elif propuesta.get("sin_prellenado"):
            st.warning("Sin propuesta: anota desde cero (mide el sesgo del pre-rellenado).")
            inicial = plantilla(tipo)
        else:
            inicial = propuesta["json"] or plantilla(tipo)
        texto_json = st.text_area(
            "JSON", json.dumps(inicial, ensure_ascii=False, indent=2), height=640, key=doc_id
        )
        guardar_btn, descartar_btn = st.columns(2)
        fila = {
            "id": doc_id,
            "tipo": tipo,
            "subtipo": doc["subtipo"],
            "schema_version": SCHEMA_VERSION,
            "fuente": "humano",
            "prellenado": not propuesta.get("sin_prellenado", False),
            "segundos": round(time.monotonic() - st.session_state["t0"]),
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        if guardar_btn.button("Guardar y siguiente", type="primary", use_container_width=True):
            try:
                datos = ESQUEMAS[tipo].model_validate_json(texto_json).model_dump(mode="json")
            except Exception as e:  # JSON mal formado o fuera del esquema
                st.error(str(e))
            else:
                guardar(fila | {"json": datos, "descartado": False})
                st.session_state["pos"] = min(pos + 1, len(ids))
                st.rerun()
        if descartar_btn.button("No es de este tipo", use_container_width=True):
            guardar(fila | {"json": None, "descartado": True})
            st.session_state["pos"] = min(pos + 1, len(ids))
            st.rerun()


if __name__ == "__main__":
    app()
