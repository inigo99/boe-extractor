"""Sistemas de referencia sobre el gold: reglas (regex) y Gemini (las propuestas ya guardadas).

Uso: uv run boe-baseline   → data/preds/reglas.jsonl y data/preds/gemini-3.5-flash-lite.jsonl

Las reglas son el suelo: si un LLM no las supera con claridad, no hace falta un LLM.
Gemini es el techo barato, con una salvedad: el gold se pre-rellenó con él, así que su nota es
optimista salvo en los 30 documentos anotados desde cero (evaluate.py lo separa).
"""

from __future__ import annotations

import json
import re
import time

from boe_extractor import normalize
from boe_extractor.fetch import DATA_DIR
from boe_extractor.schemas import ESQUEMAS

PREDS = DATA_DIR / "preds"
_ORG = re.compile(
    r", (?:de|del) (?:la |el |los |las )?(.+?)(?: \([^)]*\))?, (?:por (?:la|el) que|referente)"
)


def organismo(titulo: str, departamento: str) -> str:
    m = _ORG.search(titulo)
    return m.group(1) if m else departamento.title()


def ventana(texto: str, patron: str, n: int = 400) -> str:
    """Los n caracteres que siguen a la primera aparición de patron ('' si no aparece)."""
    m = re.search(patron, texto, re.I)
    return texto[m.end() : m.end() + n] if m else ""


def tras(texto: str, etiqueta: str) -> str | None:
    """Línea siguiente a una etiqueta de la plantilla de licitaciones ('5. Tipo de …:')."""
    m = re.search(etiqueta + r"[^\n]*:\n([^\n]+)", texto)
    return m.group(1).strip().rstrip(".").strip() if m else None


def apartado(texto: str, titulo: str) -> str | None:
    """Contenido de 'Segundo. Objeto: …' (misma línea o la siguiente), hasta 300 caracteres."""
    m = re.search(
        r"(?:Primero|Segundo|Tercero|Cuarto)\.?\s*-?\s*" + titulo + r"[.:]?\s*([^\n]*)\n([^\n]*)",
        texto,
    )
    return ((m.group(1) or m.group(2)).strip()[:300] or None) if m else None


def plazo(texto: str) -> dict | None:
    p = normalize.plazo(ventana(texto, r"plazo (?:de|para la) presentaci[oó]n", 500) or texto)
    return p.model_dump(mode="json") if p else None


def primero(texto: str, opciones: dict[str, str]) -> str | None:
    """Valor de la primera clave (regex) que aparece en el texto."""
    pos = {v: m.start() for k, v in opciones.items() if (m := re.search(k, texto, re.I))}
    return min(pos, key=pos.get) if pos else None


def convocatoria(doc: dict, texto: str) -> dict:
    inicio = texto[:4000]
    m = re.search(r"\b([\wáéíóú]+(?: y [\wáéíóú]+)?) plazas? (?:de|del)\b", inicio, re.I)
    n = normalize.numero(m.group(1)) if m else None
    bol = re.search(r"«((?:Boletín|Diari|Diario)[^»]+)» (?:número|núm)", inicio)
    tit = re.search(r"posesi[oó]n del t[ií]tulo de ([^,.;(]+)", texto, re.I)
    return {
        "organismo": organismo(doc["titulo"], doc["departamento"]),
        "puesto": None,
        "vinculo": primero(
            inicio,
            {
                r"docentes universitarios": "docente_universitario",
                r"personal laboral": "laboral",
                r"estatutari": "estatutario",
                r"funcionari|escala de administración": "funcionario",
            },
        ),
        "plazas": [{"turno": "libre", "numero": n}] if n else [],
        "sistema": primero(
            inicio,
            {
                r"concurso-oposici": "concurso_oposicion",
                r"oposici[oó]n": "oposicion",
                r"concurso": "concurso",
            },  # fmt: skip
        ),
        "plazo_solicitudes": plazo(texto),
        "titulacion": tit.group(1).strip() if tit else None,
        "tasa_eur": normalize.importe(ventana(texto, r"derechos de examen|tasa (?:por|de) ", 300)),
        "boletin_bases": bol.group(1) if bol and doc["subtipo"] == "anuncio_local" else None,
    }


def ayuda(doc: dict, texto: str) -> dict:
    bdns = re.search(r"BDNS\s*\(Identif\.\)\s*:\s*(\d+)", texto)
    bases = re.search(
        r"(?:Orden [A-Z]+/\d+/\d{4}|Real Decreto \d+/\d{4}),? de \d{1,2} de \w+",
        ventana(texto, r"bases reguladoras", 600),
    )
    p = plazo(texto)
    if p is None and (f := normalize.fecha(ventana(texto, r"plazo de presentaci[oó]n", 400))):
        p = {"dias": None, "tipo": None, "desde": None, "fecha_fin": f.isoformat()}
    return {
        "organismo": organismo(doc["titulo"], doc["departamento"]),
        "objeto": apartado(texto, r"Objeto[^.:\n]*"),
        "beneficiarios": apartado(texto, r"(?:Personas |Entidades )?Beneficiari\w*"),
        "cuantia_total_eur": normalize.importe(
            ventana(texto, r"cuant[ií]a|presupuesto|dotaci[oó]n|importe", 600)
        ),
        "importe_max_beneficiario_eur": normalize.importe(
            ventana(texto, r"(?:cuant[ií]a|importe) m[aá]xim[ao][^.]{0,80}?por ", 200)
        ),
        "plazo_solicitudes": p,
        "regimen": primero(
            texto,
            {
                r"concurrencia competitiva": "concurrencia_competitiva",
                r"concesi[oó]n directa": "concesion_directa",
            },  # fmt: skip
        ),
        "bdns_id": bdns.group(1) if bdns else None,
        "bases_reguladoras": bases.group(0) if bases else None,
    }


_PROCEDIMIENTO = {
    "abierto simplificado": "abierto_simplificado", "abierto": "abierto",
    "restringido": "restringido", "negociado": "negociado",
    "diálogo competitivo": "dialogo_competitivo",
}  # fmt: skip


def licitacion(doc: dict, texto: str) -> dict:
    proc = (tras(texto, r"Tipo de procedimiento") or "").lower()
    nuts = re.search(r"\d\.\s*Lugar[^\n]*:\n(?:[^\n]*NUTS principal:\n)?(ES\w*)", texto)
    seccion = re.split(r"\n\d+\. ", ventana(texto, r"Códigos CPV", 3000))[0]
    cpv_linea = tras(texto, r"CPV principal") or seccion
    lotes = {int(x) for x in re.findall(r"Lote (\d+):", texto)}
    dur = re.search(r"(\d+) meses", tras(texto, r"Duración del contrato") or "")
    fecha = normalize.fecha(tras(texto, r"Plazo para la recepción de ofertas") or "")
    return {
        "organo_contratacion": tras(texto, r"1\.1\) Nombre") or doc["departamento"].title(),
        "objeto": tras(texto, r"Descripción de la licitación") or doc["titulo"],
        "tipo_contrato": primero(
            texto,
            {
                r"entrega de los suministros": "suministros",
                r"prestación de los servicios": "servicios",
                r"emplazamiento principal de las obras": "obras",
            },  # fmt: skip
        ),
        "procedimiento": next((v for k, v in _PROCEDIMIENTO.items() if proc.startswith(k)), None)
        or ("otro" if proc else None),
        "valor_estimado_eur": normalize.importe(tras(texto, r"Valor estimado") or ""),
        "presupuesto_base_eur": None,
        "fecha_limite_ofertas": fecha.isoformat() if fecha else None,
        "lugar_ejecucion_nuts": nuts.group(1) if nuts else None,
        "cpv": list(dict.fromkeys(re.findall(r"\b(\d{8})\b", cpv_linea)))
        or re.findall(r"CPV: (\d{8})", texto),
        "duracion_meses": int(dur.group(1)) if dur else None,
        "lotes": max(lotes) if lotes else None,
    }


REGLAS = {"convocatoria": convocatoria, "ayuda": ayuda, "licitacion": licitacion}


def reglas(doc: dict, texto: str) -> dict:
    t0 = time.perf_counter()
    try:  # una regla que no casa no debe tumbar la ejecución: cuenta como JSON inválido
        salida = REGLAS[doc["tipo"]](doc, texto)
        datos = ESQUEMAS[doc["tipo"]].model_validate(salida).model_dump(mode="json")
    except Exception as e:
        print(f"{doc['id']}: {str(e)[:200]}")
        datos = None
    return {"id": doc["id"], "json": datos, "valido": datos is not None,
            "latencia_ms": (time.perf_counter() - t0) * 1000}  # fmt: skip


def main() -> None:
    import polars as pl

    from boe_extractor.docs import recortar
    from boe_extractor.evaluate import GOLD_FINAL, leer

    gold = leer(GOLD_FINAL)
    docs = pl.read_parquet(DATA_DIR / "documents.parquet").filter(pl.col("id").is_in(list(gold)))
    PREDS.mkdir(exist_ok=True)
    textos = recortar(docs["texto"].to_list())
    filas = [reglas(d, t) for d, t in zip(docs.iter_rows(named=True), textos, strict=True)]
    (PREDS / "reglas.jsonl").write_text(
        "".join(json.dumps(f, ensure_ascii=False) + "\n" for f in filas), encoding="utf-8"
    )
    pre = leer(DATA_DIR / "labels" / "gold_pre.jsonl")
    modelo = next(iter(pre.values()))["fuente"].removeprefix("api:")
    (PREDS / f"{modelo}.jsonl").write_text(
        "".join(json.dumps(pre[i], ensure_ascii=False) + "\n" for i in gold), encoding="utf-8"
    )
    print(f"{len(filas)} predicciones de reglas y {len(gold)} de {modelo} en {PREDS}")


if __name__ == "__main__":
    main()
