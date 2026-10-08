"""Convenciones de etiquetado que se aplican igual al silver (entrenamiento) y al gold.

Gemini etiqueta con sus propios hábitos; el gold sigue reglas acordadas en la revisión. Sin
alinear, el modelo aprende los hábitos de Gemini y el gold se los penaliza. Reglas:
- ayuda.regimen: null si el texto no menciona concurrencia ni concesión directa.
- ayuda.bases_reguladoras: forma corta de la cita ('Orden X/1/2024, de 1 de abril').
- convocatoria.plazas: una entrada por turno (se suman); en concursos de provisión entre
  funcionarios, turno libre → otro.
- convocatoria.puesto: null en anuncios locales con varios puestos distintos.
- plazo «de un mes»: 30 días naturales (Gemini a veces pone 1).
- plazo con solo fecha final: desde = otro si arranca en una fecha fija, publicacion_boe si
  arranca con la publicación y null si el texto no dice cuándo empieza.
"""

from __future__ import annotations

import copy
import re

from boe_extractor.normalize import MESES

_NORMA = r"^(?:Orden|Real Decreto|Resolución|Decreto|Ley|Acuerdo)\b"
_CITA = re.compile(r"^(.*?\b\d{1,2} de [a-záéíóú]+(?: de \d{4})?)", re.I)
_PROV_TITULO = re.compile(
    r"provisi[oó]n de(l)? puestos?|concurso (espec[ií]fico|general|de m[eé]ritos|de traslados"
    r"|ordinario)|provisi[oó]n de (destinos|vacantes)|para (la )?provisi[oó]n|se convoca vacante"
    r"|para proveer puestos?",
    re.I,
)
_PROV_TEXTO = re.compile(
    r"proveer,? (mediante|por) (el sistema de )?concurso|libre designaci|turno de movilidad"
    r"|provisi[oó]n (del|de) puesto|concurso de m[eé]ritos",
    re.I,
)
_ACCESO = re.compile(r"acceso|ingreso|pruebas selectivas|oposici|estabilizaci", re.I)
_LIBRE = re.compile(r"turno libre|acceso libre", re.I)
_PLAZO = re.compile(r"plazo (?:de|para la) presentaci[oó]n", re.I)
# Arranque en fecha fija: 'se iniciará el día 10 de noviembre', 'entre el 5 de octubre y…'
_INICIO_FIJO = re.compile(
    r"(inici|comienz|comenz|abr|desde|entre|a partir)[^.]{0,40}?\b\d{1,2} de [a-z]+", re.I
)
_PUESTO = re.compile(
    r"^[\wáéíóú]+ (?:plazas?|puestos?) de ([^,.]+?)(?:,| pertenecientes?| por el)", re.I | re.M
)


def es_provision(titulo: str, texto: str) -> bool:
    """Concurso para cubrir puestos entre funcionarios, no acceso a la función pública."""
    if _ACCESO.search(titulo):
        return False
    return bool(_PROV_TITULO.search(titulo)) or (
        bool(_PROV_TEXTO.search(texto)) and not _LIBRE.search(texto)
    )


def _plazo(p: dict | None, texto: str) -> None:
    if not p:
        return
    m = _PLAZO.search(texto)
    ventana = texto[m.end() : m.end() + 400] if m else ""
    if re.search(r"\bun mes\b", ventana, re.I) and p["dias"] in (1, None) and not p["fecha_fin"]:
        p.update(dias=30, tipo="naturales")
    if p["fecha_fin"] and p["dias"] is None:
        a, mes, d = p["fecha_fin"].split("-")
        nombre = next(k for k, v in MESES.items() if v == int(mes))
        f = re.search(rf"\b0?{int(d)} de {nombre} de {a}", texto, re.I) or re.search(
            rf"\b0?{int(d)} de {nombre}", texto, re.I
        )
        antes = texto[max(0, f.start() - 300) : f.start()] if f else ""
        if _INICIO_FIJO.search(antes):
            p["desde"] = "otro"
        elif "publicaci" in antes.lower():
            p["desde"] = "publicacion_boe"
        else:
            p["desde"] = None


def _sumar_turnos(plazas: list[dict]) -> list[dict]:
    total: dict[str, int] = {}
    for p in plazas:
        total[p["turno"]] = total.get(p["turno"], 0) + p["numero"]
    return [{"turno": t, "numero": n} for t, n in total.items()]


def alinear(tipo: str, datos: dict, titulo: str, texto: str) -> dict:
    datos = copy.deepcopy(datos)
    if tipo in ("ayuda", "convocatoria"):
        _plazo(datos.get("plazo_solicitudes"), texto)
    if tipo == "ayuda":
        if not re.search(r"concurrencia|(concesi[oó]n|adjudicaci[oó]n) directa", texto, re.I):
            datos["regimen"] = None
        bases = datos.get("bases_reguladoras")
        if bases and re.match(_NORMA, bases) and (m := _CITA.match(bases)):
            datos["bases_reguladoras"] = m.group(1)
    if tipo == "convocatoria":
        if es_provision(titulo, texto):
            for p in datos["plazas"]:
                if p["turno"] == "libre":
                    p["turno"] = "otro"
        datos["plazas"] = _sumar_turnos(datos["plazas"])
        puestos = {m.strip().lower() for m in _PUESTO.findall(texto[:3000])}
        if len(puestos) > 1:
            datos["puesto"] = None
    return datos
