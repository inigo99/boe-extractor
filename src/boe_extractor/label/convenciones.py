"""Convenciones de etiquetado que se aplican igual al silver (entrenamiento) y al gold.

Gemini etiqueta con sus propios hábitos; el gold sigue reglas acordadas en la revisión. Sin
alinear, el modelo aprende los hábitos de Gemini y el gold se los penaliza. Reglas:
- ayuda.regimen: null si el texto no menciona concurrencia ni concesión directa.
- ayuda.bases_reguladoras: forma corta de la cita ('Orden X/1/2024, de 1 de abril').
- convocatoria.plazas: en concursos de provisión entre funcionarios, turno libre → otro.
"""

from __future__ import annotations

import copy
import re

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


def es_provision(titulo: str, texto: str) -> bool:
    """Concurso para cubrir puestos entre funcionarios, no acceso a la función pública."""
    if _ACCESO.search(titulo):
        return False
    return bool(_PROV_TITULO.search(titulo)) or (
        bool(_PROV_TEXTO.search(texto)) and not _LIBRE.search(texto)
    )


def alinear(tipo: str, datos: dict, titulo: str, texto: str) -> dict:
    datos = copy.deepcopy(datos)
    if tipo == "ayuda":
        if not re.search(r"concurrencia|(concesi[oó]n|adjudicaci[oó]n) directa", texto, re.I):
            datos["regimen"] = None
        bases = datos.get("bases_reguladoras")
        if bases and re.match(_NORMA, bases) and (m := _CITA.match(bases)):
            datos["bases_reguladoras"] = m.group(1)
    if tipo == "convocatoria" and es_provision(titulo, texto):
        for p in datos["plazas"]:
            if p["turno"] == "libre":
                p["turno"] = "otro"
    return datos
