"""Del texto del BOE a valores canónicos: importes, días de plazo y fechas en español."""

from __future__ import annotations

import re
from datetime import date

from boe_extractor.schemas import Plazo

MESES = {
    m: i
    for i, m in enumerate(
        [
            "enero",
            "febrero",
            "marzo",
            "abril",
            "mayo",
            "junio",
            "julio",
            "agosto",
            "septiembre",
            "octubre",
            "noviembre",
            "diciembre",
        ],  # fmt: skip
        1,
    )
}
_UNIDADES = {
    "un": 1, "uno": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12, "trece": 13,
    "catorce": 14, "quince": 15, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18,
    "diecinueve": 19, "veinte": 20, "veintiuno": 21, "veintidós": 22, "veintitrés": 23,
    "veinticuatro": 24, "veinticinco": 25, "veintiséis": 26, "veintisiete": 27,
    "veintiocho": 28, "veintinueve": 29,
}  # fmt: skip
_DECENAS = {"treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70,
            "ochenta": 80, "noventa": 90}  # fmt: skip


def numero(texto: str) -> int | None:
    """'20', 'veinte', 'cuarenta y cinco' → int. ponytail: hasta 99, basta para días y plazas."""
    t = texto.strip().lower()
    if t.isdigit():
        return int(t)
    if t in _UNIDADES:
        return _UNIDADES[t]
    partes = t.split(" y ")
    if partes[0] in _DECENAS:
        resto = _UNIDADES.get(partes[1]) if len(partes) == 2 else 0
        return None if resto is None else _DECENAS[partes[0]] + resto
    return None


_IMPORTE = re.compile(r"(\d{1,3}(?:\.\d{3})+|\d+)(?:,(\d{1,2}))?\s*(?:de\s+)?(?:€|euros?)", re.I)


def importe(texto: str) -> float | None:
    """Primer importe en formato español: '1.766.925,56 euros' → 1766925.56."""
    m = _IMPORTE.search(texto)
    if not m:
        return None
    return float(m.group(1).replace(".", "") + "." + (m.group(2) or "0"))


_FECHA = re.compile(r"(\d{1,2}) de (" + "|".join(MESES) + r") de (\d{4})", re.I)


def fecha(texto: str) -> date | None:
    m = _FECHA.search(texto)
    return date(int(m.group(3)), MESES[m.group(2).lower()], int(m.group(1))) if m else None


_NUM = r"(\d+|[a-záéíóúñ]+(?: y [a-záéíóúñ]+)?)"
_PLAZO = re.compile(_NUM + r" días? (hábiles|naturales)", re.I)


def plazo(texto: str) -> Plazo | None:
    """'veinte días hábiles a contar desde ... el «Boletín Oficial del Estado»' → Plazo."""
    m = _PLAZO.search(texto)
    if not m:
        return None
    resto = texto[m.end() : m.end() + 250].lower()
    if "boletín oficial del estado" in resto or "presente publicación" in resto:
        desde = "publicacion_boe"
    elif "provincia" in resto:
        desde = "publicacion_boletin_provincial"
    elif re.search(
        r"diario oficial|boletín oficial de (la|las|el) "
        r"(comunidad|junta|región|generalitat|canarias|aragón|navarra)",
        resto,
    ):
        desde = "publicacion_diario_autonomico"
    else:
        desde = None
    return Plazo(
        dias=numero(m.group(1)),
        tipo="habiles" if m.group(2).lower() == "hábiles" else "naturales",
        desde=desde,
    )
