"""Esquemas de salida (fuente única de verdad del JSON). Todo opcional salvo lo que siempre aparece.

Importes en euros como float; plazos con Plazo. Los Literal fijan el vocabulario para que la
decodificación restringida no pueda inventar valores.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "v1"


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Plazo(_Base):
    dias: int | None = Field(None, description="Número de días del plazo")
    tipo: Literal["habiles", "naturales"] | None = None
    desde: (
        Literal[
            "publicacion_boe",
            "publicacion_diario_autonomico",
            "publicacion_boletin_provincial",
            "otro",
        ]
        | None
    ) = Field(None, description="Desde qué publicación se cuenta el plazo")
    fecha_fin: date | None = Field(None, description="Fecha límite si el texto la da explícita")


class PlazasTurno(_Base):
    turno: Literal["libre", "promocion_interna", "discapacidad", "otro"]
    numero: int


class Convocatoria(_Base):
    organismo: str = Field(description="Administración u organismo que convoca")
    puesto: str | None = Field(
        None, description="Cuerpo, escala, categoría o denominación de la plaza"
    )
    vinculo: (
        Literal["funcionario", "laboral", "estatutario", "docente_universitario", "otro"] | None
    ) = None
    plazas: list[PlazasTurno] = Field(default_factory=list)
    sistema: Literal["oposicion", "concurso", "concurso_oposicion"] | None = None
    plazo_solicitudes: Plazo | None = None
    titulacion: str | None = Field(None, description="Titulación exigida")
    tasa_eur: float | None = Field(None, description="Derechos de examen")
    boletin_bases: str | None = Field(
        None, description="Boletín donde se publicaron las bases, si es otro"
    )


class Ayuda(_Base):
    organismo: str = Field(description="Órgano convocante")
    objeto: str | None = Field(None, description="Línea u objeto de la ayuda, en una frase")
    beneficiarios: str | None = None
    cuantia_total_eur: float | None = None
    importe_max_beneficiario_eur: float | None = None
    plazo_solicitudes: Plazo | None = None
    regimen: Literal["concurrencia_competitiva", "concesion_directa", "otro"] | None = None
    bdns_id: str | None = Field(None, description="Identificador BDNS")
    bases_reguladoras: str | None = Field(None, description="Norma con las bases reguladoras")


class Licitacion(_Base):
    organo_contratacion: str
    objeto: str
    tipo_contrato: (
        Literal[
            "obras",
            "servicios",
            "suministros",
            "concesion_servicios",
            "concesion_obras",
            "mixto",
            "otro",
        ]
        | None
    ) = None
    procedimiento: (
        Literal[
            "abierto",
            "abierto_simplificado",
            "restringido",
            "negociado",
            "dialogo_competitivo",
            "otro",
        ]
        | None
    ) = None
    valor_estimado_eur: float | None = None
    presupuesto_base_eur: float | None = None
    fecha_limite_ofertas: date | None = None
    lugar_ejecucion_nuts: str | None = None
    cpv: list[str] = Field(default_factory=list, description="Códigos CPV de 8 dígitos")
    duracion_meses: int | None = None
    lotes: int | None = Field(None, description="Número de lotes, si hay más de uno")


ESQUEMAS: dict[str, type[_Base]] = {
    "convocatoria": Convocatoria,
    "ayuda": Ayuda,
    "licitacion": Licitacion,
}
