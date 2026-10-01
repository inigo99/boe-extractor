"""Casos reales de títulos del BOE (abr–sep 2026) para las reglas de clasify.py."""

import polars as pl
import pytest

from boe_extractor.classify import clasificar

CASOS = [
    # (seccion, epigrafe, titulo, tipo, subtipo)
    (
        "2B",
        "Personal funcionario y laboral",
        "Resolución de 14 de mayo de 2026, del Ayuntamiento de Tarragona, referente a la convocatoria "
        "para proveer varias plazas.",
        "convocatoria",
        "anuncio_local",
    ),
    (
        "2B",
        "Personal funcionario",
        "Resolución de 2 de junio de 2026, de la Subsecretaría, por la que se convoca proceso selectivo "
        "para ingreso en el Cuerpo de Gestión de la Administración Civil del Estado.",
        "convocatoria",
        "bases",
    ),
    (
        "2B",
        "Personal funcionario",
        "Resolución de 3 de julio de 2026, por la que se aprueba la relación de aspirantes admitidos y "
        "excluidos del proceso selectivo convocado por Resolución de 2 de junio.",
        None,
        None,
    ),
    (
        "2B",
        "Procedimientos de libre designación",
        "Resolución de 1 de julio de 2026, por la que se convoca la provisión de puesto de trabajo por "
        "el sistema de libre designación.",
        None,
        None,
    ),
    (
        "2B",
        "Personal funcionario y laboral",
        "Corrección de errores de la Resolución de 5 de mayo de 2026, del Ayuntamiento de Berga, "
        "referente a la convocatoria para proveer una plaza.",
        None,
        None,
    ),
    (
        "5B",
        None,
        "Extracto de la Orden, de 7 de abril de 2026, del Ministerio de Derechos Sociales, Consumo y "
        "Agenda 2030 por la que se convocan subvenciones.",
        "ayuda",
        None,
    ),
    (
        "5B",
        None,
        "Anuncio de la Delegación de Economía y Hacienda sobre subasta de bienes.",
        None,
        None,
    ),
    (
        "3",
        "Convenios",
        "Resolución de 14 de septiembre de 2026, por la que se publica el Convenio entre la Embajada de "
        "España en Costa Rica y Viscofan Centroamérica Comercial, SA.",
        None,
        None,
    ),
    (
        "5A",
        None,
        "Anuncio de licitación de: Mesa del Senado. Objeto: Servicio de limpieza. Expediente: 07/2026.",
        "licitacion",
        None,
    ),
    (
        "5A",
        None,
        "Anuncio de formalización de contratos de: Mesa del Senado. Objeto: Servicios de gestión y "
        "asistencia técnica de viajes y visitas del Senado. Expediente: 06/2026.",
        None,
        None,
    ),
    (
        "2B",
        "Personal funcionario",
        "Resolución de 19 de mayo de 2026, de la Dirección General de Justicia, de la Consellería de Presidencia, Justicia y Deportes, por la que se convoca concurso de traslado.",
        "convocatoria",
        "bases",
    ),
    (
        "3",
        "Premios",
        "Orden CLT/861/2026, de 10 de agosto, por la que se modifica la Orden CLT/451/2026, de 29 de abril, por la que se convocan los Premios Nacionales de Música.",
        None,
        None,
    ),
    (
        "2B",
        "Personal funcionario",
        "Orden de 29 de mayo de 2026, del Departamento de Justicia y Derechos Humanos, por la que se convoca la provisión de puesto de trabajo, por el sistema de libre designación, en el Servicio Común de Tramitación.",
        None,
        None,
    ),
]


@pytest.mark.parametrize("seccion,epigrafe,titulo,tipo,subtipo", CASOS)
def test_clasificar(seccion, epigrafe, titulo, tipo, subtipo):
    df = pl.DataFrame(
        {"seccion": [seccion], "epigrafe": [epigrafe], "titulo": [titulo]},
        schema={"seccion": pl.String, "epigrafe": pl.String, "titulo": pl.String},
    )
    out = clasificar(df)
    if tipo is None:
        assert out.height == 0
    else:
        assert out.row(0, named=True)["tipo"] == tipo
        assert out.row(0, named=True)["subtipo"] == subtipo
