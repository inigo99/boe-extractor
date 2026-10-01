"""Normalizadores con frases reales del BOE y esquemas validados con ejemplos anotados a mano."""

from datetime import date

import pytest
from pydantic import ValidationError

from boe_extractor import normalize as n
from boe_extractor.schemas import ESQUEMAS, Ayuda, Convocatoria, Licitacion, Plazo


@pytest.mark.parametrize(
    "texto,esperado",
    [
        ("8. Valor estimado:\n1.766.925,56 euros.", 1766925.56),
        ("asciende a doce mil euros (12.000 €), sujeta a", 12000.0),
        ("asciende a 2.500.000 de euros, con cargo", 2500000.0),
        ("235.901,56 euros.", 235901.56),
        ("tasa de 30,70 euros", 30.70),
    ],
)
def test_importe(texto, esperado):
    assert n.importe(texto) == esperado


@pytest.mark.parametrize(
    "texto,dias,tipo,desde",
    [
        (
            "El plazo de presentación de solicitudes será de veinte días hábiles a contar desde el siguiente al de la publicación de esta resolución en el «Boletín Oficial del Estado».",
            20,
            "habiles",
            "publicacion_boe",
        ),
        (
            "finalizará en el plazo de treinta días hábiles contados a partir del día siguiente al de la presente publicación.",
            30,
            "habiles",
            "publicacion_boe",
        ),
        (
            "en el plazo de 15 días naturales desde la publicación en el Boletín Oficial de la Provincia",
            15,
            "naturales",
            "publicacion_boletin_provincial",
        ),
        ("plazo de cuarenta y cinco días naturales", 45, "naturales", None),
    ],
)
def test_plazo(texto, dias, tipo, desde):
    assert n.plazo(texto) == Plazo(dias=dias, tipo=tipo, desde=desde)


def test_fecha_y_numero():
    assert n.fecha("Hasta las 23:00 horas del 15 de octubre de 2026.") == date(2026, 10, 15)
    assert n.numero("Seis") == 6 and n.numero("treinta y uno") == 31 and n.numero("mil") is None


# --- Ejemplos reales anotados a mano (resumen de los documentos) ---


def test_convocatoria_anuncio_local_BOE_A_2026_18400():
    # "Una plaza de Personal de Guardia, Vigilancia y Portería ... escala de Administración General,
    # subescala Subalterna, por el sistema de concurso-oposición, en turno reservado a personas con
    # discapacidad ... diez días hábiles ... «Boletín Oficial del Estado»"
    c = Convocatoria(
        organismo="Ayuntamiento de Sevilla",
        puesto="Personal de Guardia, Vigilancia y Portería (Administración General, subescala Subalterna)",
        vinculo="funcionario",
        plazas=[{"turno": "discapacidad", "numero": 1}],
        sistema="concurso_oposicion",
        plazo_solicitudes={"dias": 10, "tipo": "habiles", "desde": "publicacion_boe"},
        boletin_bases="Boletín Oficial de la Provincia de Sevilla núm. 99, de 26 de mayo de 2026",
    )
    assert c.plazas[0].numero == 1


def test_ayuda_BOE_B_2026_26053():
    a = Ayuda(
        organismo="Ministerio de Defensa",
        objeto="Premios Sanidad Militar 2026",
        beneficiarios="Personas físicas y jurídicas, españolas o extranjeras",
        cuantia_total_eur=12000,
        plazo_solicitudes={"dias": 30, "tipo": "habiles", "desde": "publicacion_boe"},
        regimen="concurrencia_competitiva",
        bdns_id="923073",
        bases_reguladoras="Orden DEF/410/2025, de 15 de abril",
    )
    assert a.cuantia_total_eur == 12000


def test_licitacion_BOE_B_2026_20537():
    lic = Licitacion(
        organo_contratacion="Presidencia de la Confederación Hidrográfica del Ebro",
        objeto="Asistencia técnica para la redacción y actualización de proyectos y apoyo técnico al área de Proyectos y Obras I",
        tipo_contrato="servicios",
        procedimiento="abierto",
        valor_estimado_eur=1766925.56,
        lugar_ejecucion_nuts="ES243",
        cpv=["71310000"],
        duracion_meses=24,
        lotes=2,
    )
    assert lic.lotes == 2


def test_esquemas_rechazan_valores_fuera_de_vocabulario_y_campos_extra():
    with pytest.raises(ValidationError):
        Licitacion(organo_contratacion="X", objeto="Y", procedimiento="subasta")
    with pytest.raises(ValidationError):
        Ayuda(organismo="X", inventado=1)


def test_json_schema_generable_para_decodificacion_restringida():
    for esquema in ESQUEMAS.values():
        js = esquema.model_json_schema()
        assert js["additionalProperties"] is False
