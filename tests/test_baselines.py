from pathlib import Path

from boe_extractor.baselines import organismo, reglas
from boe_extractor.docs import parsear_documento

FIX = Path(__file__).parent / "fixtures"


def doc(fichero, tipo, subtipo, titulo="", departamento=""):
    texto = parsear_documento((FIX / fichero).read_bytes())["texto"]
    d = {
        "id": fichero,
        "tipo": tipo,
        "subtipo": subtipo,
        "titulo": titulo,
        "departamento": departamento,
    }
    return d, texto


def test_organismo_del_titulo():
    t = "Resolución de 4 de agosto de 2026, del Ayuntamiento de Palomares del Río (Sevilla), referente a la convocatoria"
    assert organismo(t, "ADMINISTRACIÓN LOCAL") == "Ayuntamiento de Palomares del Río"
    assert organismo("Orden PJC/908/2026", "MINISTERIO DE CULTURA") == "Ministerio De Cultura"


def test_reglas_licitacion():
    j = reglas(*doc("BOE-B-2026-30422.xml", "licitacion", None))["json"]
    assert j["organo_contratacion"].startswith("Jefatura de la Sección Económico")
    assert j["lugar_ejecucion_nuts"] == "ES300"
    assert j["cpv"] == ["55300000"]
    assert j["fecha_limite_ofertas"] == "2026-10-15"


def test_reglas_anuncio_local():
    j = reglas(*doc("BOE-A-2026-19568.xml", "convocatoria", "anuncio_local"))["json"]
    assert j["boletin_bases"] == "Boletín Oficial de la Provincia de Valencia"
    assert j["sistema"] == "concurso"
    assert (
        j["plazo_solicitudes"]["dias"] == 15
        and j["plazo_solicitudes"]["desde"] == "publicacion_boe"
    )


def test_reglas_no_revientan_con_texto_vacio():
    d = {
        "id": "x",
        "tipo": "licitacion",
        "subtipo": None,
        "titulo": "Anuncio de licitación",
        "departamento": "MINISTERIO DE DEFENSA",
    }
    r = reglas(d, "")
    assert r["valido"] and r["json"]["organo_contratacion"] == "Ministerio De Defensa"
