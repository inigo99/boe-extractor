"""Clasifica el sumario en convocatoria / ayuda / licitacion con reglas sobre el título.

Uso: boe-classify  →  lee data/sumarios/*.parquet y escribe data/candidatos.parquet
"""

from __future__ import annotations

import polars as pl

from boe_extractor.fetch import DATA_DIR, OUT_DIR

_t = pl.col("titulo").str.to_lowercase()
_seccion = pl.col("seccion")

_no_es_original = _t.str.contains(
    r"corrección de errores|se corrigen errores|deja sin efecto|rectifica"
)
_convoca = _t.str.contains(r"convoca")

_convocatoria = (
    (_seccion == "2B")
    & _t.str.contains(r"por (?:la|el) que se convoca|referente a la convocatoria")
    # Actos posteriores: modificaciones, listas de aspirantes (con nombres), tribunales.
    & ~_t.str.contains(
        r"se modifica|se publica|relación|aspirantes|aprobad|admitid|excluid|tribunal|se resuelve"
    )
    # Libre designación: provisión de puestos entre funcionarios, no acceso al empleo público.
    & (pl.col("epigrafe") != "Procedimientos de libre designación")
)
_anuncio_local = _t.str.contains(
    r"ayuntamiento|diputació|consell (?:comarcal|insular)|cabildo|comarca|mancomunidad"
    r"|referente a la convocatoria"
)

# V.B: los extractos de la BDNS son siempre convocatorias de subvenciones (ayudas, premios,
# certámenes...), así que basta con descartar modificaciones y ampliaciones.
_ayuda_5b = (
    (_seccion == "5B")
    & _t.str.starts_with("extracto")
    & _convoca
    & ~_t.str.contains(
        r"se modifica|modificación del extracto|se amplía|ampliar el|acuerda la ampliación"
    )
)
_ayuda_3 = (
    (_seccion == "3")
    & _convoca
    & _t.str.contains(r"ayuda|subvenci|premio|beca")
    & ~_t.str.contains(r"se modifica|convenio|jurado|se publica|se autoriza|se establecen")
)
_ayuda = _ayuda_5b | _ayuda_3

_licitacion = (_seccion == "5A") & _t.str.starts_with("anuncio de licitación")

TIPO = (
    pl.when(_no_es_original)
    .then(None)
    .when(_convocatoria)
    .then(pl.lit("convocatoria"))
    .when(_ayuda)
    .then(pl.lit("ayuda"))
    .when(_licitacion)
    .then(pl.lit("licitacion"))
    .otherwise(None)
)
SUBTIPO = (
    pl.when(pl.col("tipo") == "convocatoria")
    .then(pl.when(_anuncio_local).then(pl.lit("anuncio_local")).otherwise(pl.lit("bases")))
    .otherwise(None)
)


def clasificar(df: pl.DataFrame) -> pl.DataFrame:
    """Añade `tipo` y `subtipo` y devuelve solo los candidatos."""
    return (
        df.with_columns(tipo=TIPO)
        .with_columns(subtipo=SUBTIPO)
        .filter(pl.col("tipo").is_not_null())
    )


def main() -> None:
    df = clasificar(pl.read_parquet(OUT_DIR / "*.parquet"))
    out = DATA_DIR / "candidatos.parquet"
    df.write_parquet(out)
    print(df.group_by("tipo", "subtipo").len().sort("tipo", "subtipo"))
    print(f"{df.height} candidatos → {out}")


if __name__ == "__main__":
    main()
