"""Compara predicciones con el gold: F1 por campo, JSON válido, alucinaciones y latencia.

Uso: uv run --group exp boe-eval data/preds/reglas.jsonl   → tabla + ejecución en MLflow

Predicciones: JSONL con {id, json, valido, latencia_ms?, tokens_in?, tokens_out?}.
Cada campo se aplana (plazo_solicitudes.dias, …) y se cuenta como en extracción de información:
- gold y predicción null: no cuenta.  - Solo predicción: FP.  - Solo gold: FN.
- Ambos: TP += similitud, FP y FN += 1 - similitud. La similitud es 1/0 salvo en los campos de
  texto libre, donde es el F1 de tokens (organismo, objeto…).
- Listas (cpv, plazas): conjuntos; cada elemento es un TP, FP o FN.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import get_args, get_origin

from boe_extractor.fetch import DATA_DIR

GOLD_FINAL = DATA_DIR / "gold" / "gold_final.jsonl"
TEXTO = {
    "organismo", "puesto", "titulacion", "boletin_bases", "objeto", "beneficiarios",
    "bases_reguladoras", "organo_contratacion",
}  # fmt: skip
LISTAS = {"cpv", "plazas"}
# Campos cuyo valor debería poder encontrarse literalmente en el texto (para alucinaciones)
LITERALES = TEXTO | {"cpv", "lugar_ejecucion_nuts", "bdns_id"}
IMPORTES = {"tasa_eur", "cuantia_total_eur", "importe_max_beneficiario_eur",
            "valor_estimado_eur", "presupuesto_base_eur"}  # fmt: skip


def tokens(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", str(s).casefold())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.findall(r"\w+", s)


def f1_tokens(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    comunes = sum(min(ta.count(t), tb.count(t)) for t in set(ta))
    if not comunes:
        return 0.0
    p, r = comunes / len(tb), comunes / len(ta)
    return 2 * p * r / (p + r)


def aplanar(d: dict | None, campos: list[str]) -> dict:
    """{'plazo_solicitudes': {'dias': 5}} → {'plazo_solicitudes.dias': 5}; None → todo null."""
    out = {}
    for campo in campos:
        padre, _, hijo = campo.partition(".")
        v = (d or {}).get(padre)
        out[campo] = (v or {}).get(hijo) if hijo else v
    return out


def campos_de(tipo: str) -> list[str]:
    """Campos del esquema, con los submodelos aplanados: plazo_solicitudes.dias, …"""
    from pydantic import BaseModel

    from boe_extractor.schemas import ESQUEMAS

    campos = []
    for k, f in ESQUEMAS[tipo].model_fields.items():
        args = [] if get_origin(f.annotation) is list else get_args(f.annotation)
        sub = [a for a in args if isinstance(a, type) and issubclass(a, BaseModel)]
        campos += [f"{k}.{h}" for h in sub[0].model_fields] if sub else [k]
    return campos


def similitud(campo: str, g, p) -> float:
    if campo in TEXTO:
        return f1_tokens(g, p)
    if isinstance(g, float | int) and isinstance(p, float | int):
        return float(abs(g - p) <= 0.01)
    return float(g == p)


def contar(campo: str, g, p) -> tuple[float, float, float]:
    """(TP, FP, FN) de un campo en un documento."""
    if campo.split(".")[0] in LISTAS:
        clave = lambda x: json.dumps(x, sort_keys=True)  # noqa: E731
        sg, sp = {clave(x) for x in g or []}, {clave(x) for x in p or []}
        return len(sg & sp), len(sp - sg), len(sg - sp)
    if g is None and p is None:
        return 0, 0, 0
    if g is None:
        return 0, 1, 0
    if p is None:
        return 0, 0, 1
    s = similitud(campo, g, p)
    return s, 1 - s, 1 - s


def f1(tp: float, fp: float, fn: float) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0


def _cubierto(valor: str, tokens_texto: set[str]) -> bool:
    """Al menos la mitad de los tokens del valor están en el texto."""
    tv = tokens(valor)
    return sum(t in tokens_texto for t in tv) >= len(tv) / 2


def alucinado(campo: str, v, texto: str) -> bool | None:
    """True si el valor no aparece en el texto. None si no se puede comprobar (enums, días…).

    ponytail: búsqueda literal; un valor deducido (p. ej. una suma) cuenta como alucinado.
    """
    base = campo.split(".")[0]
    if v is None or v == []:
        return None
    if base in LITERALES:
        tt = set(tokens(texto))
        return any(not _cubierto(x, tt) for x in (v if isinstance(v, list) else [v]))
    if base in IMPORTES:
        entero, dec = f"{v:.2f}".split(".")
        con_puntos = f"{int(entero):,}".replace(",", ".")
        return not any(f in texto for f in (con_puntos, entero)) or (
            dec != "00" and f",{dec}" not in texto
        )
    if campo.endswith("fecha_fin") or base == "fecha_limite_ofertas":
        from boe_extractor.normalize import MESES

        a, m, d = v.split("-")
        mes = [k for k, i in MESES.items() if i == int(m)][0]
        return not re.search(rf"\b0?{int(d)} de {mes} de {a}|{int(d)}/{m}/{a}", texto, re.I)
    return None


def evaluar(gold: dict[str, dict], preds: dict[str, dict], textos: dict[str, str]) -> dict:
    """Métricas globales y por campo. Un id sin predicción cuenta como JSON inválido."""
    cont: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    exactos: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    aluc = [0, 0]
    for i, g in gold.items():
        pred = preds.get(i, {})
        campos = campos_de(g["tipo"])
        pg = aplanar(g["json"], campos)
        pp = aplanar(pred.get("json") if pred.get("valido") else None, campos)
        for campo in campos:
            clave = f"{g['tipo']}/{campo}"
            for k, x in enumerate(contar(campo, pg[campo], pp[campo])):
                cont[clave][k] += x
            exactos[clave][0] += contar(campo, pg[campo], pp[campo])[1:] == (0, 0)
            exactos[clave][1] += 1
            a = alucinado(campo, pp[campo], textos.get(i, ""))
            if a is not None:
                aluc[0] += a
                aluc[1] += 1
    por_campo = {
        c: {"f1": f1(*v), "exacto": exactos[c][0] / exactos[c][1], "tp_fp_fn": v}
        for c, v in sorted(cont.items())
    }
    total = [sum(v[k] for v in cont.values()) for k in range(3)]
    lat = [p["latencia_ms"] for p in preds.values() if p.get("latencia_ms") is not None]
    return {
        "n_docs": len(gold),
        "f1_micro": f1(*total),
        "f1_macro": statistics.mean(c["f1"] for c in por_campo.values()),
        "json_valido": sum(bool(preds.get(i, {}).get("valido")) for i in gold) / len(gold),
        "alucinacion": aluc[0] / aluc[1] if aluc[1] else 0.0,
        "latencia_p50_ms": statistics.median(lat) if lat else None,
        "latencia_p95_ms": statistics.quantiles(lat, n=20)[18] if len(lat) > 1 else None,
        "por_campo": por_campo,
    }


def leer(path: Path) -> dict[str, dict]:
    filas = (json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip())
    return {f["id"]: f for f in filas}


def tabla(m: dict) -> str:
    filas = [f"| {c} | {v['f1']:.3f} | {v['exacto']:.3f} |" for c, v in m["por_campo"].items()]
    return "\n".join(["| campo | F1 | exacto |", "|---|---|---|", *filas])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", type=Path)
    ap.add_argument("--sistema", help="nombre en MLflow (por defecto, el del fichero)")
    ap.add_argument("--historial", type=Path, help="historial.json del QLoRA, para MLflow")
    args = ap.parse_args()
    import mlflow
    import polars as pl

    from boe_extractor.docs import recortar
    from boe_extractor.schemas import SCHEMA_VERSION

    gold, preds = leer(GOLD_FINAL), leer(args.preds)
    docs = pl.read_parquet(DATA_DIR / "documents.parquet").filter(pl.col("id").is_in(list(gold)))
    textos = dict(zip(docs["id"], recortar(docs["texto"].to_list()), strict=True))
    sistema = args.sistema or args.preds.stem
    m = evaluar(gold, preds, textos)
    cero = evaluar({i: g for i, g in gold.items() if not g["prellenado"]}, preds, textos)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")  # en la raíz del repo, fuera de git
    mlflow.set_experiment("boe-extractor")
    with mlflow.start_run(run_name=sistema):
        mlflow.log_params(
            {"sistema": sistema, "schema_version": SCHEMA_VERSION, "n_docs": len(gold)}
        )
        mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, float)})
        mlflow.log_metric("f1_micro_desde_cero", cero["f1_micro"])
        for tipo in ("convocatoria", "ayuda", "licitacion"):
            sub = {i: g for i, g in gold.items() if g["tipo"] == tipo}
            mlflow.log_metric(f"f1_micro_{tipo}", evaluar(sub, preds, textos)["f1_micro"])
        mlflow.log_metrics(
            {f"f1.{c.replace('/', '.')}": v["f1"] for c, v in m["por_campo"].items()}
        )
        mlflow.log_text(tabla(m), "por_campo.md")
        if args.historial:
            h = json.loads(args.historial.read_text())
            mlflow.log_params(h["hiper"])
            for x in h["log"]:
                for k in ("loss", "eval_loss"):
                    if k in x:
                        mlflow.log_metric(f"train.{k}", x[k], step=x["step"])
    print(tabla(m))
    resumen = {k: round(v, 3) for k, v in m.items() if isinstance(v, float)}
    print(f"\n{sistema}: {resumen} · F1 micro en los {cero['n_docs']} anotados desde cero: "
          f"{cero['f1_micro']:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
