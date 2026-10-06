# boe-extractor

Del BOE a JSON validado con un LLM pequeño ajustado con QLoRA.

> Estado: semana 3 terminada (QLoRA v1). En construcción.

## Uso

```bash
uv sync
uv run boe-fetch 20260929                # un día
uv run boe-fetch 20260401 20260930       # un rango (1 petición/s, con caché)
uv run pytest
```

Cada día se guarda en `data/sumarios/{aaaammdd}.parquet` (una fila por documento, sin la sección II.A).
El XML original queda en `data/raw/sumarios/` como caché; repetir el comando no vuelve a descargar nada.

## Datos

Fuente: [API de datos abiertos del BOE](https://www.boe.es/datosabiertos/). Reutilización sujeta a sus
[condiciones](https://www.boe.es/informacion/aviso_legal/index.php#reutilizacion). El dataset derivado se
publicará con licencia CC BY 4.0 citando al BOE como fuente.

## Etiquetas

- **Silver** (entrenamiento, abr–jul): ~1.500 documentos etiquetados por Gemini 3.5 Flash-Lite con salida
  forzada al esquema Pydantic (`uv run boe-silver`).
- **Gold** (evaluación, ago–sep): 262 documentos. Etiquetas de Gemini revisadas por Claude contra el texto;
  los desacuerdos (44) los decidió un humano. 30 documentos se anotaron a mano desde cero para medir el sesgo
  del pre-rellenado. `uv run boe-gold-final` aplica `data/gold/revision.jsonl` y genera
  `data/gold/gold_final.jsonl`, con las correcciones que lleva cada documento.

## Evaluación

```bash
uv run boe-gold-final                               # gold revisado
uv run boe-baseline                                 # reglas y Gemini → data/preds/
uv run --group exp boe-eval data/preds/reglas.jsonl # métricas + MLflow (mlflow.db)
uv run --group exp mlflow ui --backend-store-uri sqlite:///mlflow.db
uv run boe-sft                                      # conversaciones para Colab → data/sft/
```

Modelo base y QLoRA: `notebooks/colab.ipynb` (T4 gratis; vLLM con decodificación restringida al
esquema y Unsloth). Métricas: F1 por campo (similitud de tokens en texto libre), F1 micro/macro,
% de JSON válido, tasa de alucinación (valores que no aparecen en el texto) y latencia.

## Resultados v1 (gold, 262 documentos de ago–sep)

| Sistema | F1 micro | F1 macro | F1 micro, 30 docs anotados desde cero | JSON válido | Alucinación |
|---|---|---|---|---|---|
| Reglas (regex) | 0,819 | 0,696 | 0,818 | 100 % | 1,0 % |
| Qwen2.5-1.5B base + decodificación restringida | 0,701 | 0,630 | 0,662 | 99,2 % | 12,2 % |
| **Qwen2.5-1.5B + QLoRA v1** | **0,921** | **0,858** | **0,848** | **100 %** | **0,9 %** |
| Gemini 3.5 Flash-Lite (API) | 0,966* | 0,964* | 0,892 | 100 % | 0,9 % |

\* El gold se pre-rellenó con Gemini y eso infla su nota; la columna de los 30 documentos anotados
desde cero es la comparación justa. QLoRA v1: r=16, 1 época sobre ~1.400 etiquetas silver, ~50 min en
una T4 gratis de Colab. Latencia no comparable todavía (vLLM en lote frente a API secuencial).
