# boe-extractor

Del BOE a JSON validado con un LLM pequeño ajustado con QLoRA.

> Estado: semana 4 (iteración, QLoRA v2). En construcción.

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

## Resultados (gold, 262 documentos de ago–sep)

F1 micro sin los campos de texto libre (objeto, beneficiarios), que van en su propia columna.

| Sistema | F1 micro | F1 macro | F1 micro, 30 docs anotados desde cero | Texto libre | JSON válido | Alucinación |
|---|---|---|---|---|---|---|
| Reglas (regex) | 0,840 | 0,709 | 0,852 | 0,650 | 100 % | 1,0 % |
| Qwen2.5-1.5B base + decodificación restringida | 0,697 | 0,632 | 0,663 | 0,717 | 99,2 % | 12,2 % |
| Qwen2.5-1.5B + QLoRA v1 | 0,930 | 0,863 | 0,864 | 0,832 | 100 % | 0,9 % |
| **Qwen2.5-1.5B + QLoRA v2** | **0,947** | **0,882** | **0,900** | 0,844 | **100 %** | **0,9 %** |
| Gemini 3.5 Flash-Lite (API) | 0,963* | 0,960* | 0,910 | 0,964* | 100 % | 0,9 % |

\* El gold se pre-rellenó con Gemini y eso infla su nota; la columna de los 30 documentos anotados
desde cero es la comparación justa. v1: r=16, 1 época sobre ~1.400 etiquetas silver (~50 min en una
T4 gratis). v2: mismas etiquetas pasadas por las convenciones del gold (`label/convenciones.py`) y
2 épocas. Latencia no comparable todavía (vLLM en lote frente a API secuencial).
