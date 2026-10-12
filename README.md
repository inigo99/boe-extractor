# boe-extractor

Del BOE a JSON validado con un LLM pequeño ajustado con QLoRA.

> Estado: semana 4 terminada (QLoRA v3 + GGUF para CPU). En construcción.

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

En CPU (modelo GGUF exportado con la parte 4 del notebook):

```bash
uv run --with llama-cpp-python==0.3.19 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu \
  boe-infer-cpu --modelo data/modelos/<modelo>.gguf --salida data/preds/<nombre>-cpu.jsonl --limite 30
uv run --group exp boe-eval data/preds/<nombre>-cpu.jsonl --solo-predichos
```

Modelo base y QLoRA: `notebooks/colab.ipynb` (T4 gratis; vLLM con decodificación restringida al
esquema y Unsloth). Métricas: F1 por campo (similitud de tokens en texto libre), F1 micro/macro,
% de JSON válido, tasa de alucinación (valores que no aparecen en el texto) y latencia.

## Resultados (gold, 262 documentos de ago–sep)

F1 micro sin los campos de texto libre (objeto, beneficiarios), que van en su propia columna.

| Sistema | F1 micro | F1 macro | F1 micro, 30 docs anotados desde cero | Texto libre | JSON válido | Alucinación |
|---|---|---|---|---|---|---|
| Reglas (regex) | 0,840 | 0,709 | 0,852 | 0,650 | 100 % | 1,0 % |
| Qwen2.5-1.5B base + decodificación restringida | 0,697 | 0,631 | 0,659 | 0,717 | 99,2 % | 12,2 % |
| Qwen2.5-1.5B + QLoRA v1 | 0,930 | 0,863 | 0,864 | 0,832 | 100 % | 0,9 % |
| Qwen2.5-1.5B + QLoRA v2 | 0,947 | 0,882 | 0,900 | 0,844 | 100 % | 0,9 % |
| **Qwen2.5-1.5B + QLoRA v3** | **0,957** | **0,919** | **0,932** | 0,842 | **100 %** | **1,0 %** |
| Gemini 3.5 Flash-Lite (API) | 0,963* | 0,960* | 0,910 | 0,964* | 100 % | 0,9 % |

\* El gold se pre-rellenó con Gemini y eso infla su nota; la columna de los 30 documentos anotados
desde cero es la comparación justa (con solo 30 documentos, diferencias de 2 puntos no son
concluyentes). Gemini se evalúa tal cual; el modelo pequeño aprende además las convenciones del
gold, y parte de su ventaja viene de ahí.

- v1: r=16, 1 época sobre ~1.400 etiquetas silver de Gemini (~50 min en una T4 gratis de Colab).
- v2: etiquetas pasadas por las convenciones del gold (`label/convenciones.py`) y 2 épocas.
- v3: más convenciones (plazo «de un mes», plazos con solo fecha final, plazas sumadas por turno,
  puesto vacío si hay varios).

### En CPU (GGUF q4_k_m, llama.cpp), sobre los 30 documentos anotados desde cero

| Sistema | F1 micro | Texto libre | Alucinación | Latencia p50 / p95 por documento |
|---|---|---|---|---|
| QLoRA v3 en GPU (vLLM, fp16) | 0,932 | 0,718 | 1,3 % | — (en lote) |
| **QLoRA v3 en CPU (GGUF q4_k_m, ~1 GB)** | **0,924** | 0,702 | 0,7 % | 32 s / 59 s |
| Gemini 3.5 Flash-Lite (API) | 0,910 | 0,743 | 0,7 % | 1,2 s / 1,5 s |

Cuantizar a 4 bits cuesta menos de un punto de F1. En CPU tarda ~30 s por documento (entradas de
~7.000 tokens), suficiente para procesar el BOE del día (~200 documentos candidatos, ~1,5 h) en un
proceso por lotes sin GPU ni API de pago.
