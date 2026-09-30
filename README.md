# boe-extractor

Del BOE a JSON validado con un LLM pequeño ajustado con QLoRA.

> Estado: semana 1 (datos). En construcción.

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
