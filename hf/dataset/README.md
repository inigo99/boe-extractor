---
license: cc-by-4.0
language:
  - es
pretty_name: boe-extractor
size_categories:
  - 1K<n<10K
task_categories:
  - text-generation
tags:
  - information-extraction
  - structured-output
  - json
  - legal
  - boe
  - spanish
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: validation
        path: validation.jsonl
      - split: test
        path: test.jsonl
---

# boe-extractor · extracción estructurada del BOE

Disposiciones del **Boletín Oficial del Estado** (2026) emparejadas con su extracción en **JSON**
según un esquema por tipo de documento. Es el dataset con el que se entrenó y evaluó el modelo
[`inigo99/boe-extractor`](https://huggingface.co/inigo99/boe-extractor).

Código y proceso completo: [github.com/inigo99/boe-extractor](https://github.com/inigo99/boe-extractor).

## Particiones

| Split | Documentos | Periodo | Etiquetas |
|---|---|---|---|
| `train` | 1.409 | abril–julio 2026 | *silver*: Gemini 3.5 Flash-Lite + reglas de alineación |
| `validation` | 83 | abril–julio 2026 | *silver*, igual que train |
| `test` | 262 | agosto–septiembre 2026 | *gold*: revisadas (232) o anotadas desde cero (30) |

La partición es **temporal**: ningún documento de test es anterior al último de entrenamiento.

Por tipo: `convocatoria` 731 + 100 test, `licitacion` 450 + 100, `ayuda` 311 + 62.

## Campos

| Campo | Descripción |
|---|---|
| `id` | Identificador BOE (`BOE-A-2026-…`, `BOE-B-2026-…`) |
| `tipo` | `convocatoria`, `ayuda` o `licitacion` |
| `subtipo` | Solo en convocatorias: `bases` o `anuncio_local`; `null` en el resto |
| `fecha` | Fecha de publicación (AAAA-MM-DD) |
| `titulo`, `departamento` | Del sumario del BOE |
| `url_html` | Enlace al documento en boe.es |
| `texto` | Texto plano del documento, **recortado a 6.144 tokens** del tokenizador de Qwen2.5 (lo que ve el modelo) |
| `json` | La extracción, **como cadena JSON** (su forma depende del `tipo`) |
| `etiqueta` | `silver`, `revisada` o `desde_cero` |

Esquemas (Pydantic, versión `v1`, en `src/boe_extractor/schemas.py`):

- **convocatoria**: organismo, puesto, vínculo, plazas por turno, sistema, plazo de solicitudes,
  titulación, tasa, boletín de las bases.
- **ayuda**: organismo, objeto, beneficiarios, cuantía total, importe máximo por beneficiario,
  plazo, régimen, id BDNS, bases reguladoras.
- **licitacion**: órgano de contratación, objeto, tipo de contrato, procedimiento, valor estimado,
  presupuesto base, fecha límite de ofertas, NUTS, CPV, duración, lotes.

Un campo que el documento no dice es `null`.

```python
import json

from datasets import load_dataset

ds = load_dataset("inigo99/boe-extractor-dataset")
fila = ds["test"][0]
datos = json.loads(fila["json"])
```

## Cómo se etiquetó

- **Silver**: Gemini 3.5 Flash-Lite con salida forzada al esquema. Después, reglas deterministas
  (`label/convenciones.py`) alinean sus hábitos con las convenciones del gold: plazo «de un mes» =
  30 días naturales, régimen `null` si el texto no lo menciona, plazas sumadas por turno, cita corta
  de las bases reguladoras, etc.
- **Gold**: propuesta de Gemini revisada campo a campo contra el texto por un segundo modelo; los
  44 desacuerdos los decidió una persona. 30 documentos (`desde_cero`) se anotaron a mano sin
  propuesta, para medir el sesgo del pre-rellenado: son la comparación justa con Gemini.
- Se descartaron del gold 3 documentos que no eran convocatorias.

## Limitaciones

- **Silver con ruido**: son etiquetas de un modelo; las reglas corrigen errores sistemáticos, no
  todos.
- **Gold pequeño y pre-rellenado**: 232 de 262 documentos parten de la propuesta de Gemini, lo que
  favorece a Gemini al evaluar sobre ellos.
- **Texto recortado**: en bases largas, los datos que caen después de los 6.144 tokens no están en
  `texto`, y la etiqueta tampoco los recoge.
- Solo BOE de 2026, en castellano, y tres tipos de documento.

## Datos personales

Se excluye la sección II.A (nombramientos, ceses y situaciones personales). Las secciones
incluidas (II.B, III y V) son anuncios de organismos públicos y empresas, aunque pueden contener
nombres de cargos públicos firmantes, tal como aparecen en el BOE.

## Licencia y fuente

- Textos: [API de datos abiertos del BOE](https://www.boe.es/datosabiertos/), reutilización sujeta a
  sus [condiciones](https://www.boe.es/informacion/aviso_legal/index.php#reutilizacion). Fuente:
  Agencia Estatal Boletín Oficial del Estado.
- Etiquetas y dataset derivado: CC BY 4.0. Autor: Íñigo Fernández Barrill
  ([inigo99](https://github.com/inigo99)).
