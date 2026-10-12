---
license: apache-2.0
language:
  - es
base_model: Qwen/Qwen2.5-1.5B-Instruct
pipeline_tag: text-generation
library_name: gguf
tags:
  - gguf
  - llama.cpp
  - qlora
  - unsloth
  - information-extraction
  - structured-output
  - json
  - spanish
  - legal
  - boe
---

# boe-extractor · Qwen2.5-1.5B (QLoRA, GGUF q4_k_m)

Modelo pequeño que convierte disposiciones del **Boletín Oficial del Estado (BOE)** en **JSON
validado contra un esquema**. Es un [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
ajustado con QLoRA y cuantizado a 4 bits (~1 GB) para ejecutarse en CPU con llama.cpp.

Código, datos de evaluación y proceso completo: [github.com/inigo99/boe-extractor](https://github.com/inigo99/boe-extractor).

## Qué extrae

Tres tipos de documento, cada uno con su esquema Pydantic (`schemas.py`, versión `v1`):

| Tipo | Origen en el BOE | Campos |
|---|---|---|
| `convocatoria` | Sección II.B (oposiciones, concursos, plazas de ayuntamientos y universidades) | organismo, puesto, vínculo, plazas por turno, sistema, plazo, titulación, tasa, boletín de las bases |
| `ayuda` | Sección V.B (extractos BDNS) y premios/becas de la III | organismo, objeto, beneficiarios, cuantía total, importe máximo, plazo, régimen, id BDNS, bases reguladoras |
| `licitacion` | Sección V.A (anuncios de licitación) | órgano de contratación, objeto, tipo de contrato, procedimiento, valor estimado, fecha límite, NUTS, CPV, duración, lotes |

El modelo no clasifica: hay que indicarle el tipo (en el proyecto lo hace un clasificador de reglas
sobre el sumario).

## Uso

La forma más fiable es usar las funciones del proyecto, que construyen el mismo prompt que en el
entrenamiento (instrucciones + esquema JSON del tipo) y restringen la salida al esquema:

```bash
pip install "git+https://github.com/inigo99/boe-extractor" huggingface_hub
pip install llama-cpp-python==0.3.19 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

```python
from huggingface_hub import hf_hub_download
from boe_extractor.docs import recortar
from boe_extractor.infer_cpu import cargar, extraer
from boe_extractor.train.sft import ejemplo

ruta = hf_hub_download("<tu-usuario>/boe-extractor-qwen2.5-1.5b-gguf",
                       "Qwen2.5-1.5B-Instruct.Q4_K_M.gguf")
llm = cargar(ruta)

doc = {"id": "BOE-B-2026-27919", "tipo": "ayuda",
       "titulo": "Extracto de la Resolución de 17 de agosto de 2026, de la Secretaría de Estado de Educación, …",
       "departamento": "MINISTERIO DE EDUCACIÓN, FORMACIÓN PROFESIONAL Y DEPORTES"}
texto = recortar([texto_plano_del_documento])[0]   # máx. 6.144 tokens, como en el entrenamiento
r = extraer(llm, ejemplo(doc, texto)["messages"], doc["tipo"])
print(r["json"])
```

Salida esperada (la del gold para ese documento):

```json
{"organismo": "Secretaría de Estado de Educación", "objeto": "…", "beneficiarios": "…",
 "cuantia_total_eur": 510800.0, "importe_max_beneficiario_eur": null,
 "plazo_solicitudes": {"dias": 30, "tipo": "naturales", "desde": "publicacion_boe", "fecha_fin": null},
 "regimen": "concurrencia_competitiva", "bdns_id": "925842", "bases_reguladoras": null}
```

**Formato del prompt**, si se usa sin el paquete:

- *system*: las instrucciones de `label/silver.py` (`INSTRUCCIONES`) + `Tipo de documento: <tipo>.
  Responde solo con un JSON que cumpla este esquema:` + el JSON Schema del tipo.
- *user*: `TÍTULO: <título>\nDEPARTAMENTO: <departamento>\n\nTEXTO:\n<texto plano recortado>`.
- Plantilla de chat de Qwen2.5 (incluida en el GGUF), temperatura 0 y salida restringida al
  esquema (`response_format={"type": "json_object", "schema": …}` en llama-cpp-python).

## Entrenamiento

- **Datos**: ~1.400 documentos del BOE de abril a julio de 2026, etiquetados por Gemini 3.5
  Flash-Lite con salida forzada al esquema (*silver*). Las etiquetas pasan después por reglas
  deterministas que las alinean con las convenciones del conjunto de evaluación
  (`label/convenciones.py`): plazo «de un mes» = 30 días naturales, régimen vacío si el texto no lo
  menciona, plazas sumadas por turno, etc.
- **Método**: QLoRA con [Unsloth](https://github.com/unslothai/unsloth), r=16, α=16, 7 módulos
  (atención y MLP), lr 2e-4 con coseno, 2 épocas, batch efectivo 8, secuencias de hasta 8.192 tokens.
  Solo se calcula la pérdida sobre la respuesta (el JSON). ~100 min en una T4 gratuita de Colab.
- **Exportación**: adaptador fusionado y cuantizado a GGUF q4_k_m.

## Evaluación

Conjunto *gold* de 262 documentos de agosto y septiembre de 2026 (partición temporal: ningún
documento de evaluación es anterior al corte de entrenamiento), pre-rellenados por Gemini, revisados
contra el texto y con los desacuerdos decididos por una persona. 30 documentos se anotaron desde cero,
sin propuesta, y son la comparación justa con Gemini.

F1 micro por campo (similitud de tokens en texto libre; objeto y beneficiarios se reportan aparte):

| Sistema | F1 micro (262) | F1 micro (30 desde cero) | Alucinación | Latencia p50 |
|---|---|---|---|---|
| Reglas (regex) | 0,840 | 0,852 | 1,0 % | <1 ms |
| Qwen2.5-1.5B base + salida restringida | 0,697 | 0,659 | 12,2 % | — |
| QLoRA (fp16, GPU) | 0,957 | 0,932 | 1,0 % | — |
| **Este modelo (GGUF q4_k_m, CPU)** | — | **0,924** | **0,7 %** | **32 s / documento** |
| Gemini 3.5 Flash-Lite (API) | 0,963\* | 0,910 | 0,9 % | 1,2 s |

\* El gold se pre-rellenó con Gemini, lo que favorece su nota en los 262 documentos.

- JSON válido: 100 % (la salida está restringida al esquema).
- Alucinación: valores de texto, importes, fechas o códigos que no aparecen literalmente en el
  documento.
- Latencia medida en un PC de sobremesa, un documento cada vez, con entradas de ~7.000 tokens.

## Limitaciones

- **Solo tres tipos** de documento del BOE y solo en castellano; no sirve para boletines
  autonómicos ni para otras secciones sin reentrenar.
- **Entrada recortada a 6.144 tokens**: en bases de convocatoria largas, los datos que aparecen
  después (anexos, tablas de plazas) no se ven.
- **Etiquetas de un modelo mayor**: aprende de Gemini; donde Gemini se equivoca de forma
  sistemática y ninguna regla lo corrige, este modelo también.
- **Texto libre** (objeto, beneficiarios): resume con sus palabras en lugar de copiar.
- Puntos débiles conocidos: número de plazas cuando vienen en tablas largas y nivel del organismo
  en ayudas (ministerio frente a órgano convocante).
- **Muestra pequeña**: con 30 documentos, diferencias de 2–3 puntos no son significativas.
- **Deriva**: entrenado y evaluado con BOE de 2026; formatos nuevos pueden degradar el resultado.

## Uso responsable

Extracción automática con errores posibles: **verifica siempre en el BOE original** antes de
presentar una solicitud o tomar una decisión. No es asesoramiento jurídico. Los datos de
entrenamiento excluyen la sección II.A (nombramientos y situaciones personales), así que el modelo no
se ha entrenado con datos personales de particulares.

## Licencia y fuentes

- Modelo: Apache 2.0, como el modelo base Qwen2.5-1.5B-Instruct.
- Datos: [API de datos abiertos del BOE](https://www.boe.es/datosabiertos/), reutilización sujeta a
  sus [condiciones](https://www.boe.es/informacion/aviso_legal/index.php#reutilizacion).
- Autor: Íñigo Fernández Barrill ([inigo99](https://github.com/inigo99)).
