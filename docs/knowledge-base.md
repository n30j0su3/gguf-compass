# GGUF Compass — Knowledge Base

> Todo el conocimiento práctico para correr LLMs locales, curado y validado.

---

## Tabla de Contenidos

1. [Fundamentos](#fundamentos)
2. [MoE Offloading](#moe-offloading)
3. [Quantización](#quantización)
4. [MTP / Speculative Decoding](#mtp--speculative-decoding)
5. [KV Cache Optimization](#kv-cache-optimization)
6. [Hardware Recipes](#hardware-recipes)
7. [Troubleshooting](#troubleshooting)
8. [Fuentes](#fuentes)

---

## Fundamentos

### ¿Qué es un GGUF?

GGUF (GPT-Generated Unified Format) es el formato estándar para modelos LLM cuantizados. Reemplazó a GGML en 2023.

**Características:**
- Single-file (todo en un .gguf)
- Metadata embebida (tokenizer, config, etc.)
- Soporte para múltiples arquitecturas (Llama, Mistral, Qwen, Gemma, etc.)
- Extensible para futuras features

### ¿Qué es VRAM y por qué importa?

VRAM = memoria de la GPU. Es el recurso más escaso para LLMs locales.

| VRAM | Qué cabe |
|------|----------|
| 4 GB | Modelos ≤3B Q4 |
| 6 GB | Modelos ≤7B Q4, MoE 35B con offloading |
| 8 GB | Modelos ≤9B Q4, MoE 35B con offloading |
| 12 GB | Modelos ≤13B Q4, 27B Q3, MoE 35B |
| 16 GB | Modelos ≤27B Q3/Q4, MoE 35B cómodo |
| 24 GB | Modelos ≤34B Q4, 70B Q3 con offloading |
| 48 GB+ | Casi cualquier modelo consumer |

### ¿Qué es RAM y por qué importa?

RAM = memoria del sistema. Importa cuando:
- El modelo no cabe en VRAM (offloading)
- Usas `--no-mmap` (carga completa en RAM)
- Usas MoE con `--n-cpu-moe` (expertos en RAM)

**Regla:** Para MoE offloading, necesitas al menos `model_size * 1.2` en RAM.

---

## MoE Offloading

### ¿Qué es MoE?

Mixture of Experts. El modelo tiene muchos "expertos" (redes neuronales especializadas) pero solo activa unos pocos por token.

**Ejemplo: Qwen3.6-35B-A3B**
- 35B parámetros totales
- ~3B parámetros activos por token
- 256 expertos, 8 activos por token

### ¿Cómo funciona el offloading?

```
┌─────────────────────────────────────┐
│  GPU (VRAM)                         │
│  ├── Attention layers (backbone)    │
│  ├── Embedding                      │
│  └── Output head                    │
│  Total: ~30% del modelo             │
└─────────────────────────────────────┘
           ↕ PCIe/NVLink
┌─────────────────────────────────────┐
│  CPU (RAM)                          │
│  └── Expert layers (MoE feed-forward)│
│  Total: ~70% del modelo             │
└─────────────────────────────────────┘
```

### El flag mágico: `--n-cpu-moe`

```bash
--n-cpu-moe N
```

Donde N = número de capas MoE a poner en CPU.

**Cálculo rápido:**
- Si VRAM libre = 6GB y modelo = 17.6GB
- Necesitas offload ~70% de capas
- `--n-cpu-moe 20` (de 40 capas totales) es un buen punto de partida

**Receta Codacus (GTX 1060 6GB + 24GB RAM):**
```bash
llama-server \
  -m Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -ngl all \
  --n-cpu-moe 20 \
  --mlock \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  -c 8192 \
  -t 4 \
  -b 512 \
  --flash-attn auto \
  --port 8080
```

**Resultado reportado por la fuente:** ~17 tok/s en ese hardware/build. Trátalo como receta comunitaria, no como garantía portable.

### Valores de n-cpu-moe por VRAM tier (Codacus + comunidad)

| VRAM | GPU Ejemplo | n-cpu-moe inicio | Rango | Batch |
|------|-------------|------------------|-------|-------|
| **6 GB** | GTX 1060, RTX 3050 | 35-41 | 27-45 | 512-2048 |
| **8 GB** | RTX 3070, RTX 4060 | 999 (máximo) | 32-999 | 512-1024 |
| **12 GB** | RTX 3060, RTX 4070 | 32 | 24-48 | 512-4096 |
| **16 GB** | RTX 4060 Ti, RTX 5070 Ti | 16-24 | 8-32 | 1024-4096 |
| **24 GB** | RTX 3090, RTX 4090 | 0 (no usar) | 0 | 2048-4096 |

> **Regla de oro (Codacus):** Si VRAM ≤ 8GB, `--n-cpu-moe 999` (todos los expertos a CPU) es la estrategia correcta. Si VRAM ≥ 12GB, ajusta fino entre 16-48.

### Progresión de velocidad documentada (Codacus GTX 1060)

| Iteración | Cambio | tok/s | Delta |
|-----------|--------|-------|-------|
| 1 | Baseline (solo -ngl all) | 3 | — |
| 2 | + `--n-cpu-moe 41` | 10 | +233% |
| 3 | + `--no-mmap` | 13.5 | +35% |
| 4 | n-cpu-moe 41→35 | 17 | +26% |
| 5 | + `--mlock` + TurboQuant | 17 sostenido | 0% pero desbloquea 256K ctx |

### Cuándo NO usar MoE offloading

- ❌ Si el modelo cabe completo en VRAM (sin offloading es más rápido)
- ❌ Si tienes <16GB RAM (page faults matan performance)
- ❌ Si tu CPU es muy vieja (<4 cores, sin AVX2)
- ❌ Para latency-sensitive apps (offloading añade overhead)
- ❌ **Speculative decoding NO funciona** — draft model da 11 tok/s vs 17 sin él (Codacus validado)

---

## Quantización

### ¿Qué es quantización?

Reducir la precisión de los pesos del modelo para ahorrar memoria.

| Quant | Bits | Calidad | Tamaño (9B) | Uso |
|-------|------|---------|-------------|-----|
| Q2_K | 2.5 | Baja | ~3.2 GB | Testing only |
| Q3_K_S | 3.4 | Aceptable | ~3.8 GB | VRAM muy limitada |
| Q3_K_M | 3.9 | Buena | ~4.2 GB | Balance |
| Q4_K_M | 4.8 | Muy buena | ~5.2 GB | Default recomendado |
| Q5_K_M | 5.7 | Excelente | ~6.5 GB | Calidad prioritario |
| Q6_K | 6.6 | Casi perfecta | ~7.8 GB | VRAM de sobra |
| Q8_0 | 8.0 | Perfecta | ~9.5 GB | Máxima calidad |
| IQ4_XS | 4.25 | Muy buena+ | ~4.8 GB | MoE (imatrix) |
| **UD-Q4_K_XL** | **~4.5** | **Excelente** | **~4.8 GB** | **Unsloth Dynamic — mejor calidad/tamaño que Q4_K_M** |

### Reglas de quantización

1. **Q4_K_M es nuestro piso práctico para trabajo real** — no es una garantía universal; valida siempre el modelo/tarea
2. **Q5_K_M / Q6_K** — Úsalos si el modelo sigue cabiendo con margen
3. **IQ4_XS puede ser útil para MoE** cuando la quant específica fue creada con una importance matrix confiable
4. **Unsloth Dynamic (UD)** cambia la precisión por tensor; evalúa el archivo exacto, no asumas superioridad universal
5. **Q3_K_S / Q3_K_M** — Solo emergencia de memoria o pruebas; normalmente conviene un modelo menor en Q4/Q5
6. **Q2_K** — Experimentación; no recomendado para trabajo crítico
7. **No confíes en el quant por defecto de una GUI** — verifica nombre, tamaño, uploader y model card

### ¿Por qué Q4 es el piso práctico?

La pérdida por quantización depende de arquitectura, capa, tarea y método de calibración; porcentajes universales como “95/80/60%” no son evidencia. En general, Q4_K_M conserva mejor el comportamiento que Q3/Q2 con un coste de memoria todavía razonable. Para código, razonamiento o agentes, valida el archivo exacto con tu tarea.

**Regla de oro:** Si tu VRAM no permite Q4_K_M, usa un modelo más pequeño en Q4/Q5 antes que uno grande en Q3/Q2.

---

## MTP / Speculative Decoding

### ¿Qué es MTP?

Multi-Token Prediction. El modelo tiene "cabezas MTP" que predicen múltiples tokens ahead. Luego verifica en batch.

### Flags

```bash
--spec-type draft-mtp
--spec-draft-n-max 2    # Max tokens a predecir
--spec-draft-n-min 1    # Min tokens a predecir
```

### Cuándo usar MTP

| Escenario | Beneficio | Notas |
|-----------|-----------|-------|
| Coding | +67-75% | Sintaxis repetitiva = alta acceptance |
| Short context (<4K) | +60-70% | Generation-bound |
| Structured output (JSON) | +50-60% | Patrones predecibles |
| Creative writing | ~0% | No repetitivo, overhead |
| Long context (>8K) | ~0% | Prefill domina |
| MoE models | ❌ No funciona | Sparsity rompe prediction |

### Modelos con MTP

- Qwen3.5-9B-MTP
- Qwen3.6-27B-MTP
- Qwopus3.6-27B-v2-MTP
- Gemma-4-12B-MTP
- Qwen3.6-35B-A3B-MTP (limitado en MoE)

---

## KV Cache Optimization

### ¿Qué es KV cache?

Memoria que guarda los tokens ya procesados para no recalcular attention. Crece linealmente con context length.

### Tipos de KV cache quant

| Tipo | Disponibilidad | Política GGUF Compass |
|------|----------------|------------------------|
| f16 | llama.cpp upstream | Máxima precisión, mayor memoria |
| q8_0 | llama.cpp upstream | **Fallback conservador recomendado** |
| q5_0/q5_1 | llama.cpp upstream | Alternativa avanzada; validar por modelo |
| q4_0/q4_1/iq4_nl | llama.cpp upstream | Soportados, pero **q4_0 no se recomienda** por riesgo de coherencia/loops |
| turbo4/turbo3 | Solo fork TurboQuant compatible | **Default FJSON en runtime compatible**; no funciona en upstream |

### Política TurboQuant validada por FJSON

```bash
--cache-type-k turbo4
--cache-type-v turbo3
```

K conserva más precisión; V usa Turbo3. Si el binario no lista ambos tipos en `llama-server --help`, usa `q8_0/q8_0`. Nunca sustituyas silenciosamente por `q4_0/q4_0`.

### TurboQuant — estado actual (2026-07)

| Implementación | Estado | Flags | Compresión |
|----------------|--------|-------|-----------|
| **TheTom/llama-cpp-turboquant** | ✅ Funcional | `-ctk turbo4 -ctv turbo3` | ~4x |
| **turboquant_plus (AmesianX)** | ✅ Fork mejorado | `-ctk tbq3 -ctv tbq3` | ~4.7x |
| **PR #21089 (mainline)** | 🔄 Abierto | `-ctk tbq3_0 -ctv tbq3_0` | ~3-4 bits |
| **KV q8_0 (mainline)** | ✅ Upstream | `-ctk q8_0 -ctv q8_0` | Conservador |
| **KV q4_0 (mainline)** | ⚠️ Soportado, no recomendado aquí | `-ctk q4_0 -ctv q4_0` | Riesgo de degradación/loops |

> **Nota Codacus:** TurboQuant no mejora tok/s directamente, pero **desbloquea contextos largos** (256K vs 64K) que de otra forma serían imposibles.

---

## Herramientas Comparativas

### Tabla de decisión rápida

| Herramienta | Tipo | Overhead | Control | Ideal para | No usar si... |
|-------------|------|----------|---------|-----------|---------------|
| **llama.cpp** | CLI | 0% | Total | Power users, benchmarking | Quieres GUI |
| **Ollama** | CLI | +5-15% | Básico | Principiantes, Docker fans | Necesitas flags avanzados |
| **LM Studio** | GUI | ~20% | Medio | Windows/Mac, no-code | Linux, headless servers |
| **ik_llama.cpp** | CLI | -8-15%* | Alto | Q5+ quants, MoE max | Quieres estabilidad mainline |
| **vLLM** | Python | Variable | Medio | Datacenter, concurrencia | Single-user, poca VRAM |
| **BeeLlama.cpp** | CLI | Experimental | Alto | DFlash, máxima velocidad | Producción estable |

*ik_llama.cpp es *más rápido* que mainline en Q5+ quants.

### vLLM — gotcha crítico

vLLM **pre-asigna 90% de VRAM** al iniciar. Esto es normal (PagedAttention), pero significa:
- No puedes correr otros procesos GPU simultáneamente
- El "uso real" es menor, pero la reserva es total
- Para single-user, llama.cpp es más eficiente

### ik_llama.cpp — cuándo usarlo

- ✅ Quants Q5_K_M o superiores
- ✅ MoE con muchos expertos
- ✅ Necesitas `--override-tensor` avanzado
- ❌ Quants bajos (Q2, Q3) — sin beneficio
- ❌ Quieres estabilidad garantizada

### BeeLlama.cpp — DFlash

Block Diffusion para speculative decoding:

| Modelo | Vanilla | DFlash | Contexto |
|--------|---------|--------|----------|
| Qwen 27B | 48 t/s | **135 t/s** | Pico |
| Qwen 27B | 18 t/s | **72 t/s** | 200K ctx |

**No usar para:** JSON, matemáticas, cálculos precisos (difusión puede alterar números).

---

## Árbol de Decisión Rápido

```
¿Cuánta VRAM tienes?
│
├── ≤ 4GB → Modelos ≤3B Q4_K_M (mínimo aceptable)
│           ⚠️ Q3 solo si es ≤3B y no hay alternativa
│           ❌ NUNCA Q2 para trabajo real
│
├── 6-8GB → MoE 35B con --n-cpu-moe 999 (Q4_K_M o IQ4_XS)
│           O Dense 7-9B Q4_K_M full GPU
│           ❌ NO uses Q3/Q2 en modelos >7B para trabajo
│
├── 12GB → MoE 35B con --n-cpu-moe 32 (Q4_K_M)
│          O Dense 13B Q4_K_M
│
├── 16GB → MoE 35B con --n-cpu-moe 16-24 (Q4_K_M/IQ4_XS)
│          O Dense 27B Q4_K_M
│
└── 24GB+ → Dense 27B Q4_K_M full GPU
            O MoE 35B Q4_K_M sin offload
            O sube a Q5_K_M / Q6_K si sobra VRAM

¿Qué caso de uso?
│
├── Coding → Q4_K_M mínimo. MTP si dense y ctx <4K
│            DFlash si quieres máxima velocidad
│            ngram-mod si código muy repetitivo
│
├── Chat general → Q4_K_M es suficiente. Sin spec dec.
│
├── Razonamiento → Q4_K_M o superior. Dense grande. Sin spec dec. ctx 16K+
│
├── Creativo → Q4_K_M mínimo. Sin spec dec (9% acceptance)
│
└── RAG/documentos → Q4_K_M + KV q8_0/q8_0 (upstream) o Turbo4/3 (fork compatible). ctx 32K+
```

---

## Hardware Recipes

### Tier 1: GTX 1060 6GB (el legendario)

```bash
# Modelo: Qwen3.6-35B-A3B IQ4_XS
# VRAM: 6GB | RAM: 24GB | CPU: i3-8100 (4 cores)

llama-server \
  -m Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -ngl all \
  --n-cpu-moe 20 \
  --mlock \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -c 8192 -t 4 -b 512 \
  --flash-attn auto \
  --port 8080

# Resultado: ~17 tok/s
```

### Tier 2: RTX 3060 12GB (sweet spot)

```bash
# Modelo: Qwen3.5-9B Q4_K_M
# VRAM: 12GB | RAM: 16GB | CPU: 8 cores

llama-server \
  -m Qwen3.5-9B-Q4_K_M.gguf \
  -ngl all \
  -c 16384 -t 8 -b 512 \
  --flash-attn auto \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --cache-reuse 256 \
  --port 8080

# Resultado: ~70 tok/s (con MTP)
```

### Tier 3: RTX 4060 Ti 16GB (balanced)

```bash
# Modelo: Qwen3.6-27B Q3_K_S
# VRAM: 16GB | RAM: 32GB | CPU: 12 cores

llama-server \
  -m Qwen3.6-27B-Q3_K_S.gguf \
  -ngl all \
  -c 32768 -t 12 -b 512 \
  --flash-attn auto \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-reuse 256 \
  --port 8080

# Resultado: ~40 tok/s
```

### Tier 4: RTX 4090 24GB (high-end)

```bash
# Modelo: Qwen3.6-35B-A3B Q4_K_M
# VRAM: 24GB | RAM: 64GB | CPU: 16 cores

llama-server \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl all --n-cpu-moe 10 \
  -c 65536 -t 16 -b 512 \
  --flash-attn auto \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --mlock --port 8080

# Resultado: ~45 tok/s
```

### Tier 5: Apple M2 Max 64GB (unified)

```bash
# Modelo: Qwen3.6-35B-A3B Q4_K_M
# Unified: 64GB | CPU: 12 cores

llama-server \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl all \
  -c 131072 -t 12 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --cache-reuse 256 \
  --port 8080

# Resultado: ~25 tok/s + 128K context
```

---

## Troubleshooting

### Problema: "CUDA out of memory"

**Solución:**
1. Baja `-ngl` (menos capas en GPU)
2. Sube `--n-cpu-moe` (más expertos en CPU)
3. Usa quantización más agresiva
4. Cierra apps que usan GPU (browsers, OBS, etc.)

### Problema: "Modelo carga pero va muy lento"

**Checklist:**
- [ ] ¿Estás usando `--no-mmap`? (page faults matan performance)
- [ ] ¿Tienes `--mlock`? (swap mata performance)
- [ ] ¿Mataste procesos VRAM parasites? (Ollama, ComfyUI, browsers)
- [ ] ¿Estás usando threads correctos? (`-t $(nproc)`)
- [ ] ¿Flash Attention está activado? (`--flash-attn auto`)

### Problema: "MTP no hace nada"

**Diagnóstico:**
1. ¿El modelo tiene MTP heads? (busca "MTP" en el nombre del GGUF)
2. ¿Estás usando `--spec-type draft-mtp`?
3. ¿El contexto es >4K? (MTP no ayuda en long context)
4. ¿Es un modelo MoE? (MTP no funciona bien en MoE)

### Problema: "MoE offloading es más lento que CPU-only"

**Causa:** Demasiadas capas en CPU. `--n-cpu-moe` muy alto.

**Fix:** Reduce `--n-cpu-moe` gradualmente hasta encontrar el sweet spot.

---

## Fuentes

| Fuente | Tipo | Contribución |
|--------|------|-------------|
| [Codacus YouTube](https://www.youtube.com/@Codacus) | Video | MoE offloading, hardware limitado, DFlash, TurboQuant |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | Código | Runtime, flags, arquitectura |
| [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | Fork | TurboQuant KV cache |
| [Unsloth](https://github.com/unslothai/unsloth) | GGUFs | Quantizaciones optimizadas, MTP conversions |
| Internal benchmarks | Benchmarks | Validación empírica en RTX 5060 Ti, MiniV |

---

## Catálogo Codacus — Videos Relevantes (2026)

**Canal:** 28.2K subs, 50 videos, creado 2016. Todos los videos visibles son de 2026.

### Top 10 por Views (LLM local focus)

| # | Video | Views | Duración | Key Topics | Herramientas | Modelos |
|---|-------|-------|----------|------------|--------------|---------|
| 1 | Running a 35B AI Model on 6GB VRAM, FAST | 304K | 15:06 | MoE offloading, VRAM tuning | llama.cpp | Qwen 3.6 35B-A3B |
| 2 | Build Powerful Local Coding Agent | 64K | 16:56 | Coding agent, prompt processing | llama.cpp, Pi, Tailscale | REAP MoE |
| 3 | Everything That Actually Matters for Local AI | 42K | 20:57 | Model selection, quantization | llama.cpp | MoE models |
| 4 | I Asked Claude Fable 5 to Improve llama.cpp | 39K | 16:42 | AI-assisted coding | llama.cpp, Claude | — |
| 5 | DFlash on GTX 1060: Dense Models Cheat VRAM | 36K | 11:30 | DFlash, dense vs MoE | llama.cpp | Dense models |
| 6 | 1M Context in 500MB?! DeepSeek V4 + TurboQuant | 32K | 9:34 | Long context, quantization | TurboQuant | DeepSeek V4 |
| 7 | One llama.cpp Update Made Local AI 65% Faster | 25K | 8:24 | MTP, inference speed | llama.cpp (MTP) | — |
| 8 | 3.5GB model replace 35B daily driver? (Bonsai 27B) | 22K | 21:44 | Small models, comparison | — | Bonsai 27B |
| 9 | Every Local AI Shares ONE Memory (LLM Wiki + OKF) | 19K | 15:19 | Shared memory, agents | LLM Wiki, OKF | — |
| 10 | Build Your Own Private Local AI Stack | 17K | 14:41 | Full stack, automation | n8n | — |

### Shorts Destacados

| Short | Views | Insight |
|-------|-------|---------|
| An 8B model just beat Claude on a laptop | 15K | Local beats cloud |
| 3x Faster LLM, by Google | 14K | Speed optimization |
| Nvidia RTX Spark killed local AI bottleneck | 8.6K | Hardware, VRAM |
| 67% faster than llama.cpp, same model | 7.9K | Alternative runtime |
| It's Not the GPUs. It's the KV Cache. | 485 | KV cache bottleneck |

### Hallazgos del Catálogo

1. **No hay videos de 2025** — Todos son de 2026 (febrero-julio). Canal pivotó a local AI en 2026.
2. **llama.cpp es ubicuo** — ~70% de videos lo mencionan.
3. **Qwen 3.6 35B-A3B es el modelo estrella** — Aparece en los 2 videos más vistos.
4. **Hardware de consumo es el foco** — GTX 1060 6GB, RTX 3060 12GB, M1 MacBook.
5. **El video #1 (304K views)** es el tutorial de MoE offloading que validamos.

---

*Última actualización: 2026-07-20*
*Este documento se actualiza con cada release de GGUF Compass.*
