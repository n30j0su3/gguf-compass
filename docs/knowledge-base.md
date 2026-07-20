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
  -ngl 99 \
  --n-cpu-moe 20 \
  --no-mmap \
  --mlock \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  -c 8192 \
  -t 4 \
  -b 512 \
  --flash-attn auto \
  --port 8080
```

**Resultado:** 17 tok/s — production-stable por semanas.

### Cuándo NO usar MoE offloading

- ❌ Si el modelo cabe completo en VRAM (sin offloading es más rápido)
- ❌ Si tienes <16GB RAM (page faults matan performance)
- ❌ Si tu CPU es muy vieja (<4 cores, sin AVX2)
- ❌ Para latency-sensitive apps (offloading añade overhead)

---

## Quantización

### ¿Qué es quantización?

Reducir la precisión de los pesos del modelo para ahorrar memoria.

| Quant | Bits | Calidad | Tamaño (9B) | Uso |
|-------|------|---------|-------------|-----|
| Q2_K | 2.5 | Baja | ~3.2 GB | Testing only |
| Q3_K_S | 3.4 | Aceptable | ~3.8 GB | VRAM muy limitada |
| Q3_K_M | 3.9 | Buena | ~4.2 GB | Balance |
| **Q4_K_M** | **4.8** | **Muy buena** | **~5.2 GB** | **Default recomendado** |
| Q5_K_M | 5.7 | Excelente | ~6.5 GB | Calidad prioritario |
| Q6_K | 6.6 | Casi perfecta | ~7.8 GB | VRAM de sobra |
| Q8_0 | 8.0 | Perfecta | ~9.5 GB | Máxima calidad |
| IQ4_XS | 4.25 | Muy buena+ | ~4.8 GB | MoE (imatrix) |

### Reglas de quantización

1. **Q4_K_M es el sweet spot** para 90% de usuarios
2. **IQ4_XS es mejor para MoE** (importance matrix, mejor calidad por bit)
3. **Nunca uses Q2 en producción** — degradación severa
4. **Si cabe en VRAM, sube de quant** — Q6_K > Q4_K_M si tienes espacio
5. **Si no cabe, baja de quant antes de offloading** — Q3_K_S > offloading parcial

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

| Tipo | Compresión | Calidad | Requiere FA |
|------|-----------|---------|-------------|
| f16 | 1x | Perfecta | No |
| q8_0 | 2x | Muy buena | No |
| **q4_0** | **4x** | **Zero loss** | **No** |
| q4_1 | 4x | Ligeramente mejor | No |
| iq4_nl | 4x | Mejor 4-bit | **Sí** |
| **turbo4** | **~4x** | **Near-lossless** | **No** |
| **turbo2** | **~8x** | **Buena** | **No** |

### Fórmula TurboQuant (validada)

```bash
--cache-type-k turbo4    # Keys: near-lossless
--cache-type-v turbo2    # Values: compressed
```

**Por qué:** K determina qué tokens coinciden (necesita precisión). V puede comprimirse más.

---

## Hardware Recipes

### Tier 1: GTX 1060 6GB (el legendario)

```bash
# Modelo: Qwen3.6-35B-A3B IQ4_XS
# VRAM: 6GB | RAM: 24GB | CPU: i3-8100 (4 cores)

llama-server \
  -m Qwen3.6-35B-A3B-IQ4_XS.gguf \
  -ngl 99 \
  --n-cpu-moe 20 \
  --no-mmap --mlock \
  --cache-type-k q4_0 --cache-type-v q4_0 \
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
  -ngl 99 \
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
  -ngl 99 \
  -c 32768 -t 12 -b 512 \
  --flash-attn auto \
  --cache-type-k q4_0 --cache-type-v q4_0 \
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
  -ngl 99 --n-cpu-moe 10 \
  -c 65536 -t 16 -b 512 \
  --flash-attn auto \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --mlock --port 8080

# Resultado: ~45 tok/s
```

### Tier 5: Apple M2 Max 64GB (unified)

```bash
# Modelo: Qwen3.6-35B-A3B Q4_K_M
# Unified: 64GB | CPU: 12 cores

llama-server \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 \
  -c 131072 -t 12 \
  --no-mmap \
  --cache-type-k q4_0 --cache-type-v q4_0 \
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

*Última actualización: 2026-07-20*
*Este documento se actualiza con cada release de GGUF Compass.*
