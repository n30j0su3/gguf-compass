# 🧭 GGUF Compass

> **Tu brújula para LLMs locales.** Hardware → Modelo → Flags en 30 segundos.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![GitHub stars](https://img.shields.io/github/stars/n30j0su3/gguf-compass?style=social)](https://github.com/n30j0su3/gguf-compass)

---

## 🎯 ¿Qué es GGUF Compass?

GGUF Compass es una herramienta **100% open source, self-contained y sin dependencias** que te dice exactamente:

1. **¿Qué modelo descargar?** — Basado en tu GPU, RAM y caso de uso
2. **¿Qué quantización usar?** — Q4_K_M, IQ4_XS, Q6_K... explicado sin jargon
3. **¿Qué flags ejecutar?** — Comandos listos para copiar y pegar
4. **¿Qué trucos aplicar?** — MoE offloading, MTP, TurboQuant KV cache, y más

Todo desde un **single-file HTML** que puedes abrir en tu navegador. Sin instalar nada. Sin enviar datos a ningún servidor.

---

## ✨ Features

| Feature | Descripción |
|---------|-------------|
| 🎯 **Wizard Guiado** | Paso a paso: GPU → Caso de uso → Contexto → Herramienta |
| ⚡ **Modo Experto** | Todos los knobs visibles. Sliders, toggles, comando generado en tiempo real |
| 🧩 **MoE Offloading** | Calcula `--n-cpu-moe` óptimo para correr 35B en 6GB VRAM |
| 🚀 **MTP / Spec Dec** | Cuándo usar Multi-Token Prediction y cuándo evitarlo |
| 💾 **KV Cache Quant** | TurboQuant, q4_0, trade-offs de contexto vs velocidad |
| 🔧 **5 Herramientas** | llama.cpp, LM Studio, Ollama, ik_llama.cpp, vLLM |
| 🌙 **Dark/Light Mode** | Tema persistente en localStorage |
| 📱 **Responsive** | Funciona en desktop, tablet y móvil |
| 🔒 **Privacy-first** | Zero telemetry, zero analytics, zero servers |

---

## 🚀 Quick Start

### Opción 1: Abrir directamente

```bash
# Clona el repo
git clone https://github.com/n30j0su3/gguf-compass.git
cd gguf-compass

# Abre en tu navegador
open index.html  # macOS
xdg-open index.html  # Linux
start index.html  # Windows
```

### Opción 2: Servir localmente (recomendado)

```bash
# Python 3
python3 -m http.server 8080

# Node.js
npx serve .

# Luego abre http://localhost:8080
```

### Opción 3: GitHub Pages

Visita: `https://n30j0su3.github.io/gguf-compass/`

---

## 🎮 Cómo usar

### Wizard (para la mayoría)

1. **Selecciona tu GPU** — Desde GTX 1060 hasta RTX 5090, o unified memory Apple/AMD
2. **Elige tu caso de uso** — Coding, chat, razonamiento, creativo, agentes, RAG
3. **Define tu contexto** — 2K para chats rápidos, 128K para documentos masivos
4. **Escoge tu herramienta** — llama.cpp recomendado, pero hay opciones
5. **Copia tu comando** — Listo para pegar en terminal

### Modo Experto (para power users)

Activa el toggle "⚡ Experto" y ajusta:

- VRAM / RAM / CPU cores con sliders
- Arquitectura (Dense vs MoE)
- Parámetros del modelo (1B - 70B)
- Quantización exacta
- Context length
- Flash Attention, KV cache quantization, Speculative Decoding

El comando se genera en tiempo real.

---

## 📚 Casos de Uso Reales

### Caso 1: GTX 1060 6GB + 24GB RAM (el legendario Codacus setup)

```
Modelo: Qwen3.6-35B-A3B (MoE)
Quantización: IQ4_XS
Estrategia: MoE CPU Offloading
Flags clave:
  -ngl 99
  --n-cpu-moe 20
  --no-mmap
  --mlock
  --cache-type-k q4_0 --cache-type-v q4_0
  -c 8192
  -t 4

Resultado: ~17 tok/s — production-stable
```

### Caso 2: RTX 3060 12GB (sweet spot budget)

```
Modelo: Qwen3.5-9B (Dense)
Quantización: Q4_K_M
Estrategia: Full GPU
Flags clave:
  -ngl 99
  -c 16384
  -t 8
  -b 512
  --flash-attn auto
  --spec-type draft-mtp --spec-draft-n-max 2

Resultado: ~70 tok/s — coding assistant perfecto
```

### Caso 3: RTX 4090 24GB (high-end consumer)

```
Modelo: Qwen3.6-27B (Dense)
Quantización: Q4_K_M
Estrategia: Full GPU
Flags clave:
  -ngl 99
  -c 32768
  -t 16
  -b 512
  --flash-attn auto
  --cache-type-k q4_0 --cache-type-v q4_0

Resultado: ~45 tok/s — razonamiento profundo
```

### Caso 4: Apple M2 Max 64GB (unified memory)

```
Modelo: Qwen3.6-35B-A3B (MoE)
Quantización: Q4_K_M
Estrategia: Full GPU (unified memory)
Flags clave:
  -ngl 99
  -c 131072
  -t 12
  --cache-type-k q4_0 --cache-type-v q4_0
  --no-mmap

Resultado: ~25 tok/s + 128K context — best of both worlds
```

---

## 🧩 ¿Qué es MoE y por qué importa?

**MoE (Mixture of Experts)** es una arquitectura donde el modelo tiene muchos "expertos" pero solo activa unos pocos por token.

| Aspecto | Dense | MoE |
|---------|-------|-----|
| Parámetros totales | 9B | 35B |
| Parámetros activos | 9B | ~3B |
| Velocidad | 1x | ~3x más rápido |
| Calidad | Baseline | Similar o mejor |
| VRAM para weights | Todo | Solo backbone + expertos en CPU |

**El truco:** Con `--n-cpu-moe`, llama.cpp pone los expertos en RAM y solo el backbone en VRAM. Así un 35B cabe en 6GB VRAM.

---

## 🚀 Técnicas de Optimización

### 1. MTP (Multi-Token Prediction)

Speculative decoding **nativo del modelo**. El modelo predice múltiples tokens ahead y los verifica en batch.

```bash
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1
```

- ✅ **Coding:** +67-75% velocidad
- ✅ **Short context (<4K):** Ideal
- ❌ **Long context (>8K):** Sin beneficio (prefill domina)
- ❌ **MoE:** No funciona bien

### 2. KV Cache Quantization

Comprime la memoria de contexto 4x:

```bash
--cache-type-k q4_0 --cache-type-v q4_0
```

- ✅ **Long context:** Duplica tokens máximos
- ✅ **Zero quality loss** (validado empíricamente)
- ❌ **iq4_nl:** Requiere Flash Attention

### 3. TurboQuant (fork TheTom/llama-cpp-turboquant)

KV cache compression avanzada:

```bash
--cache-type-k turbo4 --cache-type-v turbo2
```

- ✅ K=turbo4 (near-lossless keys)
- ✅ V=turbo2 (compressed values)
- ✅ 2-4x más contexto en mismo VRAM

### 4. Flash Attention

```bash
--flash-attn auto
```

- ✅ Menos VRAM para attention
- ✅ Más rápido en GPUs modernas
- ✅ Auto-detecta si tu GPU lo soporta

---

## 🔧 Herramientas Soportadas

| Herramienta | Tipo | Ideal para | Comando generado |
|-------------|------|-----------|------------------|
| **llama.cpp** | CLI | Control total, benchmarking | ✅ Sí |
| **LM Studio** | GUI | Principiantes, Windows/Mac | Instrucciones |
| **Ollama** | CLI | Simplicidad, Docker-like | ✅ Sí |
| **ik_llama.cpp** | CLI | MoE optimizations extras | ✅ Sí |
| **vLLM** | Python | Datacenter, alta concurrencia | ✅ Sí |

---

## 📖 Documentación

- [Knowledge Base](docs/knowledge-base.md) — Todo el conocimiento extraído de Codacus, llama.cpp docs, y nuestra investigación
- [Presets](docs/presets.md) — Configuraciones validadas por hardware tier
- [Contributing](CONTRIBUTING.md) — Cómo contribuir al proyecto

---

## 🤝 Contributing

¡Las contribuciones son bienvenidas! Areas donde puedes ayudar:

1. **Nuevos modelos** — Añadir familias de modelos a la base de datos
2. **Hardware profiles** — Validar configuraciones en GPUs que no tenemos
3. **Traducciones** — i18n para más idiomas
4. **UI/UX** — Mejoras de diseño y accesibilidad
5. **Documentación** — Tutoriales, videos, ejemplos

Ver [CONTRIBUTING.md](CONTRIBUTING.md) para guidelines.

---

## 🙏 Agradecimientos

- **Codacus** — Por la investigación pionera en MoE offloading y hardware limitado
- **ggml-org/llama.cpp** — El runtime que hace todo esto posible
- **TheTom/llama-cpp-turboquant** — Por TurboQuant KV cache
- **Unsloth** — Por los GGUFs optimizados y MTP conversions
- **La comunidad local AI** — Por compartir conocimiento libremente

---

## 📄 License

MIT License — ver [LICENSE](LICENSE) para detalles.

---

## 🔗 Links

- **GitHub:** https://github.com/n30j0su3/gguf-compass
- **Issues:** https://github.com/n30j0su3/gguf-compass/issues
- **Discussions:** https://github.com/n30j0su3/gguf-compass/discussions

---

<p align="center">
  <strong>🧭 GGUF Compass — Encuentra tu norte en el mundo de los LLMs locales.</strong>
</p>
