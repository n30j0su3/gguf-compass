<p align="center"><img src="assets/freakingjson-logo.png" width="92" alt="FreakingJSON"></p>

# 🧭 GGUF Compass

> **Tu brújula para LLMs locales.** Hardware → Modelo → Flags en 30 segundos. Hecho por [FreakingJSON](https://freakingjson.com).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Live](https://img.shields.io/badge/demo-live-06b6d4.svg)](https://n30j0su3.github.io/gguf-compass/)
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
| 💾 **KV Cache segura** | q8_0 para upstream; Turbo4/3 solo con runtime TurboQuant compatible; nunca q4_0 |
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

### Opción 3: GitHub Pages (Live Demo)

Visita: **https://n30j0su3.github.io/gguf-compass/**

> La demo live siempre refleja la última versión de `main`.

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

## 📚 Cómo interpretar presets y resultados

GGUF Compass separa tres niveles de evidencia:

| Nivel | Significa |
|---|---|
| ✅ **Validado FJSON** | Ejecutado por FreakingJSON con hardware/modelo/runtime identificados |
| 👥 **Receta comunitaria** | Configuración atribuida a una fuente externa; puede variar en otro equipo |
| 🧮 **Estimación del wizard** | Cálculo para empezar; no es un benchmark ni promete tok/s |

Los comandos se adaptan al runtime seleccionado. En particular:

```bash
# llama.cpp oficial — compatible con upstream
--cache-type-k q8_0 --cache-type-v q8_0

# TurboQuant / ik_llama.cpp — solo en build compatible
--cache-type-k turbo4 --cache-type-v turbo3
```

> **Nunca sugerimos KV q4_0.** Puede degradar coherencia y favorecer loops en respuestas largas. Si un comando antiguo del ecosistema lo incluye, sustitúyelo por q8_0/q8_0 o usa Turbo4/3 únicamente en el fork compatible.

### Lo que más cambia el resultado

1. **Modelo exacto y quant:** no asumas que todos los “27B Q4” pesan o rinden igual.
2. **Contexto real:** 128K reserva mucha más KV cache que 8K; el máximo del modelo no es una meta.
3. **Slots paralelos:** en `llama-server`, `-c` es contexto total compartido entre slots.
4. **CPU/RAM:** `--n-cpu-moe` reduce VRAM, pero convierte RAM/CPU en parte crítica del rendimiento.
5. **Build/backend:** CUDA, Vulkan y Metal tienen capacidades distintas; TurboQuant no es upstream.

---

## 🧩 ¿Qué es MoE y por qué importa?

**MoE (Mixture of Experts)** activa solo una parte de sus expertos por token. Puede ofrecer una buena relación capacidad/coste de cómputo, pero los pesos totales siguen necesitando VRAM o RAM.

`--n-cpu-moe N` mantiene los expertos de las primeras N capas en CPU/RAM. Es útil cuando falta VRAM, con un trade-off claro: la velocidad pasa a depender del CPU y del ancho de banda de memoria. Un MoE no es automáticamente más rápido ni mejor que un dense.

---

## 🚀 Técnicas de optimización seguras

### 1. MTP (Multi-Token Prediction)

```bash
--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-n-min 1
```

Úsalo solo si el modelo incluye heads MTP y tu build los soporta. GGUF Compass no lo activa en modelos genéricos. La ganancia depende de la aceptación del draft y de la tarea.

### 2. KV cache en llama.cpp oficial

```bash
--cache-type-k q8_0 --cache-type-v q8_0
```

Es el fallback conservador para contexto largo. Upstream también soporta otros tipos, pero este proyecto evita q4_0 por su riesgo de degradación y repetición.

### 3. TurboQuant (runtime compatible)

```bash
--cache-type-k turbo4 --cache-type-v turbo3
```

Es la configuración validada por FJSON en MiniV, pero **no existe en llama.cpp upstream**. Debes elegir el runtime TurboQuant/ik_llama.cpp en el wizard. Si la ayuda de tu binario no lista `turbo4` y `turbo3`, usa q8_0/q8_0.

### 4. Flash Attention

```bash
--flash-attn auto
```

`auto` es el default seguro: el runtime usa el kernel cuando el backend/modelo lo permite y evita forzar una ruta incompatible.

### 5. Seguridad de red

`llama-server` escucha en `127.0.0.1` por defecto. Si lo expones con `--host 0.0.0.0`, protege el acceso con `--api-key-file`, firewall y CORS limitado. No habilites herramientas/agente en redes no confiables.

Fuente primaria: [llama.cpp — server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md).

---

## 🔧 Herramientas Soportadas

| Herramienta | Tipo | Ideal para | Comando generado |
|-------------|------|-----------|------------------|
| **llama.cpp oficial** | CLI | Compatibilidad upstream, control total | ✅ Sí; KV q8_0 |
| **LM Studio** | GUI | Principiantes, Windows/Mac | Instrucciones |
| **Ollama** | CLI | Gestión simple de modelos | ✅ Comando básico |
| **TurboQuant / ik_llama.cpp** | CLI | KV Turbo4/3 y optimización avanzada | ✅ Sí; requiere fork compatible |
| **vLLM** | Python | Linux/CUDA, alta concurrencia | ✅ Solo modelos/formats compatibles |

---

## 📖 Documentación

- [Knowledge Base](docs/knowledge-base.md) — Políticas de quantización, KV, MoE y seguridad con compatibilidad por runtime
- [Live demo](https://n30j0su3.github.io/gguf-compass/) — Wizard y panel experto
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

## 🔗 FreakingJSON

- **Web:** https://freakingjson.com
- **GitHub:** https://github.com/n30j0su3
- **YouTube:** https://youtube.com/@freakingjson
- **Instagram:** https://instagram.com/freakingjson
- **TikTok:** https://tiktok.com/@freakingjson
- **Todas las redes:** https://linktr.ee/freakingjson
- **Apoyar:** https://buymeacoffee.com/freakingjson
- **Proyecto / Issues:** https://github.com/n30j0su3/gguf-compass/issues

© 2026 FreakingJSON · MIT · Sin telemetría, cookies, analytics ni backend.

---

<p align="center">
  <strong>🧭 GGUF Compass — Encuentra tu norte en el mundo de los LLMs locales.</strong>
</p>
