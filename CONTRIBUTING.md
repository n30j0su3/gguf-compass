# Contributing to GGUF Compass

¡Gracias por tu interés en contribuir! GGUF Compass es un proyecto open source mantenido por la comunidad.

## Cómo contribuir

### 1. Reportar bugs

Abre un issue con:
- Descripción del problema
- Pasos para reproducir
- Hardware (GPU, RAM, OS)
- Browser y versión
- Screenshots si aplica

### 2. Sugerir features

Abre un issue con tag `enhancement`:
- Descripción clara del feature
- Caso de uso (¿quién se beneficia?)
- Mockups o ejemplos si aplica

### 3. Pull Requests

1. Fork el repo
2. Crea una branch: `git checkout -b feature/amazing-feature`
3. Commit tus cambios: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Abre un Pull Request

### Guidelines de código

- **Single-file SPA:** Todo en `index.html`. No añadir dependencias externas.
- **CSS:** Usar variables CSS definidas en `:root`. No hardcodear colores.
- **JS:** Vanilla JS. No frameworks. Funciones declaradas con `function` (hoisted).
- **Data:** Los datos de modelos/hardware van en objetos JS al inicio del script, no en JSON externo (CORS en file://).
- **Responsive:** Mobile-first. Probar en 320px width.

### Añadir nuevos modelos

Edita el objeto `MODEL_RECOMMENDATIONS` en `index.html`:

```javascript
const MODEL_RECOMMENDATIONS = {
    dense: {
        // ... existing
        newSize: {
            params: 13,
            quants: { 'Q4_K_M': 7.5, 'Q6_K': 10.0 },
            examples: ['NewModel-13B']
        }
    }
};
```

### Añadir nuevo hardware

Edita el array `GPU_OPTIONS`:

```javascript
{ id: 'gpu-20gb', label: '20 GB VRAM', vram: 20, examples: 'RTX 4080', notes: '...', icon: '🎮' }
```

### Validación

Antes de enviar PR, verifica:

```bash
# 1. No hay errores JS
# Abre index.html en browser, abre DevTools → Console, debe estar vacío

# 2. Todos los flujos funcionan
# Wizard completo: GPU → Use case → Context → Tool → Results
# Expert mode: sliders actualizan comando en tiempo real

# 3. Responsive
# DevTools → Device toolbar → iPhone SE / iPad / Desktop

# 4. Sin dependencias externas
# grep -E 'http://|https://' index.html
# Solo debe haber fonts (opcional) y GitHub links en footer
```

## Código de conducta

- Sé respetuoso
- Sé constructivo
- Atribuye fuentes
- Comparte conocimiento

## ¿Dudas?

Abre un issue con tag `question` o inicia una discussion.

¡Gracias por hacer GGUF Compass mejor! 🧭
