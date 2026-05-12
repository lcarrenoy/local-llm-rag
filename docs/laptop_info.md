# Mi Laptop HP Pavilion x360 — Documentación

## Hardware
- Modelo: HP Pavilion x360 Convertible 14-cd0xxx
- Procesador: Intel Core i5-8250U @ 1.60GHz (Turbo 3.4GHz)
- Núcleos: 4 cores · 8 threads
- RAM: 12 GB DDR4
- GPU: Intel UHD Graphics 620 (sin CUDA)
- SO: Windows 11

## Discos
- C: 464 GB (Sistema + Apps) — ~37 GB libres
- D: 465 GB (Datos + Modelos + AI) — ~409 GB libres

## Modelos LLM instalados
- Ollama: gemma3:1b (815 MB) en localhost:11434
- LM Studio: google/gemma-3-1b + nomic-embed-text-v1.5 en localhost:1234

## Servicios locales
- Ollama: puerto 11434 — siempre corriendo
- OpenClaw Gateway: puerto 18789 — Scheduled Task
- AI Core System (FastAPI): puerto 8001 — manual
- n8n: puerto 5678 — manual
- MLflow: puerto 5000 — manual

## Stack de desarrollo
- Python 3.13 + uv
- Node.js + npm
- Docker Desktop
- Git + GitHub CLI (lcarrenoy)
- PowerShell 7.6.1
- VS Code

## API Keys configuradas
- Anthropic (Claude): D:\Claves\Claude.txt
- Google (Gemini): D:\Claves\Gemini API.md
- LangChain/LangSmith: D:\Claves\langchain.txt
- LangFuse: D:\Claves\langfuse.txt
