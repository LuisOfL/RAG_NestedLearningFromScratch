# RAG, math implementation (In Procces)



# Hybrid RAG Backend (C++ / CUDA / Python)

Un backend híbrido de alto rendimiento diseñado para la ingesta de documentos, procesamiento de lenguaje natural y aceleración de cómputo en GPU. Este sistema combina la flexibilidad de **Python/FastAPI** junto con servicios en la nube de **Azure AI** para la extracción e inferencia de texto, delegando las tareas pesadas de procesamiento de texto, llamadas de red de baja latencia y clustering masivo a un backend nativo en **C++ y kernels de CUDA**.

## 🚀 Características Principales

* **Extracción en la Nube & Local:** Integración con **Azure Document Intelligence** para procesar PDFs de forma remota y soporte nativo con **Poppler C++** para parsing de archivos locales.
* **Procesamiento de Texto en C++:** Implementación nativa de un *Recursive Character Text Splitter* optimizado para la segmentación eficiente de cadenas de texto de gran volumen.
* **Consumo de API Vectorial Ultra Rápido:** Módulo en C++ utilizando `libcurl` y `nlohmann/json` para interactuar de forma directa y asíncrona con las APIs de Embeddings de OpenAI, reduciendo el overhead de runtime en Python.
* **Clustering Acelerado por GPU (CUDA):** Kernel de CUDA personalizado (`assign_clusters_kernel`) para la ejecución paralela del algoritmo de clustering **K-Means**, permitiendo agrupar y filtrar miles de vectores de embeddings en milisegundos directamente en la VRAM de la tarjeta gráfica NVIDIA.
* **Interoperabilidad Eficiente:** Enlaces nativos de C++/CUDA expuestos limpiamente a Python usando **Pybind11** bajo el nombre de módulo `cpp_backend`.
* **Capa RAG y Humanización:** Pipeline que orquesta la búsqueda semántica, recupera el contexto filtrado por los clusters más cercanos, e inyecta la información en LLMs locales/remotos (como **Phi-3/Phi-4**) para devolver respuestas coherentes y estructuradas.

---

## 🛠️ Arquitectura del Pipeline Principal

El núcleo de la aplicación ejecuta el siguiente flujo integrado:
1. **Ingesta:** Se descarga un archivo PDF desde **Azure Blob Storage** o mediante un formulario HTTP.
2. **Extracción:** Se extrae el contenido textual estructurado en Markdown usando la API de Azure o la librería local Poppler.
3. **Chunking:** El texto plano se divide en fragmentos (*chunks*) usando la función nativa en C++.
4. **Embeddings:** Cada fragmento se convierte en un vector denso llamando al endpoint de embeddings mediante C++/Curl.
5. **Clustering CUDA:** Los embeddings se cargan en la GPU, donde el kernel de CUDA ejecuta K-Means para clasificar las temáticas más relevantes del documento.
6. **Búsqueda & Síntesis:** El mensaje del usuario es vectorizado, se buscan los clusters con la información más cercana y el contexto recuperado es enviado a un modelo **Phi-3/Phi-4** de Azure OpenAI para generar la respuesta humanizada.

---

## 💻 Requisitos del Sistema

* **Sistema Operativo:** Linux (Ubuntu 20.04+ recomendado) o Windows con soporte WSL2 y CUDA configurado.
* **NVIDIA CUDA Toolkit:** Versión 11.0 o superior instalada junto con controladores de GPU NVIDIA compatibles.
* **Compilador C++:** Soporte para C++17 (`g++` u `os-clang`).
* **Librerías del Sistema:** `libpoppler-cpp-dev`, `libcurl4-openssl-dev`.
* **Python:** Versión 3.9 o superior.

---

## 🔧 Configuración del Entorno y Compilación

### 1. Variables de Entorno (.env)
Crea un archivo `.env` en el directorio raíz de Python con las siguientes credenciales:

```env
AZURE_STORAGE_CONNECTION_STRING="tu_connection_string_de_storage"
DOCUMENT_INTELLIGENCE_ENDPOINT="[https://tu-recurso.cognitiveservices.azure.com/](https://tu-recurso.cognitiveservices.azure.com/)"
DOCUMENT_INTELLIGENCE_KEY="tu_llave_de_document_intelligence"
AZURE_OPENAI_API_KEY="tu_llave_de_azure_openai"
AZURE_OPENAI_ENDPOINT="[https://tu-recurso-openai.openai.azure.com/](https://tu-recurso-openai.openai.azure.com/)"
