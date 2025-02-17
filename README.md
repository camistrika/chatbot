# Chatbot RAG (Retrieve-Augment-Generate)

## Descripción

Este proyecto implementa un chatbot basado en un sistema RAG (Retrieve-Augment-Generate). El chatbot está diseñado para responder preguntas específicas utilizando un conjunto de documentos como fuente de conocimiento. Utiliza herramientas avanzadas de procesamiento de lenguaje natural, bases de datos vectoriales y generación de respuestas contextualizadas.

## Requisitos

- Python 3.8 o superior
- pip (gestor de paquetes)
- Ollama (para ejecutar el modelo de lenguaje)
- PostgreSQL (si se desea almacenar datos en una base de datos)
- Flask (para el servidor web)
- Flask-SQLAlchemy (para la base de datos, si es necesario)
- chromadb (para la base de datos vectorial)
- pdfplumber (para procesar archivos PDF)
- LangChain (para gestionar el flujo de procesamiento de texto y consultas)

## Instalación

1. Clonar el repositorio
   
   `git clone https://github.com/usuario/chatbot.git`
   
   `cd API-Cine`

2. Crear un entorno virtual (opcional pero recomendado)
   
   `python -m venv .venv`
   
   source .venv/bin/activate  # En Windows usa `.venv\Scripts\activate`

3. Instalar dependencias
   
   `pip install -r requirements.txt`

4. Instalar Ollama y el Modelo

   Descargar Ollama: `https://ollama.com/download`
   
   Instalar el modelo Llama 3.1 (8B): `ollama pull llama3.1:8b`

6. Configuración de la Base de Datos Vectorial

   Asegúrate de tener chromadb instalado para gestionar la base de datos vectorial y almacenar los embeddings. No es necesario configurar una base de datos externa, ya que chromadb manejará esto automáticamente.

6. Ejecutar el Chatbot Localmente 
   
   `python main.py`

Luego de estos pasos, podrás interactuar con el chatbot en la terminal.


## Notas Adicionales
- El chatbot está basado en un sistema RAG (Retrieve-Augment-Generate), lo que significa que utiliza un proceso de recuperación y generación de respuestas basadas en el contenido de los documentos.
Puedes modificar los documentos y la configuración para adaptarlo a otros contextos y temas.
- Este proyecto está en constante evolución, por lo que podrían agregarse más características en el futuro.




