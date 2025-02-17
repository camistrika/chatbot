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

### 1. Clonar el Repositorio

```bash
git clone https://github.com/camistrika/chatbot.git
cd chatbot

