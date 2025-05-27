# Chatbot RAG (Retrieve-Augment-Generate)

## Descripción

Este proyecto implementa un chatbot basado en un sistema RAG (Retrieve-Augment-Generate). El chatbot está diseñado para responder preguntas específicas utilizando un conjunto de documentos como fuente de conocimiento. Utiliza herramientas avanzadas de procesamiento de lenguaje natural, bases de datos vectoriales y generación de respuestas contextualizadas.

## Requisitos

- Python 3.8 o superior.
- 7 GB o más de RAM libres.
- pip (gestor de paquetes).
- Ollama (para ejecutar el modelo de lenguaje).
- chromadb (para la base de datos vectorial).

## Instalación

1. Clonar el repositorio
   
   `git clone https://github.com/usuario/chatbot.git`

2. Crear un entorno virtual (opcional pero recomendado)
   
   `python -m venv .venv`
   
   source .venv/bin/activate  # En Windows usa `.venv\Scripts\activate`

3. Instalar dependencias
   
   `pip install -r requirements.txt`

4. Instalar Ollama y el Modelo

   Descargar Ollama: `https://ollama.com/download`
   
   Instalar el modelo Llama 3.1 (8B): `ollama pull llama3.1:8b`

5. Configuración de la Base de Datos Vectorial

   Asegúrate de tener chromadb instalado para gestionar la base de datos vectorial y almacenar los embeddings. No es necesario configurar una base de datos externa, ya que chromadb manejará esto automáticamente.

6. Ejecutar el Chatbot Localmente

   Hay dos opciones para ejecutar el chatbot:

      **Opción A:** Desde la terminal (modo consola)
      
      `python main.py`
   
      Esto iniciará el chatbot directamente en la terminal, permitiéndote interactuar por línea de comandos.
   
      **Opción B:** Desde una interfaz web con Streamlit
   
      `streamlit run app.py`
      
      Esto abrirá una interfaz gráfica en tu navegador para interactuar con el chatbot.


## Notas Adicionales
- El chatbot está basado en un sistema RAG (Retrieve-Augment-Generate), lo que significa que utiliza un proceso de recuperación y generación de respuestas basadas en el contenido de los documentos.
Puedes modificar los documentos y la configuración para adaptarlo a otros contextos y temas.
- Este proyecto está en constante evolución, por lo que podrían agregarse más características en el futuro.




