Chatbot RAG (Retrieve-Augment-Generate)
 
 Chatbot RAG (Retrieve-Augment-Generate)
Descripción
Este proyecto implementa un chatbot basado en un sistema RAG (Retrieve-Augment-Generate). El chatbot está diseñado para responder preguntas específicas utilizando un conjunto de documentos como fuente de conocimiento. Utiliza herramientas avanzadas de procesamiento de lenguaje natural, bases de datos vectoriales y generación de respuestas contextualizadas.

Requisitos
Python 3.8 o superior
pip (gestor de paquetes)
Ollama (para ejecutar el modelo de lenguaje)
PostgreSQL (si se desea almacenar datos en una base de datos)
Flask (para el servidor web)
Flask-SQLAlchemy (para la base de datos, si es necesario)
chromadb (para la base de datos vectorial)
pdfplumber (para procesar archivos PDF)
LangChain (para gestionar el flujo de procesamiento de texto y consultas)
Instalación
1. Clonar el Repositorio
Clona el repositorio con el siguiente comando:

bash
Copiar
Editar
git clone https://github.com/camistrika/chatbot.git
Navega al directorio del proyecto:

bash
Copiar
Editar
cd chatbot
2. Crear un Entorno Virtual
Crea un entorno virtual (opcional pero recomendado):

bash
Copiar
Editar
python -m venv .venv
Activa el entorno virtual:

En Windows:
bash
Copiar
Editar
.venv\Scripts\activate
En macOS/Linux:
bash
Copiar
Editar
source .venv/bin/activate
3. Instalar Dependencias
Instala las dependencias necesarias:

bash
Copiar
Editar
pip install -r requirements.txt
4. Instalar Ollama y el Modelo
Descarga e instala el cliente Ollama desde aquí. Luego, usa el siguiente comando para obtener el modelo Llama 3.2 (3B):

bash
Copiar
Editar
ollama pull llama3.1:8b
5. Configuración de la Base de Datos Vectorial
Asegúrate de tener chromadb instalado para gestionar la base de datos vectorial y almacenar los embeddings. No es necesario configurar una base de datos externa, ya que chromadb manejará esto automáticamente.

6. Ejecutar el Chatbot Localmente
Una vez configurado todo, ejecuta el archivo principal:

bash
Copiar
Editar
python main.py
La API estará corriendo localmente y podrás interactuar con el chatbot en la terminal.

Testing con pytest
Para ejecutar las pruebas, usa los siguientes comandos:

Ejecutar pruebas:
bash
Copiar
Editar
pytest
Ver cobertura de pruebas:
bash
Copiar
Editar
coverage run -m pytest && coverage report
Notas Adicionales
El chatbot está basado en un sistema RAG (Retrieve-Augment-Generate), lo que significa que utiliza un proceso de recuperación y generación de respuestas basadas en el contenido de los documentos.
Puedes modificar los documentos y la configuración para adaptarlo a otros contextos y temas.
Este proyecto está en constante evolución, por lo que podrían agregarse más características en el futuro.
Autores
Camila Strika - Desarrolladora principal - GitHub
