import pdfplumber
import re
import json
import csv
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------------------
# Funciones para la carga de archivos
# ----------------------------------------

def cargar_pdf(pdf_path: str, doc_title: str) -> list:
    """
    Carga un archivo PDF y extrae su contenido en una lista de documentos.

    Args:
        pdf_path (str): Ruta del archivo PDF.
        doc_title (str): Título del documento.

    Returns:
        list: Lista de diccionarios con el contenido extraído y metadatos.
    """
    docs = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            content = page.extract_text()
            if content:
                docs.append({"page": i + 1, "content": content, "title": doc_title})
    return docs

def cargar_json(json_path: str, doc_title: str) -> list:
    """
    Carga un archivo JSON y lo convierte en una lista de documentos.

    Args:
        json_path (str): Ruta del archivo JSON.
        doc_title (str): Título del documento.

    Returns:
        list: Lista de diccionarios con el contenido y metadatos.

    """
    docs = []
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                content = json.dumps(item, ensure_ascii=False, indent=2)
                docs.append({"page": i + 1, "content": content, "title": doc_title})
        elif isinstance(data, dict):
            content = json.dumps(data, ensure_ascii=False, indent=2)
            docs.append({"page": 1, "content": content, "title": doc_title})
        else:
            raise ValueError("El JSON no tiene un formato compatible.")
    return docs

def cargar_csv(csv_path: str, doc_title: str) -> list:
    """
    Carga un archivo CSV y lo convierte en una lista de documentos.

    Args:
        csv_path (str): Ruta del archivo CSV.
        doc_title (str): Título del documento.

    Returns:
        list: Lista de diccionarios con el contenido y metadatos.
    """
    docs = []
    with open(csv_path, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for i, row in enumerate(reader):
            content = json.dumps(row, ensure_ascii=False, indent=2)
            docs.append({"page": i + 1, "content": content, "title": doc_title})
    return docs

def cargar_txt(txt_path: str, doc_title: str) -> list:
    """
    Carga un archivo de texto plano (TXT) y lo convierte en una lista de documentos.

    Args:
        txt_path (str): Ruta del archivo de texto.
        doc_title (str): Título del documento.

    Returns:
        list: Lista de diccionarios con el contenido y metadatos.
    """
    docs = []
    with open(txt_path, "r", encoding="utf-8") as file:
        content = file.read()
        docs.append({"page": 1, "content": content, "title": doc_title})
    return docs

# ----------------------------------------
# Funciones para procesamiento de texto
# ----------------------------------------

def limpiar_documentos(docs: list) -> list:
    """
    Limpia el contenido de los documentos eliminando espacios innecesarios.

    Args:
        docs (list): Lista de documentos.

    Returns:
        list: Lista de documentos con contenido limpio.
    """
    for doc in docs:
        doc['content'] = re.sub(r'\s+', ' ', doc['content']).strip()
    return docs

def convertir_a_langchain_docs(docs: list) -> list:
    """
    Convierte una lista de documentos en objetos Document de LangChain.

    Args:
        docs (list): Lista de diccionarios con contenido y metadatos.

    Returns:
        list: Lista de objetos Document.
    """
    return [
        Document(page_content=doc["content"], metadata={"page": doc["page"], "title": doc["title"]})
        for doc in docs
    ]

def dividir_documentos(docs: list, chunk_size: int = 700, chunk_overlap: int = 200) -> list:
    """
    Divide los documentos en fragmentos más pequeños para su procesamiento.

    Args:
        docs (list): Lista de objetos Document de LangChain.
        chunk_size (int, opcional): Tamaño de cada fragmento de texto. Por defecto, 700.
        chunk_overlap (int, opcional): Cantidad de superposición entre fragmentos. Por defecto, 200.

    Returns:
        list: Lista de fragmentos de documentos.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

# ----------------------------------------
# Función para procesar archivos
# ----------------------------------------

def procesar_archivo(file_path: str, doc_title: str) -> list:
    """
    Procesa un archivo según su extensión y devuelve una lista de documentos.

    Args:
        file_path (str): Ruta del archivo a procesar.
        doc_title (str): Título del documento.

    Returns:
        list: Lista de documentos procesados.

    Raises:
        ValueError: Si el formato del archivo no está soportado.
    """
    if file_path.endswith(".pdf"):
        return cargar_pdf(file_path, doc_title)
    elif file_path.endswith(".json"):
        return cargar_json(file_path, doc_title)
    elif file_path.endswith(".csv"):
        return cargar_csv(file_path, doc_title)
    elif file_path.endswith(".txt"):
        return cargar_txt(file_path, doc_title)
    else:
        raise ValueError("Formato de archivo no soportado. Solo se aceptan PDFs, JSON, CSV y TXT.")
