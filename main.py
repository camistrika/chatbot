from db import initialize_client, create_collection, add_documents, embedding_function
from procesar import cargar_pdf, limpiar_documentos, convertir_a_langchain_docs, dividir_documentos, procesar_archivo
from modelo import prompt, invocar_modelo
import pdfplumber
import re
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from db import initialize_client, create_collection, add_documents, embedding_function
from modelo import prompt, invocar_modelo


# ----------------------------------------
# Función principal del sistema
# ----------------------------------------

def main():
    """
    Ejecuta el flujo completo del sistema:
    1. Inicializa la base de datos vectorial.
    2. Procesa y carga documentos en el vector store.
    3. Interactúa con el usuario para realizar consultas al modelo LLM.
    """

    vector_store = initialize_client("base_prueba_f")
    collection_name = "LCD"
    model_name = "paraphrase-multilingual-mpnet-base-v2"
    

    embed_function = embedding_function(model_name)
    

    collection = create_collection(vector_store, collection_name, embed_function)
    

    archivos = [
        {"path": "archivos/datos.txt", "title": "Objetivo de la carrera"},
        {"path": "archivos/plan_datos.json", "title": "Información de la carrera"}
    ]
    
    all_splits = []
    for archivo in archivos:
        docs = procesar_archivo(archivo["path"], archivo["title"])
        docs = limpiar_documentos(docs)
        formatted_docs = convertir_a_langchain_docs(docs)
        

        all_splits.extend(dividir_documentos(formatted_docs))
    

    texts = [doc.page_content for doc in all_splits]
    add_documents(collection, texts)
    print(f"Se añadieron {len(texts)} textos al vector store.")
    

    while True:
        query = input("Escribe tu consulta o 'salir' para terminar: ")
        if query.lower() == "salir":
            break
        
        template = prompt(query, top_k=5, collection=collection)
        response = invocar_modelo(template)

        print("Respuesta del modelo:", response)


if __name__ == "__main__":
    """
    Inicia la ejecución del script.
    """
    main()
