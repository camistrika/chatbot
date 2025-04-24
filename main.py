from db import initialize_client, generate_unique_name, create_collection, add_documents, embedding_function, delete_collection
from procesar import limpiar_documentos, convertir_a_langchain_docs, dividir_documentos, procesar_archivo
from modelo import prompt, invocar_modelo

# ----------------------------------------
# Función principal del sistema
# ----------------------------------------

def main():
    """
    Ejecuta el flujo completo del sistema:
    1. Inicializa la base de datos vectorial con un nombre único.
    2. Procesa y carga documentos en el vector store.
    3. Interactúa con el usuario para realizar consultas al modelo LLM.
    """

    # Genera nombres únicos para la base de datos y la colección
    db_name = generate_unique_name("base")
    collection_name = generate_unique_name("collection")
    model_name = "paraphrase-multilingual-mpnet-base-v2"

    # Inicializa la base de datos vectorial
    vector_store = initialize_client(db_name)
    embed_function = embedding_function(model_name)

    # Crea una nueva colección con un nombre único
    collection = create_collection(vector_store, embed_function)

    # Documentos a cargar
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

    # Agregar documentos al vector store
    texts = [doc.page_content for doc in all_splits]
    add_documents(collection, texts)
    print(f"Se añadieron {len(texts)} textos al vector store en la colección '{collection_name}'.")

    # Bucle de consulta interactiva
    while True:
        query = input("Escribe tu consulta o 'salir' para terminar: ")
        if query.lower() == "salir":
            break

        template = prompt(query, top_k=5, collection=collection)
        response = invocar_modelo(template)

        print("Respuesta del modelo:", response)


if __name__ == "__main__":
    main()

