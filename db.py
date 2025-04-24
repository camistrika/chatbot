import uuid
import time
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ----------------------------------------
# Funciones de inicialización
# ----------------------------------------

def initialize_model():
    """
    Inicializa el modelo de embeddings de SentenceTransformer.

    Returns:
        SentenceTransformer: Modelo para generar embeddings.
    """
    return SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

def generate_unique_name(prefix="base"):
    """
    Genera un nombre único usando un timestamp y un UUID.

    Args:
        prefix (str): Prefijo para el nombre.

    Returns:
        str: Nombre único.
    """
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:6]}"


def initialize_client(path: str) -> chromadb.PersistentClient:
    """
    Inicializa el cliente de ChromaDB con persistencia en disco.

    Args:
        path (str): Ruta para almacenar la base de datos persistente.

    Returns:
        chromadb.PersistentClient: Cliente de ChromaDB.
    """
    client = chromadb.PersistentClient(path=path)
    return client





def embedding_function(model_name: str):
    """
    Crea una función de embeddings para ChromaDB usando un modelo de SentenceTransformer.

    Args:
        model_name (str): Nombre del modelo de embeddings.

    Returns:
        embedding_functions.SentenceTransformerEmbeddingFunction: Función de embeddings para ChromaDB.
    """
    # Aquí pasamos el nombre del modelo directamente, no la función 'encode'
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name)






# ----------------------------------------
# Funciones de manipulación de colecciones
# ----------------------------------------

def create_collection(client: chromadb.PersistentClient, embedding_function) -> chromadb.Collection:
    """
    Crea una colección nueva con un nombre único en ChromaDB.

    Args:
        client (chromadb.PersistentClient): Cliente de ChromaDB.
        embedding_function: Función de embeddings.

    Returns:
        chromadb.Collection: Colección creada.
    """
    collection_name = generate_unique_name("collection")
    collection = client.create_collection(
        name=collection_name, 
        embedding_function=embedding_function, 
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Collection '{collection_name}' creada exitosamente.")
    
    return collection


def add_documents(collection: chromadb.Collection, documents: list) -> chromadb.Collection:
    """
    Añade documentos a una colección en ChromaDB.

    Args:
        collection (chromadb.Collection): Colección donde se añadirán los documentos.
        documents (list): Lista de documentos a insertar.

    Returns:
        chromadb.Collection: Colección actualizada.
    """
    collection.add(
        documents=documents, 
        ids=[str(i) for i in range(len(documents))]
    )
    return collection


# ----------------------------------------
# Funciones de consulta
# ----------------------------------------

def retrieve_data(collection: chromadb.Collection, query: str, top_k: int = 3) -> dict:
    """
    Recupera documentos relevantes de la colección según una consulta.

    Args:
        collection (chromadb.Collection): Colección a consultar.
        query (str): Consulta de búsqueda.
        top_k (int, opcional): Número de resultados a recuperar. Por defecto es 3.

    Returns:
        dict: Resultados de la consulta, incluyendo documentos y metadatos.
    """
    results = collection.query(
        query_texts=query, 
        n_results=top_k, 
        include=["documents", "metadatas"]
    )
    return results

def delete_collection(client: chromadb.PersistentClient, collection_name: str):
    """
    Elimina una colección en ChromaDB.

    Args:
        client (chromadb.PersistentClient): Cliente de ChromaDB.
        collection_name (str): Nombre de la colección a eliminar.
    """
    if collection_name in client.list_collections():
        client.delete_collection(collection_name)
        print(f"Colección '{collection_name}' eliminada exitosamente.")
    else:
        print(f"La colección '{collection_name}' no existe.")
