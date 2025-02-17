from db import retrieve_data
from langchain_ollama.llms import OllamaLLM

# ----------------------------------------
# Funciones para la generación de prompts
# ----------------------------------------

def prompt(query: str, top_k: int, collection: str) -> str:
    """
    Genera un prompt para el modelo basado en una consulta y un contexto recuperado de la base de datos.

    Args:
        query (str): Pregunta o consulta a resolver.
        top_k (int): Número máximo de documentos relevantes a recuperar.
        collection (str): Colección de la base de datos a consultar.

    Returns:
        str: Prompt formateado con el contexto y la consulta.
    """

    retrieve = retrieve_data(collection, query, top_k)
    context = retrieve['documents'][0] if retrieve['documents'] else "No hay suficiente información en el contexto."


    prompt_engineering = f"""
    Eres un asistente experto en la Licenciatura en Ciencia de Datos de la UNSAM. 
    Tu objetivo es responder de manera completa, precisa y detallada. 
    Si la pregunta involucra una lista, asegúrate de incluir todos los elementos relevantes sin omitir ninguno.

    **Contexto**:
    {context}

    **Pregunta**:
    {query}

    **Respuesta**:
    """
    return prompt_engineering

# ----------------------------------------
# Función para invocar al modelo LLM
# ----------------------------------------

def invocar_modelo(template: str, model_name: str = "llama3.2:3b") -> str:
    """
    Invoca el modelo LLM de Ollama con un prompt generado.

    Args:
        template (str): Prompt que se enviará al modelo.
        model_name (str, opcional): Nombre del modelo LLM. Por defecto "llama3.2:3b".

    Returns:
        str: Respuesta generada por el modelo LLM.
    """

    model = OllamaLLM(
        model=model_name,
        temperature=0.2
    )
    

    response = model.invoke(template)
    return response
