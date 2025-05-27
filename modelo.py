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
Sos un asistente virtual diseñado para responder exclusivamente en base al contenido de los documentos proporcionados por el usuario.

**Tu objetivo:**  
Brindar respuestas claras, completas y precisas, utilizando únicamente la información contenida en dichos documentos.

**Reglas fundamentales:**
- No inventes información. Si la respuesta no está en los documentos, indicá con claridad que no podés responder.
- No utilices conocimientos externos. Limitate estrictamente al contenido disponible.
- Usá un tono profesional, claro, útil y muy amigable.
- Si la información está distribuida en varios documentos, integrala de forma coherente y precisa.
- Mantené el contexto entre preguntas, salvo que el usuario indique lo contrario.

**Checklist antes de responder:**
Revisá que tu respuesta cumpla con todos los siguientes puntos:
1. ¿La respuesta se basa **solo** en el contenido del contexto?
2. ¿Es clara, completa, relevante y útil?
3. ¿Evita suposiciones o información externa?
4. Si no hay datos suficientes en los documentos, ¿avisás explícitamente que no podés responder?

**Respondé solo si podés contestar con certeza cumpliendo todo lo anterior.**

---

**Contexto disponible**:
{context}

**Pregunta del usuario**:
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
