import os
import streamlit as st
from db import initialize_client, create_collection, add_documents, embedding_function
from procesar import procesar_archivo, convertir_a_langchain_docs, dividir_documentos
from modelo import prompt, invocar_modelo

# ----------------------------------------
# Configuración inicial de la aplicación
# ----------------------------------------

st.set_page_config(page_title="Chatbot RAG", layout="wide")

st.markdown(
    """
    <style>
        html, body, [class*="css"]  {
            font-size: 14px !important;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-size: 2rem !important;
        }
        section.main > div {
            max-width: 80%;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ----------------------------------------
# Inicialización de la base de datos y embeddings
# ----------------------------------------

DB_PATH = "./chroma_db"
client = initialize_client(DB_PATH)
embedding_func = embedding_function("paraphrase-multilingual-mpnet-base-v2")
collection = create_collection(client, embedding_func)


# ----------------------------------------
# Interfaz de usuario
# ----------------------------------------


st.title("📄 Chatbot Personalizado")

st.sidebar.header("Cargar documentos")
uploaded_files = st.sidebar.file_uploader("Subir archivos (PDF, JSON, CSV, TXT)", accept_multiple_files=True)


# ----------------------------------------
# Gestión de archivos temporales y estado del chat
# ----------------------------------------

temp_dir = "./temp/"
os.makedirs(temp_dir, exist_ok=True)  


if "chats" not in st.session_state:
    st.session_state["chats"] = [] 

if "active_chat" not in st.session_state:
    st.session_state["active_chat"] = None


# ----------------------------------------
# Función para iniciar un nuevo chat
# ----------------------------------------

def iniciar_nuevo_chat():
    """
    Inicia un nuevo chat con una colección nueva y vacía.
    """
    st.session_state["active_chat"] = {
        "documents": [],
        "collection": create_collection(client, embedding_func),
        "history": []
    }


# ----------------------------------------
# Procesamiento y carga de documentos
# ----------------------------------------

if uploaded_files:
    if not st.session_state["active_chat"]:
        iniciar_nuevo_chat()

    docs_procesados = []

    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        docs = procesar_archivo(file_path, file.name)
        docs_limpios = convertir_a_langchain_docs(docs)
        docs_fragmentados = dividir_documentos(docs_limpios)

        docs_procesados.extend([doc.page_content for doc in docs_fragmentados])

    add_documents(st.session_state["active_chat"]["collection"], docs_procesados)
    st.session_state["active_chat"]["documents"].extend(docs_procesados)

    st.sidebar.success("📚 Documentos procesados y almacenados en la base de datos.")


# ----------------------------------------
# Sección de Chat
# ----------------------------------------

st.subheader("💬 Chat con el asistente")

if st.session_state["active_chat"]:
    chat_container = st.container()

    with chat_container:

        for msg in st.session_state["active_chat"]["history"]:
            st.markdown(f"**👤Usuario:** {msg['query']}")
            st.markdown(f"**🤖 Asistente:** {msg['response']}")


    input_container = st.empty()
    query = st.text_input("Escribe tu pregunta aquí:", key=str(len(st.session_state["active_chat"]["history"])))


    if query:
        collection = st.session_state["active_chat"]["collection"]
        template = prompt(query, top_k=3, collection=collection)
        response = invocar_modelo(template)


        st.session_state["active_chat"]["history"].append({"query": query, "response": response})

        st.rerun()

else:
    st.warning("Carga un documento antes de hacer preguntas.")