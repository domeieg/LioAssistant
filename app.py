# import os
# import streamlit as st
# import faiss
# import pickle
# import numpy as np
# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from groq import Groq

# # Configuración inicial
# os.environ["GROG_API_KEY"] = "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"
# client = Groq(api_key=os.environ.get("GROG_API_KEY"))

# # Cargar embeddings y datos
# with open('embeddings4.pkl', 'rb') as f:
#     embeddings = pickle.load(f)

# df = pd.read_csv('fragmentos_dos.csv')  # Archivo CSV con los fragmentos
# texts = df['contenido_fragmento'].tolist()

# # Cargar el índice FAISS previamente guardado
# index = faiss.read_index('faiss_index.index')  # Carga el índice guardado

# # Cargar modelo de embeddings para consultas
# model = SentenceTransformer('all-mpnet-base-v2')

# # Función para obtener el contexto relevante del índice FAISS
# def get_context_from_faiss(query, k=2):
#     query_embedding = model.encode([query])[0]  # Generar embedding de la consulta
#     query_embedding = np.array([query_embedding], dtype=np.float32)
#     _, indices = index.search(query_embedding, k)  # Buscar los k fragmentos más relevantes
#     return [texts[i] for i in indices[0]]  # Recuperar los textos correspondientes

# # Función para obtener el sistema de prompt con contexto y lenguaje
# def get_system_prompt(context: str, language: str = "Spanish"):
#     return f"""Te llamas Lio, eres un asistente virtual experto en auditoría financiera, creado para asistir a los auditores en la ejecución de sus tareas. Tu conocimiento está basado en las Normas Internacionales de Información Financiera (NIIF), las Normas Internacionales de Contabilidad (NIC), las leyes tributarias, laborales de Ecuador y las mejores prácticas en auditoría financiera.
# Tu propósito es apoyar en todo lo relacionado con auditoría financiera, incluyendo:

# - Interpretación y aplicación de las NIIF y NIC.
# - Aplicación de las Normas Internacionales de Auditoría (NIA), incluyendo NIA 320, 315, y COSO.
# - Análisis de datos financieros, identificando posibles errores y valores aberrantes.
# - Sugerencias de procesos de auditoría eficientes y mejores prácticas.
# - Asesoramiento sobre cumplimiento con las leyes fiscales, laborales de Ecuador y normativas contables.
# - Cálculo de vacaciones, décimo tercer y cuarto sueldo, liquidación de haberes y demás cálculos relacionados basados en el código de trabajo.
# - Uso de herramientas y tablas como el Modelo de Carta de Requerimiento, la Tabla de Porcentaje de Retenciones en la FTE de IR y el Calendario de Obligaciones del SRI.
# - Análisis de normativas de sostenibilidad e impacto ambiental en auditoría financiera.
# - Recomendaciones en base a casos prácticos como el Caso Enron, Caso Coopera y la Ley Serbanex SOX.
# - Prevención de lavado de activos y evaluación de riesgos mediante Normas UAFFE, ERM y otros índices financieros.
# - Plan de cuentas.
# - Generar respuestas estructuradas

# No responderás preguntas fuera del ámbito de la auditoría financiera, la contabilidad, la normativa fiscal ecuatoriana o las leyes laborales. Tu propósito es ser un apoyo experto para todo lo relacionado con estos temas.

# Este contexto puedes usarlo para tu tarea:
# '''
# {context}
# '''

# El mensaje del usuario es lo único que dijo el usuario. Debes generar una complementación basada en ese mensaje y usar el contexto si es necesario. Si NO tienes un contexto, dile lo que sabes.
# '''"""

# # Título de la aplicación
# st.title("Lio Assistant")

# # Mostrar el mensaje de bienvenida
# st.write("¡Hola! Soy Lio, tu asistente virtual experto en auditoría financiera.")

# mensaje_inicial={
#     "role": "system",
#     "content": "¿En qué te puedo ayudar hoy?"
# }


# # Inicializa el historial del chat
# if "messages" not in st.session_state:
#     st.session_state.messages = [mensaje_inicial]

# # Mostrar los mensajes anteriores en el chat (tanto los del usuario como los del asistente)
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # Entrada de texto para la pregunta del usuario
# if prompt := st.chat_input("¿En qué puedo ayudarte?"):
#     # Obtener el contexto relevante del índice FAISS
#     context = get_context_from_faiss(prompt, k=2)
#     context_str = " ".join(context)  # Unir los fragmentos de texto para reducir el uso de tokens

#     # Crear el prompt con el contexto actualizado solo para el sistema
#     system_message = get_system_prompt(context_str, "Spanish")

#     # Agregar la pregunta del usuario al historial de la conversación
#     user_message = {
#         "role": "user",
#         "content": prompt
#     }
#     st.session_state.messages.append(user_message)

#     # Mostrar el mensaje del usuario en el chat
#     with st.chat_message("user"):
#         st.markdown(prompt)  # Mostrar pregunta del usuario

#     llm_response = client.chat.completions.create(
#         model="llama-3.1-70b-versatile",
#         messages=[
#             {"role": "system", "content": system_message}
#         ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
#     )

#     assistant_reply = llm_response.choices[0].message.content
#     # Mostrar la respuesta del asistente en el chat
#     with st.chat_message("assistant"):
#         st.markdown(assistant_reply)


#     # Agregar la respuesta del asistente al historial de la conversación
#     st.session_state.messages.append({
#         "role": "assistant",
#         "content": assistant_reply
#     })

import os
import streamlit as st
import faiss
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq

# Configuración inicial
os.environ["GROG_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROG_API_KEY"))

# Cargar embeddings y datos
with open('embeddings4.pkl', 'rb') as f:
    embeddings = pickle.load(f)

df = pd.read_csv('fragmentos_dos.csv')
texts = df['contenido_fragmento'].tolist()

# Cargar el índice FAISS previamente guardado
index = faiss.read_index('faiss_index.index')

# Cargar modelo de embeddings para consultas
model = SentenceTransformer('all-mpnet-base-v2')

# Función para obtener el contexto relevante del índice FAISS
def get_context_from_faiss(query, k=2):
    query_embedding = model.encode([query])[0]
    query_embedding = np.array([query_embedding], dtype=np.float32)
    _, indices = index.search(query_embedding, k)
    return [texts[i] for i in indices[0]]

# Función para obtener el sistema de prompt con contexto y lenguaje
def get_system_prompt(context: str, language: str = "Spanish"):
    return f"""Te llamas Lio, eres un asistente virtual experto en auditoría financiera, creado para asistir a los auditores en la ejecución de sus tareas. Tu conocimiento está basado en las Normas Internacionales de Información Financiera (NIIF), las Normas Internacionales de Contabilidad (NIC), las leyes tributarias, laborales de Ecuador y las mejores prácticas en auditoría financiera.
    ...
    Este contexto puedes usarlo para tu tarea:
    '''
    {context}
    '''
    """

# Inicialización del estado de la aplicación
if "chats" not in st.session_state:
    st.session_state.chats = {"Chat 1": []}  # Diccionario para almacenar múltiples chats
    st.session_state.current_chat = "Chat 1"  # Chat activo por defecto

# Función para manejar nuevo chat
def nuevo_chat():
    new_chat_name = f"Chat {len(st.session_state.chats) + 1}"
    st.session_state.chats[new_chat_name] = []
    st.session_state.current_chat = new_chat_name

# Función para renombrar un chat
def renombrar_chat(old_name, new_name):
    if new_name and new_name not in st.session_state.chats:
        st.session_state.chats[new_name] = st.session_state.chats.pop(old_name)
        st.session_state.current_chat = new_name

# Barra lateral para gestionar chats
with st.sidebar:
    st.title("Conversaciones anteriores")
    chat_names = list(st.session_state.chats.keys())
    selected_chat = st.selectbox("Seleccionar chat", chat_names, index=chat_names.index(st.session_state.current_chat))

    if st.button("Nuevo Chat"):
        nuevo_chat()

    st.session_state.current_chat = selected_chat

    # Opciones para renombrar y exportar chats
    with st.expander("Opciones de chat"):
        new_name = st.text_input("Renombrar chat", value=selected_chat)
        if st.button("Renombrar"):
            renombrar_chat(selected_chat, new_name)

        if st.button("Exportar chat"):
            chat_data = st.session_state.chats[selected_chat]
            chat_df = pd.DataFrame(chat_data)
            chat_df.to_csv(f"{selected_chat}.csv", index=False)
            st.success(f"Chat exportado como {selected_chat}.csv")

# Mensajes del chat actual
current_messages = st.session_state.chats[st.session_state.current_chat]

# Mostrar mensajes en la interfaz
st.title("Lio Assistant")
st.write("¡Hola! Soy Lio, tu asistente virtual experto en auditoría financiera.")

for message in current_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de texto para la pregunta del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    context = get_context_from_faiss(prompt, k=2)
    context_str = " ".join(context)

    system_message = get_system_prompt(context_str, "Spanish")

    user_message = {
        "role": "user",
        "content": prompt
    }
    current_messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    llm_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": system_message}
        ] + [{"role": m["role"], "content": m["content"]} for m in current_messages]
    )

    assistant_reply = llm_response.choices[0].message.content

    with st.chat_message("assistant"):
        st.markdown(assistant_reply)

    current_messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

# Guardar el estado del chat actual
st.session_state.chats[st.session_state.current_chat] = current_messages
