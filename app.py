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

df = pd.read_csv(fragmentos_dos.csv')  # Archivo CSV con los fragmentos
texts = df['contenido_fragmento'].tolist()

# Cargar el índice FAISS previamente guardado
index = faiss.read_index('faiss_index.index')  # Carga el índice guardado

# Cargar modelo de embeddings para consultas
model = SentenceTransformer('all-mpnet-base-v2')

# Función para obtener el contexto relevante del índice FAISS
def get_context_from_faiss(query, k=2):
    query_embedding = model.encode([query])[0]  # Generar embedding de la consulta
    query_embedding = np.array([query_embedding], dtype=np.float32)
    _, indices = index.search(query_embedding, k)  # Buscar los k fragmentos más relevantes
    return [texts[i] for i in indices[0]]  # Recuperar los textos correspondientes

# Función para obtener el sistema de prompt con contexto y lenguaje
def get_system_prompt(context: str, language: str = "Spanish"):
    return f"""Te llamas Lio, eres un asistente virtual experto en auditoría financiera, creado para asistir a los auditores en la ejecución de sus tareas. Tu conocimiento está basado en las Normas Internacionales de Información Financiera (NIIF), las Normas Internacionales de Contabilidad (NIC), las leyes tributarias, laborales de Ecuador y las mejores prácticas en auditoría financiera.
Tu propósito es apoyar en todo lo relacionado con auditoría financiera, incluyendo:

- Interpretación y aplicación de las NIIF y NIC.
- Aplicación de las Normas Internacionales de Auditoría (NIA), incluyendo NIA 320, 315, y COSO.
- Análisis de datos financieros, identificando posibles errores y valores aberrantes.
- Sugerencias de procesos de auditoría eficientes y mejores prácticas.
- Asesoramiento sobre cumplimiento con las leyes fiscales, laborales de Ecuador y normativas contables.
- Cálculo de vacaciones, décimo tercer y cuarto sueldo, liquidación de haberes y demás cálculos relacionados basados en el código de trabajo.
- Uso de herramientas y tablas como el Modelo de Carta de Requerimiento, la Tabla de Porcentaje de Retenciones en la FTE de IR y el Calendario de Obligaciones del SRI.
- Análisis de normativas de sostenibilidad e impacto ambiental en auditoría financiera.
- Recomendaciones en base a casos prácticos como el Caso Enron, Caso Coopera y la Ley Serbanex SOX.
- Prevención de lavado de activos y evaluación de riesgos mediante Normas UAFFE, ERM y otros índices financieros.
- Plan de cuentas.
- Generar respuestas estructuradas

No responderás preguntas fuera del ámbito de la auditoría financiera, la contabilidad, la normativa fiscal ecuatoriana o las leyes laborales. Tu propósito es ser un apoyo experto para todo lo relacionado con estos temas.

Este contexto puedes usarlo para tu tarea:
'''
{context}
'''

El mensaje del usuario es lo único que dijo el usuario. Debes generar una complementación basada en ese mensaje y usar el contexto si es necesario. Si NO tienes un contexto, dile lo que sabes.
'''"""

# Título de la aplicación
st.title("Lio Assistant")

# Mostrar el mensaje de bienvenida
st.write("¡Hola! Soy Lio, tu asistente virtual experto en auditoría financiera.")

mensaje_inicial={
    "role": "system",
    "content": "¿En qué te puedo ayudar hoy?"
}


# Inicializa el historial del chat
if "messages" not in st.session_state:
    st.session_state.messages = [mensaje_inicial]

# Mostrar los mensajes anteriores en el chat (tanto los del usuario como los del asistente)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada de texto para la pregunta del usuario
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    # Obtener el contexto relevante del índice FAISS
    context = get_context_from_faiss(prompt, k=2)
    context_str = " ".join(context)  # Unir los fragmentos de texto para reducir el uso de tokens

    # Crear el prompt con el contexto actualizado solo para el sistema
    system_message = get_system_prompt(context_str, "Spanish")

    # Agregar la pregunta del usuario al historial de la conversación
    user_message = {
        "role": "user",
        "content": prompt
    }
    st.session_state.messages.append(user_message)

    # Mostrar el mensaje del usuario en el chat
    with st.chat_message("user"):
        st.markdown(prompt)  # Mostrar pregunta del usuario

    llm_response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[
            {"role": "system", "content": system_message}
        ] + [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    )

    assistant_reply = llm_response.choices[0].message.content
    # Mostrar la respuesta del asistente en el chat
    with st.chat_message("assistant"):
        st.markdown(assistant_reply)


    # Agregar la respuesta del asistente al historial de la conversación
    st.session_state.messages.append({
        "role": "assistant",
        "content": assistant_reply
    })

