"""
Explorador de LLMs con Groq — Streamlit App
Permite ingresar una API key de Groq, listar los modelos disponibles
(incluyendo modelos GPT-OSS de OpenAI alojados en Groq), generar texto
ajustando parámetros del modelo, y explorar conceptos de NLP:
tokens/token IDs, bag of words, embeddings y métricas de similitud.
"""

import numpy as np
import pandas as pd
import streamlit as st
import tiktoken
from groq import Groq
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

st.set_page_config(page_title="Explorador LLM — Groq", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------------------
# Recursos cacheados
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Preparando embeddings...")
def load_embedder():
    """Embeddings basados en TF-IDF (alternativa ligera sin sentence-transformers)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(max_features=100, strip_accents='unicode')


@st.cache_data(show_spinner="Consultando modelos disponibles en Groq...")
def get_groq_models(_client, api_key):
    models = _client.models.list()
    return sorted(models.data, key=lambda m: m.id)


# ---------------------------------------------------------------------------
# Barra lateral: configuración
# ---------------------------------------------------------------------------
st.sidebar.title("⚙️ Configuración")

api_key = st.sidebar.text_input(
    "Groq API key",
    type="password",
    help="Consíguela en https://console.groq.com/keys",
)

client = None
model_list = []
model_ids = []

if api_key:
    try:
        client = Groq(api_key=api_key)
        model_list = get_groq_models(client, api_key)
        model_ids = [m.id for m in model_list]
    except Exception as e:
        st.sidebar.error(f"No se pudo conectar con Groq: {e}")

selected_model = st.sidebar.selectbox(
    "Modelo",
    options=model_ids if model_ids else ["— ingresa tu API key —"],
    index=0,
)

st.sidebar.subheader("Parámetros de generación")
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
max_tokens = st.sidebar.slider("Max tokens", 64, 8192, 512, 64)
top_p = st.sidebar.slider("Top P", 0.0, 1.0, 1.0, 0.05)

gpt_models = [m for m in model_ids if "gpt" in m.lower()]
st.sidebar.divider()
st.sidebar.caption("Modelos GPT disponibles en Groq")
st.sidebar.write("\n".join(f"- {m}" for m in gpt_models) if gpt_models else "—")


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------
st.title("🧠 Explorador de Modelos de Lenguaje (Groq)")

tab_models, tab_gen, tab_tok, tab_bow, tab_emb, tab_sim = st.tabs(
    [
        "🤖 Modelos",
        "✍️ Generación de texto",
        "🔤 Tokens / Token IDs",
        "📊 Bag of Words",
        "🧬 Embeddings",
        "📐 Similitud",
    ]
)

# --- Modelos disponibles -----------------------------------------------
with tab_models:
    st.subheader("Modelos disponibles en Groq")
    if not api_key:
        st.info("Ingresa tu API key en la barra lateral para listar los modelos.")
    elif model_list:
        df_models = pd.DataFrame(
            [
                {
                    "ID": m.id,
                    "Propietario": getattr(m, "owned_by", "-"),
                    "Ventana de contexto": getattr(m, "context_window", "-"),
                    "Activo": getattr(m, "active", "-"),
                }
                for m in model_list
            ]
        )
        st.dataframe(df_models, use_container_width=True)
    else:
        st.warning("No se pudieron obtener modelos con esta API key.")

# --- Generación de texto -------------------------------------------------
with tab_gen:
    st.subheader("Generación de texto")
    prompt = st.text_area("Prompt", height=150, placeholder="Escribe tu prompt...")
    if st.button("Generar", type="primary"):
        if not api_key or client is None:
            st.error("Ingresa una API key válida de Groq.")
        elif selected_model not in model_ids:
            st.error("Selecciona un modelo válido.")
        elif not prompt.strip():
            st.error("Escribe un prompt.")
        else:
            with st.spinner("Generando..."):
                try:
                    completion = client.chat.completions.create(
                        model=selected_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
                    st.markdown(completion.choices[0].message.content)
                    with st.expander("Detalles de uso"):
                        usage = completion.usage
                        st.json(
                            usage.model_dump() if hasattr(usage, "model_dump") else dict(usage)
                        )
                except Exception as e:
                    st.error(f"Error al generar: {e}")

# --- Tokens / Token IDs ---------------------------------------------------
with tab_tok:
    st.subheader("Tokenización: tokens y token IDs")
    text_tok = st.text_area(
        "Texto a tokenizar",
        "El procesamiento de lenguaje natural es fascinante.",
        key="tok_text",
    )
    encoding_name = st.selectbox(
        "Encoding (tiktoken)", ["cl100k_base", "o200k_base", "p50k_base"], index=0
    )
    if text_tok.strip():
        enc = tiktoken.get_encoding(encoding_name)
        ids = enc.encode(text_tok)
        pieces = [enc.decode([i]) for i in ids]
        df_tok = pd.DataFrame({"#": range(1, len(ids) + 1), "Token": pieces, "Token ID": ids})
        st.dataframe(df_tok, use_container_width=True)
        st.caption(f"Total de tokens: {len(ids)}")

# --- Bag of Words ----------------------------------------------------------
with tab_bow:
    st.subheader("Bag of Words")
    st.caption("Una oración por línea")
    bow_text = st.text_area(
        "Corpus",
        "El gato duerme en la casa\nEl perro juega en el parque\nEl gato y el perro son amigos",
        height=120,
    )
    docs = [line.strip() for line in bow_text.split("\n") if line.strip()]
    if docs:
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(docs)
        df_bow = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i + 1}" for i in range(len(docs))],
        )
        st.dataframe(df_bow, use_container_width=True)

# --- Embeddings --------------------------------------------------------
with tab_emb:
    st.subheader("Embeddings (TF-IDF)")
    emb_text = st.text_area(
        "Textos (uno por línea)",
        "El gato duerme\nEl perro corre\nLa inteligencia artificial avanza rápido",
        height=120,
        key="emb_text",
    )
    emb_docs = [line.strip() for line in emb_text.split("\n") if line.strip()]
    if st.button("Calcular embeddings"):
        if not emb_docs:
            st.error("Ingresa al menos un texto.")
        elif len(emb_docs) < 2:
            st.error("Necesitas al menos 2 textos para calcular embeddings.")
        else:
            vectorizer = load_embedder()
            vectors = vectorizer.fit_transform(emb_docs).toarray()
            st.write(f"Dimensión de cada embedding: **{vectors.shape[1]}**")
            n_preview = min(10, vectors.shape[1])
            st.dataframe(
                pd.DataFrame(
                    vectors[:, :n_preview],
                    index=emb_docs,
                    columns=[f"dim_{i}" for i in range(n_preview)],
                )
            )
            if len(emb_docs) >= 2:
                pca = PCA(n_components=2)
                coords = pca.fit_transform(vectors)
                df_plot = pd.DataFrame(coords, columns=["x", "y"])
                df_plot["texto"] = emb_docs
                st.scatter_chart(df_plot, x="x", y="y")

# --- Similitud -----------------------------------------------------------
with tab_sim:
    st.subheader("Métricas de similitud")
    col1, col2 = st.columns(2)
    with col1:
        text_a = st.text_area("Texto A", "El gato duerme en el sofá", key="sim_a")
    with col2:
        text_b = st.text_area("Texto B", "El felino descansa en el mueble", key="sim_b")
    metodo = st.radio(
        "Método de vectorización",
        ["Embeddings (sentence-transformers)", "Bag of Words"],
        horizontal=True,
    )
    if st.button("Calcular similitud"):
        if not text_a.strip() or not text_b.strip():
            st.error("Completa ambos textos.")
        else:
            if metodo.startswith("Embeddings"):
                vectorizer = load_embedder()
                vecs = vectorizer.fit_transform([text_a, text_b]).toarray()
            else:
                vectorizer = CountVectorizer()
                vecs = vectorizer.fit_transform([text_a, text_b]).toarray()

            vec_a, vec_b = vecs[0].reshape(1, -1), vecs[1].reshape(1, -1)
            cos = cosine_similarity(vec_a, vec_b)[0][0]
            euc = euclidean_distances(vec_a, vec_b)[0][0]
            dot = float(np.dot(vecs[0], vecs[1]))

            c1, c2, c3 = st.columns(3)
            c1.metric("Similitud coseno", f"{cos:.4f}")
            c2.metric("Distancia euclidiana", f"{euc:.4f}")
            c3.metric("Producto punto", f"{dot:.4f}")
