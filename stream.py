import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ------------------------- CONFIG -------------------------
st.set_page_config(page_title=" Smart Product Recommender", layout="wide", page_icon="🛍️")

# ------------------------- STYLE -------------------------
st.markdown("""
<style>
:root {
    --bg: #0b0f14;
    --card: #111823;
    --accent: #00ffa2;
    --text: #e6eef3;
    --muted: #9aa6b2;
}
.stApp { background-color: var(--bg); color: var(--text); }

.title { text-align:center; font-size:32px; font-weight:700; color: var(--accent); margin-bottom:0; }
.subtitle { text-align:center; color: var(--muted); margin-bottom:40px; }

input[type="text"] {
    background: #0e1621 !important;
    color: var(--text) !important;
    border-radius: 10px;
    border: 1px solid #1f2937 !important;
    font-size:16px !important;
    padding:10px !important;
}

.stButton>button {
    background: var(--accent);
    color: #000;
    border-radius: 10px;
    border: none;
    font-weight: 700;
    padding: 10px 24px;
    transition: 0.3s;
}
.stButton>button:hover {
    background: #00ffcc;
    transform: scale(1.05);
}

.card {
    background: var(--card);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    box-shadow: 0px 0px 14px rgba(0, 255, 162, 0.06);
    transition: all 0.3s ease;
}
.card:hover {
    box-shadow: 0px 0px 20px rgba(0, 255, 162, 0.3);
    transform: translateY(-5px);
}
.prod-name { font-size:17px; font-weight:700; color: var(--text); margin-bottom:4px; }
.prod-meta { color: var(--muted); font-size:13px; }
.score { color: var(--accent); font-weight:600; }
.center { display:flex; justify-content:center; align-items:center; height:100px; color: var(--muted); }
footer { color: var(--muted); text-align:center; padding:20px 0; margin-top:40px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ------------------------- DATA + LOGIC -------------------------
@st.cache_data
def load_products(path: str):
    df = pd.read_csv(path)
    if 'product_id' not in df.columns:
        df.insert(0, 'product_id', range(1, len(df) + 1))
    for col in ['product_name', 'category', 'price', 'description']:
        if col not in df.columns:
            df[col] = ''
    return df

@st.cache_resource
def build_content_matrix(df):
    df = df.copy()
    df['combined_text'] = ((df['product_name'].fillna('') + ' ') * 5) + df['description'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    matrix = tfidf.fit_transform(df['combined_text'])
    return matrix, tfidf, df

def compute_similarity_matrix(content_matrix):
    return linear_kernel(content_matrix, content_matrix)

def tokenize_query(q):
    return re.findall(r'\w+', q.lower())

def recommend(query, df, tfidf, content_sim, top_n=10, name_boost=1.5):
    if not query.strip():
        return pd.DataFrame()

    q_vec = tfidf.transform([query])
    prod_vecs = tfidf.transform(df['combined_text'].fillna(''))
    sims_to_query = linear_kernel(q_vec, prod_vecs).flatten()
    best_idx = int(np.argmax(sims_to_query))
    sim_scores = content_sim[best_idx].copy()

    tokens = tokenize_query(query)
    if tokens:
        combined_low = df['combined_text'].str.lower().fillna('')
        mask = combined_low.apply(lambda txt: any(t in txt for t in tokens))
        sim_scores[~mask.values] = -1e9

    name_low = df['product_name'].str.lower().fillna('')
    for t in tokens:
        match_idx = name_low.str.contains(re.escape(t))
        sim_scores[match_idx.values] *= name_boost

    order = np.argsort(sim_scores)[::-1]
    order = [i for i in order if i != best_idx]
    recs = df.iloc[order[:top_n]].copy()
    recs['score'] = sim_scores[order[:top_n]]
    return recs.sort_values('score', ascending=False).reset_index(drop=True)

# ------------------------- UI -------------------------
st.markdown('<div class="title">🛍️ Smart Product Recommender</div>'
            '<div class="subtitle">Find the perfect match instantly</div>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    csv_path = st.text_input("Products CSV path", "products.csv")
    top_n = st.slider("Results to show", 3, 20, 9)
    name_boost = st.slider("Name match boost", 1.0, 3.0, 1.5, step=0.1)
    show_scores = st.checkbox("Show similarity scores", True)

try:
    products_df = load_products(csv_path)
except Exception as e:
    st.error(f"⚠️ Error loading data: {e}")
    st.stop()

content_matrix, tfidf, products_df = build_content_matrix(products_df)
content_sim = compute_similarity_matrix(content_matrix)

query = st.text_input("", placeholder="Search product (e.g., iPhone, jeans, DSLR, laptop)...")

if query:
    recs = recommend(query, products_df, tfidf, content_sim, top_n, name_boost)
    if recs.empty:
        st.warning("No matching products found.")
    else:
        n_cols = 3
        rows = int(np.ceil(len(recs) / n_cols))
        idx = 0
        for r in range(rows):
            cols = st.columns(n_cols)
            for c in range(n_cols):
                if idx >= len(recs): break
                row = recs.iloc[idx]
                with cols[c]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown(f'<div class="prod-name">{row["product_name"]}</div>', unsafe_allow_html=True)
                    cat = row.get("category", "")
                    price = row.get("price", "")
                    price_disp = f' • ₹{int(price):,}' if str(price).isdigit() else f' • {price}'
                    st.markdown(f'<div class="prod-meta">{cat}{price_disp}</div>', unsafe_allow_html=True)
                    if show_scores:
                        st.markdown(f'<div class="prod-meta">Score: <span class="score">{row["score"]:.3f}</span></div>', unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                idx += 1
else:
    st.markdown("<div class='center'>Start typing above to explore similar products 🔍</div>", unsafe_allow_html=True)

st.markdown('<footer>✨ Built with Streamlit — Dark Neon Edition</footer>', unsafe_allow_html=True)
