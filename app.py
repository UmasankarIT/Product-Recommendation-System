# app.py
import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ------------------------- Config -------------------------
st.set_page_config(page_title=" Recommender", layout="wide", page_icon="🖤")

# ------------------------- Styles -------------------------
st.markdown(
    """
    <style>
    :root { --bg: #0b0f14; --card: #0f1720; --muted: #9aa6b2; --accent: #6ee7b7; --text:#e6eef3; }
    .stApp { background-color: var(--bg); color: var(--text); }
    .header { text-align:center; padding-top:14px; padding-bottom:6px;}
    .title { font-size:30px; font-weight:700; color: var(--accent); margin-bottom:0;}
    .subtitle { color: var(--muted); margin-top:0; margin-bottom:18px; }
    .search-box { display:flex; justify-content:center; margin-bottom:10px; }
    .card { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.03)); 
            border-radius:12px; padding:12px; margin-bottom:12px; color:var(--text); }
    .prod-name { font-size:16px; font-weight:700; color:var(--text); }
    .prod-meta { color:var(--muted); font-size:13px; margin-top:4px; }
    .score { color: var(--accent); font-weight:700; }
    .center { display:flex; justify-content:center; align-items:center; }
    .big-btn { background-color: #111827; color: var(--accent); border: 1px solid rgba(110,231,183,0.08);
            padding:8px 18px; border-radius:8px; font-weight:600; }
    input[type="text"] { background: #071018 !important; color: var(--text) !important; border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------- Utilities -------------------------

@st.cache_data
def load_products(path: str):
    """Load products CSV — must contain product_id, product_name (optionally description, category, price)."""
    df = pd.read_csv(path)
    # Ensure essential columns exist
    if 'product_id' not in df.columns:
        df.insert(0, 'product_id', range(1, len(df) + 1))
    if 'product_name' not in df.columns:
        raise ValueError("products.csv must contain 'product_name' column.")
    # fill missing optional columns for consistent UI
    if 'category' not in df.columns:
        df['category'] = 'Misc'
    if 'price' not in df.columns:
        df['price'] = ''
    if 'description' not in df.columns:
        df['description'] = ''
    return df

@st.cache_data
def load_ratings(path: str):
    """Load ratings CSV (optional) with columns user_id, product_id, rating."""
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=['user_id','product_id','rating'])

@st.cache_resource
def build_content_matrix(products_df: pd.DataFrame):
    """
    Build a TF-IDF matrix using product_name (heavily weighted) + description (light).
    Returns matrix, tfidf vectorizer, and the dataframe copy (with combined_text).
    """
    df = products_df.copy()
    # strongly weight product_name by repeating
    df['combined_text'] = ((df['product_name'].fillna('') + ' ') * 5) + df['description'].fillna('')
    tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    matrix = tfidf.fit_transform(df['combined_text'])
    return matrix, tfidf, df

def compute_similarity_matrix(content_matrix):
    return linear_kernel(content_matrix, content_matrix)

def tokenize_query(q: str):
    return re.findall(r'\w+', q.lower())

# ------------------------- Recommendation Logic -------------------------

def recommend(query: str,
            products_df: pd.DataFrame,
            tfidf,
            content_sim: np.ndarray,
            top_n: int = 10,
            name_boost: float = 1.5):
    """
    Hybrid-ish recommendation:
    - compute TF-IDF similarity of query vs products (via tfidf)
    - use content_sim row corresponding to best-matching product
    - apply a token filter to discourage totally unrelated matches
    - boost items whose product_name contains the query tokens
    """
    if not isinstance(query, str) or query.strip() == "":
        return pd.DataFrame()

    # find best matching product index by TF-IDF similarity to combined_text
    q_vec = tfidf.transform([query])
    prod_vecs = tfidf.transform(products_df['combined_text'].fillna(''))
    sims_to_query = linear_kernel(q_vec, prod_vecs).flatten()
    best_idx = int(np.argmax(sims_to_query))
    # base similarity scores from content_sim
    sim_scores = content_sim[best_idx].copy()

    # token filter: prefer items that contain at least one token; demote others strongly
    tokens = tokenize_query(query)
    if tokens:
        combined_low = products_df['combined_text'].str.lower().fillna('')
        mask = combined_low.apply(lambda txt: any(t in txt for t in tokens))
        if mask.sum() >= max(3, int(0.05 * len(products_df))):  # if reasonable number match, demote non-matches
            sim_scores[~mask.values] = -1e9

    # name boost: multiply scores for items whose product_name contains token
    name_low = products_df['product_name'].str.lower().fillna('')
    for t in tokens:
        match_idx = name_low.str.contains(re.escape(t))
        sim_scores[match_idx.values] = sim_scores[match_idx.values] * name_boost

    # pick top items (exclude the seed itself if present)
    order = np.argsort(sim_scores)[::-1]
    # remove seed index if it appears first
    order = [i for i in order if i != best_idx]
    top_idx = order[:top_n]
    recs = products_df.iloc[top_idx].copy()
    recs['score'] = sim_scores[top_idx]
    # sort by score descending
    recs = recs.sort_values('score', ascending=False).reset_index(drop=True)
    return recs

# ------------------------- App UI -------------------------

# Top header
st.markdown('<div class="header"><div class="title"> Product Recommendation System </div>'
            '<div class="subtitle">Type a product (e.g., "iphone", "jeans", "dslr") — results appear below</div></div>',
            unsafe_allow_html=True)

# Sidebar inputs
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    products_path = st.text_input("Products CSV", "products.csv")
    ratings_path = st.text_input("Ratings CSV (optional)", "ratings.csv")
    top_n = st.slider("Results per search", min_value=3, max_value=20, value=9)
    name_boost = st.slider("Name match boost", 1.0, 3.0, 1.5, step=0.1)
    show_scores = st.checkbox("Show internal scores", value=True)

# Load datasets
try:
    products_df = load_products(products_path)
except Exception as e:
    st.error(f"Could not load products CSV: {e}")
    st.stop()

ratings_df = load_ratings(ratings_path)  # optional, not used in current flow but kept for extension

# Build vectorizer + matrix
content_matrix, tfidf, products_df = build_content_matrix(products_df)
content_sim = compute_similarity_matrix(content_matrix)

# session state for last results
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""
if 'last_recs' not in st.session_state:
    st.session_state.last_recs = pd.DataFrame()

# centered search bar and button
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    query = st.text_input("", placeholder="Search products (e.g. iphone, jeans, camera)...", key="search_input")
    get_btn = st.button("Get Recommendations", key="get_recs_btn")

# handle action
if get_btn and query:
    recs = recommend(query, products_df, tfidf, content_sim, top_n=top_n, name_boost=name_boost)
    if recs.empty:
        st.warning("No good matches found for that query. Try another keyword.")
    else:
        st.session_state.last_query = query
        st.session_state.last_recs = recs.copy()
elif get_btn and not query:
    st.info("Please type something to search (e.g., 'iphone', 'shirt').")

# show results only if we have last results
if st.session_state.last_recs is not None and not st.session_state.last_recs.empty:
    st.markdown(f"### Results for: **{st.session_state.last_query}**")
    # layout results in 3 columns per row
    n_cols = 3
    rows = int(np.ceil(len(st.session_state.last_recs) / n_cols))
    idx = 0
    for r in range(rows):
        cols = st.columns(n_cols)
        for c in range(n_cols):
            if idx >= len(st.session_state.last_recs):
                break
            row = st.session_state.last_recs.iloc[idx]
            with cols[c]:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown(f'<div class="prod-name">{row["product_name"]}</div>', unsafe_allow_html=True)
                meta = f'{row.get("category","")}'
                price = row.get("price", "")
                if price != "":
                    try:
                        price_disp = f' • ₹{int(price):,}'
                    except Exception:
                        price_disp = f' • {price}'
                else:
                    price_disp = ""
                st.markdown(f'<div class="prod-meta">{meta}{price_disp}</div>', unsafe_allow_html=True)
                if show_scores:
                    st.markdown(f'<div class="prod-meta">Score: <span class="score">{row["score"]:.3f}</span></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            idx += 1
else:
    # If nothing searched yet, show a minimal hint area (no default product table)
    st.markdown("## Start by searching for a product above")
    st.markdown("> Tip: try short keywords like `iphone`, `jeans`, `dslr`, `sneakers`.")

# footer
st.markdown("---")
st.markdown('<div class="center"><small style="color:#6c7781">Built with Streamlit • TF-IDF name-weighted recommendations</small></div>', unsafe_allow_html=True)
