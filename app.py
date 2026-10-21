"""
Run:  streamlit run app.py
"""
import json
import math
import os
import random
from collections import Counter

import pandas as pd
import streamlit as st

from src.pipeline import (
    NGramModel, InterpolatedModel,
    load_documents, split_by_document,
    top_k_accuracy,
    BOS, EOS, UNK,
)
from src.evaluate import backoff_accuracy

# ───────────────────────────────────────────────────────── page config
st.set_page_config(page_title="Smart Keyboard V1", page_icon="⌨️", layout="wide")

# ───────────────────────────────────────────────────────── CSS (MODE CLAIR UNIQUEMENT)
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ═══════════ Thème CLAIR (Forcé de manière définitive) ═══════════ */
:root {
    color-scheme: light !important; /* Force les éléments natifs du navigateur en clair */
    --bg:
        radial-gradient(900px 520px at 6% -8%, rgba(99, 102, 241, 0.13), transparent 60%),
        radial-gradient(800px 500px at 100% 0%, rgba(14, 165, 233, 0.11), transparent 55%),
        linear-gradient(180deg, #f8faff 0%, #eef2f9 100%);
    --text: #0f172a;
    --muted: #56657d;
    --card: rgba(255, 255, 255, 0.82);
    --card-solid: #ffffff;
    --card-border: rgba(15, 23, 42, 0.10);
    --accent-1: #4f46e5;
    --accent-2: #0284c7;
    --accent-soft: rgba(79, 70, 229, 0.10);
    --shadow: 0 6px 22px rgba(15, 23, 42, 0.07);
    --shadow-lg: 0 10px 28px rgba(79, 70, 229, 0.25);
    --input-bg: #ffffff;
    --sidebar-bg: rgba(255, 255, 255, 0.90);
    /* dégradé des éléments « pleins » */
    --btn-1: #6366f1;
    --btn-2: #0ea5e9;
    --grad: linear-gradient(135deg, var(--btn-1), var(--btn-2));
}

/* ═══════════ Base ═══════════ */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background-color: #f8faff !important; /* Sécurité anti-dark mode */
    background-image: var(--bg);
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-size: cover;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"], [data-testid="stToolbar"] * { color: var(--text) !important; }
.block-container { padding-top: 2.6rem; padding-bottom: 3rem; max-width: 1280px; }

/* Texte : toujours piloté par nos variables */
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stApp p, .stApp li, .stApp label,
[data-testid="stWidgetLabel"] p,
[data-testid="stMarkdownContainer"] p {
    color: var(--text);
}
[data-testid="stCaptionContainer"] p, .stApp small { color: var(--muted) !important; }
hr { border-color: var(--card-border) !important; opacity: 1; }
a { color: var(--accent-1) !important; }

/* ═══════════ Sidebar ═══════════ */
[data-testid="stSidebar"][aria-expanded="true"] {
    min-width: 262px !important;
    max-width: 262px !important;
}
[data-testid="stSidebar"] {
    background-color: var(--sidebar-bg) !important;
    border-right: 1px solid var(--card-border);
    backdrop-filter: blur(14px);
}
[data-testid="stSidebar"] > div:first-child { background: transparent !important; }
.sidebar-brand {
    display: flex; align-items: center; gap: 0.55rem;
    font-size: 1.15rem; font-weight: 700; margin-bottom: 0.1rem;
}
.sidebar-brand svg { width: 22px; height: 22px; stroke: var(--accent-1); flex: none; }
.sidebar-brand span {
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent;
}
[data-testid="stSidebar"] [data-testid="stRadio"] label p { font-weight: 500; }
[data-testid="stSliderThumbValue"],
[data-testid="stSliderThumbValue"] * { color: var(--accent-1) !important; font-weight: 600; }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--muted) !important; }

/* ═══════════ Hero Header (glassmorphism) ═══════════ */
.hero-header {
    text-align: center; padding: 2.2rem 1rem 1.9rem; margin-bottom: 1.6rem;
    background: var(--card); border: 1px solid var(--card-border);
    border-radius: 24px; box-shadow: var(--shadow); backdrop-filter: blur(12px);
    position: relative; overflow: hidden;
}
.hero-header::before {
    content: ''; position: absolute; inset: 0; pointer-events: none;
    background: radial-gradient(620px 150px at 50% -30px, var(--accent-soft), transparent 70%);
}
.hero-title {
    font-size: 2.7rem; font-weight: 800; margin: 0; padding: 0; letter-spacing: -1px;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important;
}
.hero-sub { color: var(--muted) !important; font-size: 1rem; margin-top: 8px; font-weight: 500; }

/* ═══════════ Titres de section avec icône SVG ═══════════ */
.section-title { display: flex; align-items: center; gap: 0.55rem; margin: 0.2rem 0 0.9rem; }
.section-title svg { width: 20px; height: 20px; stroke: var(--accent-1); flex: none; }
.section-title h4 { margin: 0; padding: 0; font-size: 1.05rem; font-weight: 700; color: var(--text) !important; }

/* ═══════════ Métriques (cartes) ═══════════ */
[data-testid="stMetric"] {
    background: var(--card); border: 1px solid var(--card-border);
    padding: 0.9rem 1.2rem; border-radius: 16px; box-shadow: var(--shadow);
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    font-weight: 800; font-size: 1.7rem !important; line-height: 1.25;
    background: linear-gradient(90deg, var(--accent-1), var(--accent-2));
    -webkit-background-clip: text; background-clip: text;
    -webkit-text-fill-color: transparent; color: transparent !important;
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-weight: 500; }

/* ═══════════ Inputs ═══════════ */
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stTextInput"] [data-baseweb="base-input"],
[data-testid="stNumberInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background-color: var(--input-bg) !important;
    border-radius: 12px !important;
}
[data-testid="stTextInput"] [data-baseweb="input"],
[data-testid="stNumberInput"] [data-baseweb="input"] {
    border: 1px solid var(--card-border) !important;
    box-shadow: var(--shadow);
    transition: border-color .15s ease, box-shadow .15s ease;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: transparent !important;
    color: var(--text) !important;
    -webkit-text-fill-color: var(--text) !important;
    caret-color: var(--accent-1);
    padding: 0.7rem 0.95rem;
}
[data-testid="stTextInput"] input::placeholder { color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: .8; }
[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border-color: var(--accent-1) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
[data-testid="stNumberInput"] button { background: transparent !important; color: var(--text) !important; }
[data-testid="stNumberInput"] button svg { fill: var(--text); }

/* ═══════════ Boutons ═══════════ */
.stButton button, [data-testid^="stBaseButton"] {
    background: var(--card-solid) !important;
    color: var(--text) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px; padding: 0.55rem 1.1rem; font-weight: 600;
    box-shadow: var(--shadow);
    transition: transform .15s ease, box-shadow .15s ease, background .15s ease, border-color .15s ease;
}
.stButton button p, [data-testid^="stBaseButton"] p { color: var(--text) !important; font-weight: 600; }
.stButton button:hover, [data-testid^="stBaseButton"]:hover {
    background: var(--grad) !important;
    border-color: transparent !important;
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}
.stButton button:hover p, [data-testid^="stBaseButton"]:hover p { color: #ffffff !important; }
.stButton button:active, [data-testid^="stBaseButton"]:active { transform: translateY(0); }
.stButton button:focus-visible { outline: 2px solid var(--accent-1); outline-offset: 2px; }

/* ═══════════ Onglets ═══════════ */
.stTabs [data-baseweb="tab-list"] {
    background: var(--card); border: 1px solid var(--card-border);
    border-radius: 14px; padding: 5px; gap: 4px; box-shadow: var(--shadow);
    overflow-x: auto;
}
.stTabs [data-baseweb="tab"] {
    height: auto; min-height: 0; border-radius: 10px;
    padding: 8px 18px !important; font-weight: 600;
    color: var(--muted) !important; transition: all .18s ease;
}
.stTabs [data-baseweb="tab"] p { color: inherit !important; }
.stTabs [data-baseweb="tab"]:hover { color: var(--text) !important; background: var(--accent-soft); }
.stTabs [aria-selected="true"] {
    background: var(--grad) !important;
    color: #ffffff !important; box-shadow: 0 3px 10px rgba(99, 102, 241, 0.32);
}
.stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * { color: #ffffff !important; }
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

/* ═══════════ Alertes, expanders, tableaux, graphiques ═══════════ */
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--card-border); }
[data-testid="stExpander"] details {
    border: 1px solid var(--card-border) !important; border-radius: 14px;
    background: var(--card); box-shadow: var(--shadow);
}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color: var(--text) !important; font-weight: 600; }
[data-testid="stDataFrame"] {
    border: 1px solid var(--card-border); border-radius: 14px; overflow: hidden; box-shadow: var(--shadow);
}
[data-testid="stVegaLiteChart"], [data-testid="stImage"] {
    border: 1px solid var(--card-border); border-radius: 16px; overflow: hidden;
    padding: 0.4rem; background: var(--card); box-shadow: var(--shadow);
}

/* ═══════════ Badge ═══════════ */
.badge {
    display: inline-block; background: var(--grad);
    border-radius: 20px; padding: 3px 13px; font-size: 0.8rem; font-weight: 700; color: #ffffff !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────── ICÔNES SVG (valides partout)
_SVG_PATHS = {
    "keyboard": '<rect x="2" y="6" width="20" height="12" rx="2"/><path d="M6 10h.01M10 10h.01M14 10h.01M18 10h.01M6 14h.01M18 14h.01M9 14h6"/>',
    "balance": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "note": '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z"/>',
    "dice": '<rect x="3" y="3" width="18" height="18" rx="4"/><circle cx="8.5" cy="8.5" r="1" fill="currentColor"/><circle cx="15.5" cy="8.5" r="1" fill="currentColor"/><circle cx="8.5" cy="15.5" r="1" fill="currentColor"/><circle cx="15.5" cy="15.5" r="1" fill="currentColor"/>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><line x1="12" x2="12" y1="20" y2="10"/><line x1="18" x2="18" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="16"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "sparkles": '<path d="m12 3 1.9 5.8a2 2 0 0 0 1.3 1.3L21 12l-5.8 1.9a2 2 0 0 0-1.3 1.3L12 21l-1.9-5.8a2 2 0 0 0-1.3-1.3L3 12l5.8-1.9a2 2 0 0 0 1.3-1.3Z"/>',
}
def _svg(name):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
            f'stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f"{_SVG_PATHS[name]}</svg>")
def section(icon, text, badge=None):
    b = f"<span class='badge'>{badge}</span>" if badge else ""
    st.markdown(f"<div class='section-title'>{_svg(icon)}<h4>{text}</h4>{b}</div>", unsafe_allow_html=True)

# Couleur des graphiques : lisible sur fond clair
CHART_COLOR = "#6366f1"

# ───────────────────────────────────────────────────────── CONSTANTES
MODES = {
    "Back-off 3→2→1": "backoff",
    "Trigramme":       "trigram",
    "Interpolé 3/2/1": "interp",
    "Bigramme":        "bigram",
    "Unigramme":       "unigram",
}

# ───────────────────────────────────────────────────────── CACHES
@st.cache_data(show_spinner=False)
def load_corpus():
    docs = load_documents()
    train, test = split_by_document(docs)
    return train, test, [s for d in docs.values() for s in d]


@st.cache_resource(show_spinner="Entraînement des modèles…")
def get_models(k, use_full, lambdas):
    train, _, full = load_corpus()
    data = full if use_full else train
    return {
        "unigram": NGramModel(data, 1, k),
        "bigram":  NGramModel(data, 2, k),
        "trigram": NGramModel(data, 3, k),
        "interp":  InterpolatedModel(data, k=k, lambdas=lambdas),
    }


@st.cache_data(show_spinner="Évaluation sur le test…")
def live_eval(k, lambdas):
    _, test, _ = load_corpus()
    models = get_models(k, False, lambdas)
    rows = []
    for name, key in [("Unigramme", "unigram"), ("Bigramme", "bigram"),
                      ("Trigramme", "trigram"), ("Interpolé 3/2/1", "interp")]:
        m = models[key]
        ppl = m.perplexity(test)[0]
        acc = top_k_accuracy(m, test, 3)[0]
        rows.append({"modèle": name, "perplexité": ppl, "top-3": acc})
    bo = backoff_accuracy(models["trigram"], models["bigram"], models["unigram"], test)
    rows.append({"modèle": "Back-off 3→2→1", "perplexité": float("nan"), "top-3": bo})
    return rows


# ───────────────────────────────────────────────────────── CHARGEMENT
try:
    train_s, test_s, full_s = load_corpus()
except FileNotFoundError:
    st.error("`data/corpus.csv` introuvable. Lance `python data/make_data.py` puis recharge.")
    st.stop()

# ───────────────────────────────────────────────────────── SIDEBAR
with st.sidebar:
    st.markdown(f"<div class='sidebar-brand'>{_svg('keyboard')}<span>Smart Keyboard V1</span></div>",
                unsafe_allow_html=True)
    st.caption("Lab 02 · n-grammes")
    st.divider()

    mode_label = st.radio("Méthode de prédiction", list(MODES), index=0,
        help="Back-off = utilise le plus grand n-gramme dont le contexte a été vu.")
    mode = MODES[mode_label]

    k = st.select_slider("Lissage k", options=[0.001, 0.01, 0.1, 0.5, 1.0], value=0.1,
        help="k=0 → probas nulles ; k trop grand → masse gaspillée. Optimum trigramme : 0.01.")
    n_sug = st.slider("Nombre de suggestions", 1, 6, 3)
    use_full = st.toggle("Utiliser tout le corpus", value=False,
        help="Inclut les documents de test dans l'entraînement (fuite) → éval live désactivée.")

    with st.expander("Poids de l'interpolation"):
        l3 = st.slider("λ₃ trigramme", 0.0, 1.0, 0.6, 0.05)
        l2 = min(st.slider("λ₂ bigramme", 0.0, 1.0, 0.3, 0.05), 1.0 - l3)
        l1 = round(max(0.0, 1.0 - l3 - l2), 4)
        st.caption(f"λ₁ unigramme = {l1:.2f} (somme = 1)")
    lambdas = (l3, round(l2, 2), l1)

    st.divider()
    st.caption("Modèles locaux · aucune donnée envoyée.")

models = get_models(k, use_full, lambdas)
uni = models["unigram"]

# ───────────────────────────────────────────────────────── LOGIQUE INLINE
def tokenize(text):
    return text.lower().replace(".", " . ").split()

def split_partial(words, partial):
    if not partial or not words: return words, ""
    return words[:-1], words[-1]

def _full_dist(model, ctx_words):
    if isinstance(model, InterpolatedModel):
        hist = model.clean(ctx_words)
        return {w: model.prob_words(hist, w) for w in model.vocab if w != UNK}
    return model.next_word_distribution(ctx_words)

def _topk_from(model, ctx_words, n, prefix=""):
    dist = _full_dist(model, ctx_words)
    items = [(w, p) for w, p in dist.items() if not prefix or w.startswith(prefix)]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:n]

def predict(mode_key, ctx_words, n=3, prefix=""):
    if mode_key == "backoff":
        for key in ("trigram", "bigram", "unigram"):
            m = models[key]
            if m.seen_context(ctx_words) or m.n == 1:
                return _topk_from(m, ctx_words, n, prefix), f"{m.n}-gram", True, m
    m = models[mode_key]
    if mode_key == "interp":
        return _topk_from(m, ctx_words, n, prefix), "interp 3/2/1", True, m
    seen = m.seen_context(ctx_words)
    return _topk_from(m, ctx_words, n, prefix), f"{m.n}-gram", seen, m

def score_sentence(model, tokens):
    rows = []
    if isinstance(model, InterpolatedModel):
        words = model.clean(tokens)
        for i, w in enumerate(words + [EOS]):
            p = model.prob_words(words[:i], w)
            rows.append({"position": i + 1, "word": w, "p": p,
                         "surprise_bits": -math.log2(p) if p > 0 else float("inf"), "unknown": w == UNK})
    else:
        words = model.clean(tokens)
        padded = [BOS] * (model.n - 1) + words + [EOS]
        for i in range(model.n - 1, len(padded)):
            ctx = tuple(padded[i - model.n + 1:i])
            w = padded[i]
            p = model.prob(ctx, w)
            rows.append({"position": i - model.n + 2, "word": w, "p": p,
                         "surprise_bits": -math.log2(p) if p > 0 else float("inf"), "unknown": w == UNK})
    if not rows: return rows, None
    if any(r["p"] <= 0 for r in rows): return rows, float("inf")
    return rows, math.exp(-sum(math.log(r["p"]) for r in rows) / len(rows))

def generate(model, rng, temp=1.0, max_len=15):
    words = []
    for _ in range(max_len):
        dist = _full_dist(model, words)
        if temp != 1.0:
            dist = {w: p ** (1 / temp) for w, p in dist.items()}
            total = sum(dist.values())
            if total <= 0: break
            dist = {w: p / total for w, p in dist.items()}
        items = list(dist.items())
        w = rng.choices([x[0] for x in items], weights=[x[1] for x in items], k=1)[0]
        if w == EOS: break
        words.append(w)
    return " ".join(words)

def explain(model, ctx_words, candidates):
    if not isinstance(model, NGramModel): return None
    ctx = model.context(ctx_words)
    counts = model.ctx_counts.get(ctx, Counter())
    n_ctx = model.ctx_totals.get(ctx, 0)
    rows = [{"mot": w, "n(c,w)": counts.get(w, 0), "n(c)": n_ctx, "P": model.prob(ctx, w)} for w in candidates]
    return {"context": ctx, "n_context": n_ctx, "distinct_next": len(counts), "V": model.V, "k": model.k, "rows": rows}

def read_json(path):
    try:
        with open(path, encoding="utf-8") as f: return json.load(f)
    except (OSError, ValueError): return None

# ───────────────────────────────────────────────────────── SESSION STATE
if "text" not in st.session_state:
    st.session_state.text = "the happy girls "

def add_word(w, is_partial):
    t = st.session_state.text.rstrip()
    if w == EOS:
        st.session_state.text = ""
        return
    if is_partial:
        head = t.rsplit(" ", 1)
        t = head[0] if len(head) > 1 else ""
    st.session_state.text = (t + " " + w + " ") if t else (w + " ")

def clear_state():
    st.session_state.text = ""

# ───────────────────────────────────────────────────────── HEADER & METRICS
st.markdown("""<div class="hero-header"><h1 class="hero-title">Smart Keyboard V1</h1>
<p class="hero-sub">Suggestions n-gram · add-k smoothing · back-off &amp; interpolation</p></div>""",
    unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Vocabulaire V", uni.V)
c2.metric("Phrases train", len(full_s) if use_full else len(train_s))
c3.metric("Méthode", mode_label)
c4.metric("Lissage k", k)

if use_full:
    st.warning("Corpus complet : documents de test inclus → évaluation live désactivée.")

# ───────────────────────────────────────────────────────── TABS
t1, t2, t3, t4, t5, t6 = st.tabs([
    ":material/keyboard: Clavier",
    ":material/balance: Comparer",
    ":material/edit_note: Phrase",
    ":material/casino: Générer",
    ":material/bar_chart: Évaluation",
    ":material/info: À propos"
])

# ══════════════════════════════════ TAB 1 - CLAVIER
with t1:
    st.text_input("Début de phrase (tape ou clique sur une suggestion)", key="text", max_chars=300, placeholder="the happy girls …")
    col_a, col_b = st.columns([3, 1])
    partial = col_a.checkbox("Je suis en train d'écrire un mot (compléter le préfixe)", key="partial",
                             help="Active l'auto-complétion par préfixe, comme sur un vrai clavier.")
    col_b.button("Effacer", on_click=clear_state, use_container_width=True)

    words_all = tokenize(st.session_state.text)
    ctx_words, prefix = split_partial(words_all, partial)

    unknown = sorted({w for w in ctx_words if w not in uni.vocab})
    if unknown:
        st.warning(f"Mots hors vocabulaire (traités comme `{UNK}`) : {', '.join(unknown[:10])}")
    if not words_all:
        st.info("Texte vide : voici comment les phrases commencent d'habitude.")

    suggestions, used_label, seen, active_model = predict(mode, ctx_words, n_sug, prefix)

    if prefix and not suggestions:
        st.warning(f"Aucun mot du vocabulaire ne commence par « {prefix} ».")
    if mode in ("trigram", "bigram") and not seen and not prefix:
        st.warning("Contexte jamais vu → probabilités uniformes (tri alphabétique). Préfère le back-off ou l'interpolation.")
    if mode == "backoff" and used_label != "3-gram":
        st.info(f"Contexte trigramme inconnu → repli sur le modèle **{used_label}**.")

    if suggestions:
        section("sparkles", "Suggestions", badge=used_label)

        cols = st.columns(len(suggestions))
        for i, (col, (w, p)) in enumerate(zip(cols, suggestions)):
            label = "⏎ fin de phrase" if w == EOS else w
            col.button(f"{label}  ·  {p:.1%}", key=f"sug_{i}", on_click=add_word, args=(w, bool(prefix)),
                       use_container_width=True, help=f"P = {p:.4f}")

        top10 = _topk_from(active_model, ctx_words, 10, prefix)
        if top10:
            chart = pd.DataFrame({"probabilité": [p for _, p in top10]}, index=["⏎ fin" if w == EOS else w for w, _ in top10])
            st.bar_chart(chart, height=240, color=CHART_COLOR)

        ctx_str = " ".join(ctx_words[-2:]) or "<début de phrase>"
        st.caption(f"Contexte : « {ctx_str} »" + (f" · préfixe : « {prefix} »" if prefix else ""))

        with st.expander("Pourquoi ces suggestions ? (comptes réels)"):
            info = explain(active_model, ctx_words, [w for w, _ in suggestions])
            if info:
                st.write(f"Contexte : `{' '.join(info['context']) or '()'}` · vu **{info['n_context']}** fois, suivi de **{info['distinct_next']}** mots différents.")
                st.latex(rf"P(w\mid c)=\frac{{n(c,w)+k}}{{n(c)+k\,V}}=\frac{{n(c,w)+{info['k']}}}{{{info['n_context']}+{info['k']}\times {info['V']}}}")
                if info["n_context"] == 0:
                    st.warning(f"n(c) = 0 → P = k/(k·V) = 1/V pour tous les mots.")
                st.dataframe(pd.DataFrame(info["rows"]), use_container_width=True, hide_index=True)
            else:
                st.write(f"Modèle interpolé : P = λ₃·P₃ + λ₂·P₂ + λ₁·P₁ avec λ = {tuple(round(x, 2) for x in lambdas)}")


# ══════════════════════════════════ TAB 2 - COMPARER
with t2:
    section("balance", "Suggestions des 5 méthodes pour le texte courant")
    ctx_words, prefix = split_partial(tokenize(st.session_state.text), st.session_state.partial)
    rows = []
    for label, key in MODES.items():
        sugg, used, seen, _ = predict(key, ctx_words, 3, prefix)
        rows.append({
            "méthode": label,
            "modèle réellement utilisé": used,
            "contexte vu à l'entraînement": "oui" if seen else "non",
            "top-3": " · ".join(f"{('⏎' if w == EOS else w)} ({p:.0%})" for w, p in sugg) or "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Un trigramme sur contexte inconnu donne des suggestions arbitraires : c'est ce que corrige le back-off.")


# ══════════════════════════════════ TAB 3 - PHRASE
with t3:
    section("note", "Perplexité et surprise par mot")
    sent = st.text_input("Phrase complète (termine par `.` pour inclure la fin de phrase)",
                         "the girls watch the birds in the park .", key="sentence", max_chars=300)
    model_sent = models["trigram" if mode in ("backoff", "trigram") else mode]
    rows, ppl = score_sentence(model_sent, tokenize(sent))

    if not rows:
        st.info("Tape une phrase pour la noter.")
    else:
        df = pd.DataFrame(rows)
        df["word"] = df["word"].replace({EOS: "⏎ fin"})
        st.metric("Perplexité de la phrase", "∞" if ppl is None or math.isinf(ppl) else f"{ppl:.2f}")
        st.bar_chart(df.set_index("word")["surprise_bits"], height=220, color=CHART_COLOR)

        display = df.assign(
            probabilité=df["p"].map(lambda x: f"{x:.2%}" if x > 0 else "0"),
            **{"surprise (bits)": df["surprise_bits"].map(lambda x: "∞" if math.isinf(x) else f"{x:.2f}")},
            inconnu=df["unknown"].map({True: "oui", False: ""}),
        )
        st.dataframe(display[["position", "word", "probabilité", "surprise (bits)", "inconnu"]],
                     use_container_width=True, hide_index=True)
        if len(df):
            w_idx = df["surprise_bits"].astype(float).idxmax()
            worst = df.loc[w_idx]
            st.caption(f"Mot le plus surprenant : « {worst['word']} » ({worst['surprise_bits']:.1f} bits).")


# ══════════════════════════════════ TAB 4 - GÉNÉRER
with t4:
    section("dice", "Génération aléatoire")
    g1, g2, g3 = st.columns(3)
    temp = g1.slider("Température", 0.2, 2.0, 1.0, 0.1, help="< 1 : prudent · > 1 : aléatoire")
    n_gen = g2.slider("Nombre de phrases", 1, 10, 5)
    seed = g3.number_input("Graine aléatoire", 0, 9999, 42)
    rng = random.Random(int(seed))
    model_gen = models["trigram" if mode in ("backoff", "trigram") else mode]
    for _ in range(n_gen):
        st.write(f"• {generate(model_gen, rng, temp) or '(phrase vide)'}")


# ══════════════════════════════════ TAB 5 - ÉVALUATION
with t5:
    section("chart", "Résultats du rapport (`python -m src.evaluate`)")
    metrics = read_json("results/metrics.json")
    if metrics:
        df = pd.DataFrame(metrics["rows"])
        df["perplexity"] = df["perplexity"].map(lambda v: "∞" if v in (float("inf"), "Infinity") else f"{float(v):.2f}")
        df["zero_prob_share"] = df["zero_prob_share"].map(lambda v: f"{v:.1%}")
        df["top3"] = df["top3"].map(lambda v: f"{v:.1%}")
        st.dataframe(df.rename(columns={"model": "modèle", "perplexity": "perplexité", "zero_prob_share": "tokens p=0", "top3": "top-3"}),
                     use_container_width=True, hide_index=True)
        st.caption(f"Back-off 3→2→1 (k=0.1) : top-3 = {metrics['backoff_top3']:.1%}")
    else:
        st.info("`results/metrics.json` absent : lance `python -m src.evaluate`.")

    if os.path.exists("results/ngram_results.png"):
        st.image("results/ngram_results.png", caption="Perplexité selon k · qualité des suggestions")

    if os.path.exists("results/failures.txt"):
        with st.expander("Analyse d'échecs (`results/failures.txt`)"):
            with open("results/failures.txt", encoding="utf-8") as f:
                st.code(f.read(), language="text")

    section("chart", "Évaluation en direct (paramètres actuels)")
    if use_full:
        st.warning("Désactivée : corpus complet inclus (fuite).")
    elif st.button("Calculer maintenant"):
        with st.spinner("Évaluation…"):
            res = live_eval(k, lambdas)
        df = pd.DataFrame(res)
        df["perplexité"] = df["perplexité"].map(lambda v: "—" if isinstance(v, float) and math.isnan(v) else "∞" if isinstance(v, float) and math.isinf(v) else f"{v:.2f}")
        df["top-3"] = df["top-3"].map(lambda v: f"{v:.1%}")
        st.dataframe(df, use_container_width=True, hide_index=True)


# ══════════════════════════════════ TAB 6 - À PROPOS
with t6:
    section("info", "À propos")
    st.markdown(f"""
**Données.** Corpus synthétique (1200 phrases, 60 documents, graine 42). Le langage
est plus régulier que l'anglais réel → perplexités faibles (100+ est normal sur du texte réel).

**Évaluation honnête.** Split **par document** (12 derniers docs = test) → pas de fuite
de vocabulaire thématique entre train et test. `<unk>` absorbe les mots inconnus,
`</s>` est prédit comme n'importe quel mot.

**Limites.** Un n-gramme ne comprend pas le sens (« eat the » → plusieurs aliments
équiprobables) et ne voit que n-1 mots de contexte.

**Vie privée.** Tout tourne en local : le texte tapé n'est envoyé à aucun serveur
et n'est pas enregistré. Un clavier entraîné sur des messages personnels doit rester
sur l'appareil.

**Paramètres actuels.** V = {uni.V} mots · k = {k} · λ = {lambdas} ·
méthode = {mode_label} · corpus = {"complet" if use_full else "train only"}.
""")

st.divider()
st.caption("Lab 02 · Smart Keyboard V1 · n-grammes + add-k + back-off + interpolation")