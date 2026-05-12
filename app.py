import streamlit as st
import numpy as np
import pickle
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EstateIQ · House Price Predictor",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0C10 !important;
    color: #E8E0D4 !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background: #0A0C10 !important;
}

[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 2rem 4rem 2rem !important; max-width: 1280px !important; }

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0F1117 0%, #141820 60%, #0A0C10 100%);
    border-bottom: 1px solid #2A2D36;
    padding: 4rem 3rem 3rem;
    margin: 0 -2rem 3rem -2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(196,160,90,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 20%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(100,140,200,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #C4A05A;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.8rem, 5vw, 4.5rem);
    font-weight: 300;
    line-height: 1.08;
    color: #F0EAE0;
    letter-spacing: -0.01em;
    margin-bottom: 1rem;
}
.hero-title span { color: #C4A05A; font-weight: 500; }
.hero-sub {
    font-size: 0.95rem;
    color: #7A7E8A;
    max-width: 480px;
    line-height: 1.7;
}

/* ── Section headers ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #C4A05A;
    margin-bottom: 1.2rem;
    margin-top: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #2A2D36, transparent);
}

/* ── Card panels ── */
.card {
    background: #111318;
    border: 1px solid #1E2128;
    border-radius: 12px;
    padding: 1.8rem 1.6rem;
    margin-bottom: 1.2rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #2E3240; }

/* ── Streamlit widget overrides ── */
label, .stSelectbox label, .stNumberInput label,
div[data-baseweb="select"] label, .stCheckbox label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    color: #9498A6 !important;
    letter-spacing: 0.04em !important;
    margin-bottom: 0.3rem !important;
}

div[data-baseweb="select"] > div {
    background: #161921 !important;
    border: 1px solid #272B36 !important;
    border-radius: 8px !important;
    color: #E8E0D4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="select"] > div:focus-within {
    border-color: #C4A05A !important;
    box-shadow: 0 0 0 3px rgba(196,160,90,0.08) !important;
}

input[type="number"], .stNumberInput input {
    background: #161921 !important;
    border: 1px solid #272B36 !important;
    border-radius: 8px !important;
    color: #E8E0D4 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.55rem 0.8rem !important;
    transition: border-color 0.2s !important;
}
input[type="number"]:focus {
    border-color: #C4A05A !important;
    box-shadow: 0 0 0 3px rgba(196,160,90,0.08) !important;
    outline: none !important;
}

.stCheckbox > label {
    background: #161921 !important;
    border: 1px solid #272B36 !important;
    border-radius: 8px !important;
    padding: 0.55rem 1rem !important;
    color: #C8C2BA !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s, background 0.2s !important;
    cursor: pointer !important;
}
.stCheckbox > label:hover {
    border-color: #C4A05A !important;
    background: #1A1D26 !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #C4A05A 0%, #A8844A 100%) !important;
    color: #0A0C10 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.75rem 2.5rem !important;
    width: 100% !important;
    height: auto !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.15s !important;
    box-shadow: 0 4px 20px rgba(196,160,90,0.25) !important;
}
div[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(196,160,90,0.35) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
    background: linear-gradient(135deg, #13160E 0%, #111318 100%);
    border: 1px solid #2B3020;
    border-radius: 14px;
    padding: 2.4rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 60%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #C4A05A, transparent);
}
.result-label {
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #5A6040;
    margin-bottom: 0.6rem;
    font-weight: 500;
}
.result-price {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.4rem, 5vw, 3.6rem);
    font-weight: 500;
    color: #C4A05A;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}
.result-amount {
    font-size: 0.88rem;
    color: #6A7060;
    margin-top: 0.4rem;
}
.result-badge {
    display: inline-block;
    background: rgba(196,160,90,0.1);
    border: 1px solid rgba(196,160,90,0.2);
    border-radius: 100px;
    padding: 0.25rem 0.9rem;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #C4A05A;
    margin-top: 1.2rem;
}

/* ── Info pill ── */
.info-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #161921;
    border: 1px solid #1E2128;
    border-radius: 100px;
    padding: 0.3rem 0.9rem;
    font-size: 0.75rem;
    color: #6A6E7A;
    margin-bottom: 2rem;
}

/* ── Metric tiles ── */
.metric-tile {
    background: #111318;
    border: 1px solid #1E2128;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-val {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.6rem;
    font-weight: 500;
    color: #E8E0D4;
}
.metric-key {
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4E5260;
    margin-top: 0.2rem;
}

/* ── Divider ── */
hr.fancy {
    border: none;
    border-top: 1px solid #1A1D26;
    margin: 2rem 0;
}

/* ── Dropdown menu dark ── */
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #161921 !important;
    border: 1px solid #272B36 !important;
    border-radius: 8px !important;
}
li[role="option"] {
    color: #C8C2BA !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
}
li[role="option"]:hover { background: #1E222E !important; }

/* ── Streamlit top padding fix ── */
#root > div:first-child { margin-top: 0 !important; }
.main > div:first-child { padding-top: 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0C10; }
::-webkit-scrollbar-thumb { background: #222530; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #2E3340; }
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">🏛️ &nbsp; EstateIQ · AI Valuation Engine</div>
    <div class="hero-title">Predict your property's<br><span>true market value</span></div>
    <div class="hero-sub">Enter property details below and let the model surface an intelligent price estimate — powered by real market data.</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="info-pill">⚡ &nbsp; Model inputs auto-encode categorical features before inference</div>', unsafe_allow_html=True)

# ── Load model (optional) ─────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None

model = load_model()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Basic Details
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">01 &nbsp; Core Property Details</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        index_val    = st.number_input("Property Index", min_value=0, value=69244, step=1)
    with c2:
        carpet_area  = st.number_input("Carpet Area (sq ft)", min_value=100, value=1000, step=10)
    with c3:
        super_area   = st.number_input("Super Area (sq ft)", min_value=100, value=1250, step=10)
    with c4:
        bhk          = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        bathroom     = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)
    with c6:
        balcony      = st.number_input("Balconies", min_value=0, max_value=10, value=2, step=1)
    with c7:
        floor_num    = st.number_input("Floor Number", min_value=0, value=1, step=1)
    with c8:
        total_floor  = st.number_input("Total Floors", min_value=1, value=10, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Status, Transaction & Furnishing
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">02 &nbsp; Status · Transaction · Furnishing</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        status_rtm   = st.checkbox("Ready to Move", value=True)
    with c2:
        transaction  = st.selectbox("Transaction Type", ["New Property", "Resale"])
    with c3:
        furnishing   = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
    st.markdown('</div>', unsafe_allow_html=True)

# encode
txn_new      = 1 if transaction == "New Property" else 0
txn_resale   = 1 if transaction == "Resale"       else 0
furn_f       = 1 if furnishing  == "Furnished"    else 0
furn_sf      = 1 if furnishing  == "Semi-Furnished" else 0
furn_uf      = 1 if furnishing  == "Unfurnished"  else 0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 · Facing & Overlooking
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">03 &nbsp; Facing Direction & Overlooking View</div>', unsafe_allow_html=True)

FACING_OPTIONS = [
    "East", "North", "North - East", "North - West",
    "South", "South - East", "South -West", "West"
]
OVERLOOKING_OPTIONS = [
    "Garden/Park",
    "Garden/Park, Main Road",
    "Garden/Park, Main Road, Pool",
    "Garden/Park, Not Available",
    "Garden/Park, Pool",
    "Garden/Park, Pool, Main Road",
    "Garden/Park, Pool, Main Road, Not Available",
    "Main Road",
    "Main Road, Garden/Park",
    "Main Road, Garden/Park, Pool",
    "Main Road, Pool",
    "Main Road, Pool, Garden/Park",
    "Pool",
    "Pool, Garden/Park",
    "Pool, Garden/Park, Main Road",
    "Pool, Main Road",
    "Pool, Main Road, Garden/Park",
    "Pool, Main Road, Not Available",
]

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        facing      = st.selectbox("Facing Direction", FACING_OPTIONS)
    with c2:
        overlooking = st.selectbox("Overlooking View", OVERLOOKING_OPTIONS)
    st.markdown('</div>', unsafe_allow_html=True)

# one-hot facing
facing_map = {o: 0 for o in FACING_OPTIONS}
facing_map[facing] = 1

# one-hot overlooking
overlooking_map = {o: 0 for o in OVERLOOKING_OPTIONS}
overlooking_map[overlooking] = 1

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 · Ownership & Encoded IDs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">04 &nbsp; Ownership · Society · Location</div>', unsafe_allow_html=True)

OWNERSHIP_OPTIONS = [
    "Co-operative Society", "Freehold", "Leasehold", "Power Of Attorney"
]

with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ownership        = st.selectbox("Ownership Type", OWNERSHIP_OPTIONS)
    with c2:
        society_encoded  = st.number_input("Society Encoded ID", min_value=0, value=0, step=1,
                                           help="Numeric label-encoded value for the society")
    with c3:
        location_encoded = st.number_input("Location Encoded ID", min_value=0, value=0, step=1,
                                           help="Numeric label-encoded value for the location")
    st.markdown('</div>', unsafe_allow_html=True)

# one-hot ownership
own_coop = 1 if ownership == "Co-operative Society" else 0
own_free = 1 if ownership == "Freehold"            else 0
own_leas = 1 if ownership == "Leasehold"           else 0
own_poa  = 1 if ownership == "Power Of Attorney"   else 0

# ─────────────────────────────────────────────────────────────────────────────
# Build feature vector (matches X_train column order)
# ─────────────────────────────────────────────────────────────────────────────
def build_feature_vector():
    return np.array([[
        index_val, carpet_area, bathroom, balcony, super_area,
        floor_num, total_floor, bhk,
        int(status_rtm),            # Status_Ready to Move
        txn_new, txn_resale,        # Transaction_New Property / Resale
        furn_f, furn_sf, furn_uf,   # Furnishing
        # facing (8)
        facing_map["East"],
        facing_map["North"],
        facing_map["North - East"],
        facing_map["North - West"],
        facing_map["South"],
        facing_map["South - East"],
        facing_map["South -West"],
        facing_map["West"],
        # overlooking (18)
        overlooking_map["Garden/Park"],
        overlooking_map["Garden/Park, Main Road"],
        overlooking_map["Garden/Park, Main Road, Pool"],
        overlooking_map["Garden/Park, Not Available"],
        overlooking_map["Garden/Park, Pool"],
        overlooking_map["Garden/Park, Pool, Main Road"],
        overlooking_map["Garden/Park, Pool, Main Road, Not Available"],
        overlooking_map["Main Road"],
        overlooking_map["Main Road, Garden/Park"],
        overlooking_map["Main Road, Garden/Park, Pool"],
        overlooking_map["Main Road, Pool"],
        overlooking_map["Main Road, Pool, Garden/Park"],
        overlooking_map["Pool"],
        overlooking_map["Pool, Garden/Park"],
        overlooking_map["Pool, Garden/Park, Main Road"],
        overlooking_map["Pool, Main Road"],
        overlooking_map["Pool, Main Road, Garden/Park"],
        overlooking_map["Pool, Main Road, Not Available"],
        # ownership (4)
        own_coop, own_free, own_leas, own_poa,
        # encoded
        society_encoded, location_encoded,
    ]])

# ─────────────────────────────────────────────────────────────────────────────
# Summary metrics
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<hr class="fancy">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Summary Snapshot</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
tiles = [
    (f"{carpet_area} sq ft", "Carpet Area"),
    (f"{bhk} BHK / {bathroom}B",  "Config"),
    (f"Floor {floor_num}/{total_floor}", "Floor"),
    (furnishing,              "Furnishing"),
    (transaction,             "Transaction"),
]
for col, (val, key) in zip([m1, m2, m3, m4, m5], tiles):
    col.markdown(f"""
    <div class="metric-tile">
        <div class="metric-val">{val}</div>
        <div class="metric-key">{key}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Predict button
# ─────────────────────────────────────────────────────────────────────────────
_, btn_col, _ = st.columns([1.5, 2, 1.5])
with btn_col:
    predict_clicked = st.button("✦  Generate Valuation")

# ─────────────────────────────────────────────────────────────────────────────
# Result
# ─────────────────────────────────────────────────────────────────────────────
if predict_clicked:
    X = build_feature_vector()

    if model is not None:
        prediction = model.predict(X)
        # Handle multi-output: [price_per_sqft, amount]
        if hasattr(prediction[0], '__len__') and len(prediction[0]) == 2:
            price_per_sqft = float(prediction[0][0])
            amount         = float(prediction[0][1])
        else:
            price_per_sqft = float(prediction[0])
            amount         = price_per_sqft * carpet_area
    else:
        # Demo values when no model.pkl is present
        price_per_sqft = 8500.0
        amount         = price_per_sqft * carpet_area
        st.info("ℹ️  No `model.pkl` found — showing demo output. Place your trained model file in the same directory.", icon="💡")

    amount_cr  = amount / 1e7
    amount_lk  = amount / 1e5
    display_cr = f"₹ {amount_cr:.2f} Cr" if amount_cr >= 1 else f"₹ {amount_lk:.1f} L"

    st.markdown("<br>", unsafe_allow_html=True)
    _, res_col, _ = st.columns([0.5, 3, 0.5])
    with res_col:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Market Value</div>
            <div class="result-price">{display_cr}</div>
            <div class="result-amount">₹ {amount:,.0f} &nbsp;·&nbsp; ₹ {price_per_sqft:,.0f} / sq ft</div>
            <div class="result-badge">✦ &nbsp; AI Valuation Complete</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        d1, d2, d3 = st.columns(3)
        d1.markdown(f"""<div class="metric-tile">
            <div class="metric-val">₹ {price_per_sqft:,.0f}</div>
            <div class="metric-key">Price / sq ft</div></div>""", unsafe_allow_html=True)
        d2.markdown(f"""<div class="metric-tile">
            <div class="metric-val">{carpet_area} ft²</div>
            <div class="metric-key">Carpet Area</div></div>""", unsafe_allow_html=True)
        d3.markdown(f"""<div class="metric-tile">
            <div class="metric-val">₹ {amount/1e5:,.1f} L</div>
            <div class="metric-key">Total (Lakhs)</div></div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<br><br>
<div style="text-align:center; font-size:0.7rem; color:#2E3240; letter-spacing:0.12em; text-transform:uppercase;">
    EstateIQ &nbsp;·&nbsp; House Price Prediction Engine &nbsp;·&nbsp; Place <code style="color:#3A4050">model.pkl</code> in the same folder to activate live inference
</div>
""", unsafe_allow_html=True)