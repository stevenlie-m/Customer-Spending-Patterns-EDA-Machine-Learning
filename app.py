import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Spending Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * { color: white !important; }

.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.06);
    border-left: 4px solid;
    transition: transform .15s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-val   { font-size: 2rem; font-weight: 700; line-height: 1.1; margin: .3rem 0 .15rem; }
.kpi-label { font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.kpi-delta { font-size: .78rem; font-weight: 500; margin-top: .2rem; }

.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    background: linear-gradient(90deg, #334155 0%, #475569 100%);
    border-radius: 8px;
    padding: .55rem 1rem;
    margin: 1.4rem 0 .9rem;
    letter-spacing: .01em;
}

.insight-box {
    background: #f8fafc;
    border-radius: 10px;
    padding: .85rem 1.1rem;
    border-left: 3px solid #6366f1;
    margin: .6rem 0 1rem;
    font-size: .85rem;
    color: #334155;
    line-height: 1.6;
}
.insight-box b { color: #4f46e5; }

.cluster-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    box-shadow: 0 1px 8px rgba(0,0,0,.08);
    border-top: 4px solid;
    margin-bottom: .5rem;
}
.cluster-label { font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.cluster-name  { font-size: 1.1rem; font-weight: 700; margin: .2rem 0 .4rem; }
.cluster-desc  { font-size: .82rem; color: #475569; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    df = pd.read_csv("shopping_behavior_updated.csv")
    df = df.rename(columns={"Purchase Amount (USD)": "Purchase Amount"})
    df["Frequency of Purchases"] = df["Frequency of Purchases"].replace(
        {"Bi-Weekly": "Fortnightly", "Every 3 Months": "Quarterly"}
    )
    df = df.drop(columns=["Discount Applied"])
    df["Age Group"]    = pd.cut(df["Age"], bins=[17,25,35,45,55,70],
                                labels=["18-25","26-35","36-45","46-55","56+"])
    df["Spending Tier"]= pd.qcut(df["Purchase Amount"], q=3, labels=["Low","Mid","High"])
    df["Loyalty Tier"] = pd.qcut(df["Previous Purchases"], q=3, labels=["New","Regular","Loyal"])
    urutan_freq = ["Weekly","Fortnightly","Monthly","Quarterly","Annually"]
    freq_map    = {f: i for i, f in enumerate(urutan_freq[::-1])}
    df["Has Promo"]    = (df["Promo Code Used"] == "Yes").astype(int)
    df["Freq Ordinal"] = df["Frequency of Purchases"].map(freq_map).fillna(0).astype(int)
    return df

@st.cache_data
def run_clustering(df_input):
    urutan_freq = ["Weekly","Fortnightly","Monthly","Quarterly","Annually"]
    results = {}
    configs = {
        "A": {
            "label": "RFM-Proxy",
            "feats": ["Previous Purchases","Purchase Amount","Freq Ordinal"],
            "desc" : "Loyalitas & CRM"
        },
        "B": {
            "label": "Behavioral",
            "feats": ["Purchase Amount","Freq Ordinal","Review Rating","Has Promo"],
            "desc" : "Strategi Promosi"
        },
        "C": {
            "label": "Demo-Behavioral",
            "feats": ["Age","Purchase Amount","Previous Purchases","Freq Ordinal"],
            "desc" : "Targeting & Channel"
        },
    }
    for key, cfg in configs.items():
        X      = df_input[cfg["feats"]].values
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        # Find best k via silhouette
        best_k, best_sil = 3, -1
        sil_scores, inertias = [], []
        for k in range(2, 7):
            km  = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_sc)
            s   = silhouette_score(X_sc, lbl)
            sil_scores.append(s)
            inertias.append(km.inertia_)
            if s > best_sil:
                best_sil, best_k = s, k
        # Fit final model
        km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels   = km_final.fit_predict(X_sc)
        # PCA 2D
        pca     = PCA(n_components=2, random_state=42)
        X_pca   = pca.fit_transform(X_sc)
        results[key] = {
            "feats"      : cfg["feats"],
            "label"      : cfg["label"],
            "desc"       : cfg["desc"],
            "k"          : best_k,
            "sil"        : best_sil,
            "labels"     : labels,
            "X_sc"       : X_sc,
            "X_pca"      : X_pca,
            "var_exp"    : pca.explained_variance_ratio_,
            "inertias"   : inertias,
            "sil_scores" : sil_scores,
            "scaler"     : scaler,
        }
    return results

df         = load_data()
URUTAN_FREQ= ["Weekly","Fortnightly","Monthly","Quarterly","Annually"]
PAL        = sns.color_palette("Set2", 8)
CMAP       = "YlOrRd"

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(val, label, color, delta=""):
    return f"""<div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val" style="color:{color}">{val}</div>
        <div class="kpi-delta">{delta}</div></div>"""

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">📌 {text}</div>', unsafe_allow_html=True)

def base_fig(w=6, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax

def cluster_card(name, size, pct, desc, color):
    st.markdown(
        f'<div class="cluster-card" style="border-top-color:{color}">'
        f'<div class="cluster-label">Segment</div>'
        f'<div class="cluster-name" style="color:{color}">{name}</div>'
        f'<div class="cluster-desc">{desc}</div>'
        f'<div style="margin-top:.5rem;font-size:.8rem;color:#94a3b8">'
        f'{size:,} pelanggan &nbsp;·&nbsp; {pct:.1f}%</div></div>',
        unsafe_allow_html=True
    )

PAL_HEX = ["#66c2a5","#fc8d62","#8da0cb","#e78ac3","#a6d854","#ffd92f","#e5c494","#b3b3b3"]

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛒 Shopping Dashboard")
    st.markdown("---")
    page = st.radio("Navigasi", [
        "📊 Overview",
        "👤 Demografi",
        "💳 Pola Spending",
        "📦 Produk & Kategori",
        "🚚 Free Shipping",
        "🔵 Clustering",
        "📝 Kesimpulan & Saran",
    ])
    st.markdown("---")
    st.markdown("**Filter Global**")
    gender_f  = st.multiselect("Gender", df["Gender"].unique().tolist(),
                                default=df["Gender"].unique().tolist())
    season_f  = st.multiselect("Musim",  df["Season"].unique().tolist(),
                                default=df["Season"].unique().tolist())
    age_min, age_max = st.slider("Rentang Usia",
                                  int(df["Age"].min()), int(df["Age"].max()), (18, 70))
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} records · 17 fitur")

mask = (df["Gender"].isin(gender_f) &
        df["Season"].isin(season_f) &
        df["Age"].between(age_min, age_max))
dff = df[mask]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Spending Patterns")
    st.caption("Exploratory Data Analysis — Shopping Behavior Dataset")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi(f"{len(dff):,}","Total Transaksi","#6366f1",
                    f"{len(dff)/len(df)*100:.0f}% dari total"), unsafe_allow_html=True)
    c2.markdown(kpi(f"${dff['Purchase Amount'].mean():.0f}","Avg Spending","#10b981",
                    f"Median ${dff['Purchase Amount'].median():.0f}"), unsafe_allow_html=True)
    c3.markdown(kpi(f"${dff['Purchase Amount'].sum():,.0f}","Total Revenue","#f59e0b",
                    "seluruh transaksi"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{dff['Review Rating'].mean():.2f} ★","Avg Rating","#ef4444",
                    "dari 5.0"), unsafe_allow_html=True)
    c5.markdown(kpi(f"{(dff['Subscription Status']=='Yes').mean()*100:.0f}%",
                    "Subscriber Rate","#8b5cf6",
                    f"{(dff['Subscription Status']=='Yes').sum():,} pelanggan"),
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    section("Distribusi Purchase Amount & Revenue per Kategori")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_fig()
        ax.hist(dff["Purchase Amount"], bins=30, color=PAL[0], edgecolor="white", linewidth=0.6)
        ax.axvline(dff["Purchase Amount"].mean(), color="red", linestyle="--", linewidth=1.8,
                   label=f"Mean=${dff['Purchase Amount'].mean():.0f}")
        ax.axvline(dff["Purchase Amount"].median(), color="navy", linestyle=":", linewidth=1.8,
                   label=f"Median=${dff['Purchase Amount'].median():.0f}")
        ax.set_title("Distribusi Purchase Amount", fontweight="bold")
        ax.set_xlabel("USD"); ax.set_ylabel("Frekuensi"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight("Distribusi <b>uniform sempurna $20–$100</b>. Mean=59.8 ≈ Median=60.0 — distribusi simetris tanpa skewness. Tidak ada harga pembelian yang lebih populer dari harga lainnya.")

    with c2:
        cat_rev = dff.groupby("Category")["Purchase Amount"].sum().sort_values()
        fig, ax = base_fig()
        bars = ax.barh(cat_rev.index, cat_rev.values, color=PAL[:len(cat_rev)], edgecolor="white")
        ax.set_title("Total Revenue per Kategori", fontweight="bold")
        ax.set_xlabel("Total Revenue (USD)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        for bar in bars:
            ax.text(bar.get_width()+200, bar.get_y()+bar.get_height()/2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight("<b>Clothing</b> mendominasi total revenue karena volume transaksi terbesar (1.737 item). Bukan karena harga tertinggi — semua kategori rata-rata $57–$63, dengan <b>Outerwear justru terendah ($57)</b>.")

    section("Heatmap Rata-rata Spending: Kategori × Musim")
    pivot = dff.pivot_table(values="Purchase Amount", index="Category", columns="Season", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=CMAP, linewidths=0.5, ax=ax,
                cbar_kws={"label":"Avg Purchase (USD)"})
    ax.set_title("Rata-rata Spending per Kategori & Musim", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("<b>Footwear-Fall tertinggi (63.7)</b>. <b>Outerwear terendah</b> di hampir semua musim (54.6–59.8). Range keseluruhan hanya $9 → tidak ada pola musiman yang dramatis. Spring paling rendah di hampir semua kategori.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMOGRAFI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Demografi":
    st.title("👤 Analisis Demografi Pelanggan")

    section("Distribusi Variabel Numerik")
    cols_num = st.columns(4)
    for col, (var, lbl, color) in zip(cols_num, [
        ("Age","Usia Pelanggan (tahun)",PAL[0]),
        ("Purchase Amount","Jumlah Pembelian (USD)",PAL[1]),
        ("Review Rating","Rating Ulasan (1–5)",PAL[2]),
        ("Previous Purchases","Riwayat Pembelian",PAL[3]),
    ]):
        fig, ax = plt.subplots(figsize=(4,3.2))
        ax.hist(dff[var], bins=25, color=color, edgecolor="white", alpha=0.9)
        ax.axvline(dff[var].mean(), color="red", linestyle="--", linewidth=1.6,
                   label=f"Mean={dff[var].mean():.1f}")
        ax.axvline(dff[var].median(), color="navy", linestyle=":", linewidth=1.6,
                   label=f"Median={dff[var].median():.1f}")
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.legend(fontsize=7); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); col.pyplot(fig); plt.close()

    insight("""<b>Age</b>: Uniform merata 18–70 tahun, Mean=44.1 ≈ Median=44.0.&nbsp;|&nbsp;
<b>Purchase Amount</b>: Uniform $20–$100, Mean=59.8 ≈ Median=60.0 — simetris sempurna.&nbsp;|&nbsp;
<b>Review Rating</b>: <b>Flat/uniform</b> di rentang 2.5–5.0, Mean=Median=3.7 — tidak ada skewness, rating merata di semua level.&nbsp;|&nbsp;
<b>Previous Purchases</b>: <b>Hampir uniform</b> 0–50, Mean=25.4 ≈ Median=25.0 — tidak ada dominasi pelanggan baru maupun lama.""")

    section("Distribusi Variabel Kategorikal")
    row1 = st.columns(4); row2 = st.columns(4)
    for ax_col, col in zip(row1+row2, ["Gender","Category","Season","Subscription Status",
                                        "Shipping Type","Payment Method","Size","Frequency of Purchases"]):
        vc = dff[col].value_counts().reindex(URUTAN_FREQ).dropna() if col=="Frequency of Purchases" \
             else dff[col].value_counts()
        fig, ax = plt.subplots(figsize=(4,3))
        bars = ax.bar(vc.index, vc.values, color=PAL[:len(vc)], edgecolor="white")
        ax.set_title(col, fontsize=8, fontweight="bold"); ax.set_ylabel("Count",fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(int(bar.get_height())), ha="center", fontsize=6, color="#333")
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()

    insight("""<b>Gender</b>: Pria=2.652 (~68%) vs Wanita=1.248 (~32%).&nbsp;|&nbsp;
<b>Category</b>: Clothing terbanyak (1.737), diikuti Accessories (1.240).&nbsp;|&nbsp;
<b>Season</b>: Sangat merata (955–999).&nbsp;|&nbsp;
<b>Subscription</b>: No=2.847 (73%) vs Yes=1.053 (27%).&nbsp;|&nbsp;
<b>Shipping & Payment</b>: Sangat merata di semua pilihan (627–675 dan 612–677) — tidak ada dominan.&nbsp;|&nbsp;
<b>Size</b>: M mendominasi (1.755).&nbsp;|&nbsp;
<b>Frequency</b>: <b>Quarterly terbanyak (1.147)</b>, Weekly terendah (539).""")

    section("Rata-rata Spending per Kelompok Usia & Gender")
    c1, c2 = st.columns(2)
    with c1:
        age_sp = dff.groupby("Age Group", observed=True)["Purchase Amount"].mean()
        fig, ax = base_fig(6, 3.8)
        ax.plot(age_sp.index, age_sp.values, marker="o", color=PAL[2], linewidth=2, markersize=7)
        ax.fill_between(range(len(age_sp)), age_sp.values, alpha=0.15, color=PAL[2])
        ax.set_xticks(range(len(age_sp))); ax.set_xticklabels(age_sp.index)
        ax.set_title("Avg Spending per Kelompok Usia", fontweight="bold")
        ax.set_ylabel("Avg Purchase (USD)"); ax.set_ylim(50, 75)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        gen_sp = dff.groupby(["Age Group","Gender"], observed=True)["Purchase Amount"].mean().unstack()
        fig, ax = base_fig(6, 3.8)
        gen_sp.plot(kind="bar", ax=ax, color=PAL[:2], edgecolor="white", rot=0)
        ax.set_title("Avg Spending: Usia × Gender", fontweight="bold")
        ax.set_xlabel("Kelompok Usia"); ax.set_ylabel("Avg Purchase (USD)")
        ax.legend(title="Gender", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("Avg spending <b>flat ~$58–$62</b> di semua kelompok usia untuk kedua gender — usia dan gender bukan prediktor spending. Selisih tidak melebihi $4.")

    section("Matriks Segmentasi: Spending Tier × Loyalty Tier")
    pivot_seg = dff.pivot_table(values="Customer ID", index="Loyalty Tier",
                                columns="Spending Tier", aggfunc="count", observed=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6,3.5))
        sns.heatmap(pivot_seg, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax,
                    cbar_kws={"label":"Jumlah Pelanggan"})
        ax.set_title("Matriks Segmentasi (Count)", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        pivot_seg.plot(kind="bar", stacked=True, figsize=(6,3.5),
                       color=PAL[:3], edgecolor="white")
        plt.title("Komposisi Segmen per Loyalty Tier", fontweight="bold")
        plt.xlabel("Loyalty Tier"); plt.xticks(rotation=0)
        plt.legend(title="Spending Tier", fontsize=9); plt.tight_layout()
        st.pyplot(plt.gcf()); plt.close()
    insight("Distribusi <b>sangat merata</b> di semua 9 sel (407–470) → loyalitas tidak memengaruhi tier spending. <b>New-High Spender (462)</b> ≈ Loyal-High Spender (428).")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POLA SPENDING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Pola Spending":
    st.title("💳 Pola Spending Pelanggan")

    section("Rata-rata Spending vs Variabel Kategorikal")
    cat_targets = [("Gender","Spending per Gender"),("Category","Spending per Kategori"),
                   ("Season","Spending per Musim"),("Subscription Status","Subscriber vs Non-Subscriber")]
    cols_biv = st.columns(4)
    for ax_col, (col, title) in zip(cols_biv, cat_targets):
        order = dff.groupby(col)["Purchase Amount"].mean().sort_values(ascending=False).index
        means = dff.groupby(col)["Purchase Amount"].mean().reindex(order)
        fig, ax = plt.subplots(figsize=(4, 3.8))
        bars = ax.bar(means.index, means.values, color=PAL[:len(means)],
                      edgecolor="white", width=0.55, zorder=3)
        ax.set_ylim(means.values.min()*0.94, means.values.max()*1.08)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
        ax.set_ylabel("Avg Purchase (USD)", fontsize=9)
        ax.tick_params(axis="x", rotation=25, labelsize=8.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:.0f}"))
        ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True); ax.spines[["top","right"]].set_visible(False)
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_y()+bar.get_height()/2,
                    f"${val:.1f}", ha="center", va="center",
                    fontsize=9, fontweight="bold", color="white")
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()

    insight("""<b>Gender</b>: Pria dan wanita identik $60.&nbsp;|&nbsp;
<b>Kategori</b>: <b>Footwear & Clothing tertinggi ($60)</b>, <b>Outerwear terendah ($57)</b>.&nbsp;|&nbsp;
<b>Musim</b>: Fall tertinggi ($62), Summer terendah ($58) — selisih $4.&nbsp;|&nbsp;
<b>Subscription</b>: No=$60 vs Yes=$59 — hampir identik.""")

    section("Heatmap Spending: Kategori × Musim")
    pivot_hm = dff.pivot_table(values="Purchase Amount", index="Category",
                                columns="Season", aggfunc="mean").round(1)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot_hm, annot=True, fmt=".1f", cmap=CMAP, linewidths=0.5, ax=ax,
                cbar_kws={"label":"Avg Purchase (USD)"})
    ax.set_title("Heatmap Rata-rata Spending: Kategori × Musim", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("<b>Footwear-Fall (63.7)</b> tertinggi. <b>Outerwear terendah</b> di hampir semua musim (54.6–59.8). Range hanya $9 → tidak ada pola musiman dramatis. Spring paling rendah.")

    section("Distribusi Spending: Musim & Metode Pembayaran")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_fig(6, 4)
        sns.violinplot(data=dff, x="Season", y="Purchase Amount", palette="Set2", inner="quartile", ax=ax)
        ax.set_title("Purchase Amount per Musim", fontweight="bold")
        ax.set_xlabel("Musim"); ax.set_ylabel("Purchase Amount (USD)")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        order_pay = dff.groupby("Payment Method")["Purchase Amount"].median().sort_values(ascending=False).index
        fig, ax = base_fig(6, 4)
        sns.boxplot(data=dff, x="Payment Method", y="Purchase Amount",
                    order=order_pay, palette="Set3", ax=ax)
        ax.set_title("Purchase Amount per Metode Pembayaran", fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("Violin plot: bentuk <b>hampir identik</b> di semua musim. Box plot: <b>semua metode memiliki IQR sangat serupa</b> (~$40–$80) dan median ~$60.")

    section("Correlation Matrix")
    df_corr = dff.copy()
    for col in ["Gender","Subscription Status","Promo Code Used"]:
        df_corr[col] = df_corr[col].map({"Yes":1,"No":0,"Male":1,"Female":0})
    num_c = ["Age","Purchase Amount","Review Rating","Previous Purchases",
             "Gender","Subscription Status","Promo Code Used"]
    corr = df_corr[num_c].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"shrink":0.8,"label":"Korelasi Pearson"})
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("""<b>Purchase Amount ≤ |0.03|</b> dengan semua variabel — tidak dapat diprediksi dari fitur yang ada.
Korelasi bermakna: <b>Subscription↔Promo (0.70)</b>, <b>Gender↔Promo (0.60)</b>, <b>Gender↔Subscription (0.42)</b>
→ Pria berlangganan = segmen paling aktif pakai promo, namun tidak meningkatkan nilai transaksi.""")

    section("Dampak Promo Code terhadap Spending")
    c1, c2, c3 = st.columns(3)
    with c1:
        promo_avg = dff.groupby("Promo Code Used")["Purchase Amount"].mean()
        fig, ax = base_fig(4, 3.5)
        bars = ax.bar(promo_avg.index, promo_avg.values, color=[PAL[3],PAL[0]], edgecolor="white", width=0.5)
        ax.set_title("Avg Spend: Promo Code", fontweight="bold"); ax.set_ylim(55, 65)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f"${bar.get_height():.1f}", ha="center", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        promo_cat = dff.groupby(["Category","Promo Code Used"])["Purchase Amount"].mean().unstack()
        fig, ax = base_fig(5, 3.5)
        promo_cat.plot(kind="bar", ax=ax, color=PAL[:2], edgecolor="white", rot=0)
        ax.set_title("Avg Spend per Kategori & Promo", fontweight="bold")
        ax.legend(title="Promo Code", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c3:
        df_fp = dff[dff["Frequency of Purchases"].isin(URUTAN_FREQ)]
        freq_promo = (df_fp.groupby(["Frequency of Purchases","Promo Code Used"])
                     .size().unstack().reindex(URUTAN_FREQ))
        fig, ax = base_fig(5, 3.5)
        freq_promo.plot(kind="bar", ax=ax, color=PAL[:2], edgecolor="white", rot=30)
        ax.set_title("Penggunaan Promo per Frekuensi", fontweight="bold")
        ax.legend(title="Promo Code", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("No=$60.1 vs Yes=$59.3 → <b>promo tidak meningkatkan nilai transaksi</b> (selisih $0.8). Di semua kelompok frekuensi, transaksi tanpa promo selalu lebih banyak (~60:40).")

    section("Spending per Frekuensi Pembelian")
    df_freq2 = dff[dff["Frequency of Purchases"].isin(URUTAN_FREQ)].copy()
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_fig(6, 4)
        sns.boxplot(data=df_freq2, x="Frequency of Purchases", y="Purchase Amount",
                    order=URUTAN_FREQ, palette="Set3", ax=ax)
        ax.set_title("Distribusi Spending per Frekuensi", fontweight="bold")
        ax.set_xlabel("Frekuensi Pembelian"); ax.set_ylabel("Purchase Amount (USD)")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        rev_freq = df_freq2.groupby("Frequency of Purchases")["Purchase Amount"].sum().reindex(URUTAN_FREQ)
        fig, ax = base_fig(6, 4)
        bars = ax.bar(rev_freq.index, rev_freq.values, color=PAL[:5], edgecolor="white")
        ax.set_title("Total Revenue per Frekuensi Pembelian", fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                    f"${bar.get_height():,.0f}", ha="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("Median spending identik ~$60 di semua frekuensi. <b>Quarterly=$68.859 revenue terbesar</b>, Fortnightly=$65.207. <b>Weekly terkecil ($31.786)</b> — volume transaksi paling sedikit.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUK & KATEGORI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Produk & Kategori":
    st.title("📦 Analisis Produk & Kategori")

    section("Distribusi Kategori & Top Item Terlaris")
    c1, c2 = st.columns(2)
    with c1:
        cat_c = dff["Category"].value_counts()
        fig, ax = base_fig(5, 4)
        ax.pie(cat_c, labels=cat_c.index, autopct="%1.1f%%",
               colors=PAL[:len(cat_c)], startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":1.5})
        ax.set_title("Komposisi Penjualan per Kategori", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        top_items = dff["Item Purchased"].value_counts().head(10)
        fig, ax = base_fig(5, 4)
        ax.barh(top_items.index[::-1], top_items.values[::-1], color=PAL[2], edgecolor="white")
        ax.set_title("Top 10 Item Terlaris", fontweight="bold"); ax.set_xlabel("Jumlah Transaksi")
        for i, v in enumerate(top_items.values[::-1]):
            ax.text(v+1, i, str(v), va="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("<b>Clothing</b> mendominasi volume (44.5%). Dominasi di revenue murni karena volume transaksi, bukan harga per item yang lebih tinggi.")

    section("Rata-rata Spending per Item — Top 5 per Kategori")
    top_sp = (dff.groupby(["Category","Item Purchased"])["Purchase Amount"]
               .mean().reset_index()
               .sort_values(["Category","Purchase Amount"], ascending=[True,False]))
    cols_items = st.columns(4)
    for ax_col, cat in zip(cols_items, sorted(dff["Category"].unique())):
        sub = top_sp[top_sp["Category"]==cat].head(5)
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        ax.barh(sub["Item Purchased"][::-1], sub["Purchase Amount"][::-1],
                color=PAL[:5], edgecolor="white")
        ax.set_title(cat, fontweight="bold", fontsize=10); ax.set_xlabel("Avg USD")
        for bar in ax.patches:
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    f"${bar.get_width():.0f}", va="center", fontsize=7)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()
    insight("Range harga <b>sangat sempit ($57–$63)</b>. Clothing: T-shirt tertinggi ($63). Footwear: Boots tertinggi ($63). <b>Outerwear: Coat=$58, Jacket=$57 — terendah di semua kategori.</b>")

    section("Top 5 Item Paling Kurang Laku per Musim")
    seasons = sorted(dff["Season"].unique())
    season_colors = {"Spring":PAL[1],"Summer":PAL[2],"Fall":PAL[0],"Winter":PAL[3]}
    cols_sl = st.columns(len(seasons))
    for ax_col, season in zip(cols_sl, seasons):
        df_s    = dff[dff["Season"]==season]
        bottom5 = df_s["Item Purchased"].value_counts().nsmallest(5).sort_values(ascending=True)
        color   = season_colors.get(season, PAL[0])
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        bars = ax.barh(bottom5.index, bottom5.values, color=color, alpha=0.85, edgecolor="white")
        ax.set_title(f"Musim: {season}", fontweight="bold", fontsize=10); ax.set_xlabel("Jumlah Transaksi")
        x_max = bottom5.max()*1.35; ax.set_xlim(0, x_max)
        for bar, val in zip(bars, bottom5.values):
            ax.text(bar.get_width()+x_max*0.02, bar.get_y()+bar.get_height()/2,
                    str(int(val)), va="center", fontsize=9, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()
    insight("Item kurang laku yang konsisten muncul di banyak musim perlu evaluasi strategi produk. Item yang hanya kurang laku di musim tertentu dapat ditangani dengan promosi musiman atau bundle.")

    section("Distribusi Warna & Ukuran")
    c1, c2 = st.columns(2)
    with c1:
        top_colors = dff["Color"].value_counts().head(10)
        fig, ax = base_fig(6,3.8)
        ax.bar(top_colors.index, top_colors.values, color=PAL[3], edgecolor="white")
        ax.set_title("Top 10 Warna Favorit", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        size_order = [s for s in ["XS","S","M","L","XL"] if s in dff["Size"].values]
        size_c = dff["Size"].value_counts().reindex(size_order).dropna()
        fig, ax = base_fig(6,3.8)
        bars = ax.bar(size_c.index, size_c.values, color=PAL[4], edgecolor="white")
        ax.set_title("Distribusi Ukuran", fontweight="bold")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(int(bar.get_height())), ha="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("Ukuran <b>M sangat mendominasi (1.755)</b>, diikuti L (1.053). XS (429) dan XL (663) paling sedikit diminati.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FREE SHIPPING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🚚 Free Shipping":
    st.title("🚚 Analisis Free Shipping")
    df_free = dff[dff["Shipping Type"]=="Free Shipping"].copy()

    c1,c2,c3 = st.columns(3)
    c1.markdown(kpi(f"{len(df_free):,}","Transaksi Free Shipping","#6366f1",
                    f"{len(df_free)/len(dff)*100:.1f}% dari total"), unsafe_allow_html=True)
    c2.markdown(kpi(f"${df_free['Purchase Amount'].mean():.0f}","Avg Spending (FS)","#10b981",
                    "vs all avg $60"), unsafe_allow_html=True)
    top_season = df_free["Season"].value_counts().index[0]
    c3.markdown(kpi(top_season,"Musim Terbanyak","#f59e0b",
                    f"{df_free['Season'].value_counts().iloc[0]} transaksi"), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    section("Free Shipping Rate per Segmen Pelanggan")
    c1, c2, c3 = st.columns(3)
    with c1:
        gen_fs  = df_free["Gender"].value_counts()
        gen_all = dff["Gender"].value_counts()
        gen_pct = (gen_fs/gen_all*100).round(1)
        fig, ax = base_fig(4, 3.5)
        bars = ax.bar(gen_pct.index, gen_pct.values, color=PAL[:2], edgecolor="white", width=0.5)
        ax.set_title("Rate per Gender", fontweight="bold"); ax.set_ylim(0, gen_pct.max()*1.25)
        ax.set_ylabel("% dari total grup tsb")
        ax.spines[["top","right"]].set_visible(False)
        for bar,val in zip(bars,gen_pct.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{val:.1f}%", ha="center", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        sub_fs  = df_free["Subscription Status"].value_counts()
        sub_all = dff["Subscription Status"].value_counts()
        sub_pct = (sub_fs/sub_all*100).round(1)
        fig, ax = base_fig(4, 3.5)
        bars = ax.bar(sub_pct.index, sub_pct.values, color=[PAL[2],PAL[5]], edgecolor="white", width=0.5)
        ax.set_title("Rate per Subscription", fontweight="bold"); ax.set_ylim(0, sub_pct.max()*1.25)
        ax.set_ylabel("% dari total grup tsb")
        ax.spines[["top","right"]].set_visible(False)
        for bar,val in zip(bars,sub_pct.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{val:.1f}%", ha="center", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c3:
        freq_fs  = df_free[df_free["Frequency of Purchases"].isin(URUTAN_FREQ)]["Frequency of Purchases"].value_counts().reindex(URUTAN_FREQ)
        freq_all = dff[dff["Frequency of Purchases"].isin(URUTAN_FREQ)]["Frequency of Purchases"].value_counts().reindex(URUTAN_FREQ)
        freq_pct = (freq_fs/freq_all*100).round(1)
        fig, ax = base_fig(5, 3.5)
        bars = ax.bar(freq_pct.index, freq_pct.values, color=PAL[:5], edgecolor="white")
        ax.set_title("Rate per Frekuensi Belanja", fontweight="bold")
        ax.set_ylabel("% dari total grup tsb"); ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, freq_pct.max()*1.25)
        ax.spines[["top","right"]].set_visible(False)
        for bar,val in zip(bars,freq_pct.dropna().values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    section("Volume & Rate Free Shipping per Musim")
    c1, c2 = st.columns(2)
    with c1:
        season_fs  = df_free["Season"].value_counts().sort_values(ascending=False)
        fig, ax = base_fig(6, 4)
        bars = ax.bar(season_fs.index, season_fs.values, color=PAL[:4], edgecolor="white")
        ax.set_title("Volume Free Shipping per Musim", fontweight="bold"); ax.set_ylabel("Jumlah Transaksi")
        ax.spines[["top","right"]].set_visible(False)
        for bar,val in zip(bars,season_fs.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(val),
                    ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        season_all = dff["Season"].value_counts()
        season_pct = (season_fs/season_all*100).round(1)
        fig, ax = base_fig(6, 4)
        bars = ax.bar(season_pct.index, season_pct.values, color=PAL[:4], edgecolor="white")
        ax.set_title("Free Shipping Rate per Musim (%)", fontweight="bold")
        ax.set_ylabel("% transaksi menggunakan Free Shipping"); ax.set_ylim(0, season_pct.max()*1.25)
        ax.spines[["top","right"]].set_visible(False)
        for bar,val in zip(bars,season_pct.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("""Rate Free Shipping antara semua segmen (gender, subscription, frekuensi) sangat serupa (~17%) → <b>tidak ada tipe pelanggan yang secara khusus mendominasi pilihan Free Shipping</b>.
Konsisten dengan pola distribusi uniform dataset ini. Free Shipping dapat dijadikan <b>insentif threshold</b> (misal: gratis ongkir untuk pembelian >$75) untuk mendorong peningkatan Average Order Value.""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵 Clustering":
    st.title("🔵 K-Means Clustering — Segmentasi Pelanggan")
    st.caption("Tiga pendekatan clustering untuk keperluan CRM & Retensi")

    with st.spinner("Menjalankan clustering... (di-cache setelah pertama kali)"):
        clust_results = run_clustering(df)

    # Tabs per opsi
    tab_a, tab_b, tab_c, tab_comp = st.tabs([
        "Opsi A — RFM-Proxy", "Opsi B — Behavioral",
        "Opsi C — Demo-Behavioral", "Komparasi"
    ])

    for tab, opt_key, opt_title, opt_focus in [
        (tab_a, "A", "RFM-Proxy",       "CRM & Loyalitas"),
        (tab_b, "B", "Behavioral",       "Strategi Promosi"),
        (tab_c, "C", "Demo-Behavioral",  "Targeting & Channel"),
    ]:
        with tab:
            res = clust_results[opt_key]
            k   = res["k"]
            labels_col = f"Cluster_{opt_key}"
            df[labels_col] = res["labels"]

            st.markdown(f"**Fitur:** {', '.join(res['feats'])} &nbsp;·&nbsp; "
                        f"**Fokus:** {opt_focus} &nbsp;·&nbsp; "
                        f"**k={k}** &nbsp;·&nbsp; "
                        f"**Silhouette={res['sil']:.4f}**")

            section("Elbow & Silhouette Score")
            c1, c2 = st.columns(2)
            ks = list(range(2, 7))
            with c1:
                fig, ax = base_fig(5, 3.5)
                ax.plot(ks, res["inertias"], "o-", color=PAL[0], linewidth=2, markersize=7)
                ax.axvline(k, color="red", linestyle="--", linewidth=1.5, label=f"k={k}")
                ax.set_title("Elbow Curve", fontweight="bold")
                ax.set_xlabel("Jumlah Cluster (k)"); ax.set_ylabel("Inertia")
                ax.legend(fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with c2:
                fig, ax = base_fig(5, 3.5)
                ax.plot(ks, res["sil_scores"], "s-", color=PAL[1], linewidth=2, markersize=7)
                ax.axvline(k, color="red", linestyle="--", linewidth=1.5, label=f"Best k={k} ({res['sil']:.3f})")
                ax.set_title("Silhouette Score", fontweight="bold")
                ax.set_xlabel("Jumlah Cluster (k)"); ax.set_ylabel("Silhouette Score")
                ax.legend(fontsize=8)
                plt.tight_layout(); st.pyplot(fig); plt.close()

            section("Visualisasi Cluster & Profil")
            c1, c2 = st.columns(2)
            with c1:
                colors_plot = [PAL_HEX[i] for i in res["labels"]]
                fig, ax = base_fig(6, 4.5)
                ax.scatter(res["X_pca"][:,0], res["X_pca"][:,1],
                           c=colors_plot, alpha=0.4, s=15)
                patches = [mpatches.Patch(color=PAL_HEX[i], label=f"Cluster {i}") for i in range(k)]
                ax.legend(handles=patches, fontsize=8)
                ax.set_title("Cluster Plot (PCA 2D)", fontweight="bold")
                ax.set_xlabel(f"PC1 ({res['var_exp'][0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({res['var_exp'][1]*100:.1f}%)")
                plt.tight_layout(); st.pyplot(fig); plt.close()
            with c2:
                profile = df.groupby(labels_col)[res["feats"]].mean()
                profile_norm = (profile - profile.min()) / (profile.max()-profile.min()+1e-9)
                fig, ax = plt.subplots(figsize=(6, 4.5))
                sns.heatmap(profile_norm.T, annot=profile.T, fmt=".1f", cmap="YlOrRd",
                            linewidths=0.5, ax=ax, cbar_kws={"label":"Normalized (0-1)"})
                ax.set_title("Profil Rata-rata per Cluster", fontweight="bold")
                ax.set_xlabel("Cluster"); ax.set_ylabel("")
                plt.tight_layout(); st.pyplot(fig); plt.close()

            section("Profil & Ukuran Cluster")
            cluster_sizes = df[labels_col].value_counts().sort_index()

            # Cluster cards
            card_cols = st.columns(k)
            cluster_names = {
                "A": {
                    0: ("Segmen 0","Lihat profil heatmap di atas untuk karakteristik spesifik cluster ini."),
                    1: ("Segmen 1","Lihat profil heatmap di atas untuk karakteristik spesifik cluster ini."),
                    2: ("Segmen 2","Lihat profil heatmap di atas untuk karakteristik spesifik cluster ini."),
                    3: ("Segmen 3","Lihat profil heatmap di atas untuk karakteristik spesifik cluster ini."),
                },
            }
            for i, col in enumerate(card_cols):
                size = cluster_sizes.get(i, 0)
                pct  = size / len(df) * 100
                col.markdown(
                    f'<div class="cluster-card" style="border-top-color:{PAL_HEX[i]}">'
                    f'<div class="cluster-label">Cluster {i}</div>'
                    f'<div class="cluster-name" style="color:{PAL_HEX[i]}">Segmen {i}</div>'
                    f'<div class="cluster-desc">Lihat profil heatmap untuk karakteristik lengkap.</div>'
                    f'<div style="margin-top:.5rem;font-size:.8rem;color:#94a3b8">'
                    f'{size:,} pelanggan · {pct:.1f}%</div></div>',
                    unsafe_allow_html=True
                )

            # Summary table
            summary = df.groupby(labels_col)[res["feats"]].mean().round(2)
            summary["Jumlah"] = cluster_sizes
            summary["Pct (%)"] = (cluster_sizes / len(df) * 100).round(1)
            st.dataframe(summary, use_container_width=True)

            # Insight per opsi
            if opt_key == "A":
                insight("""Cluster dengan <b>Previous Purchases tinggi + Amount tinggi</b> = Loyal High-Value → pertahankan dengan program eksklusif.<br>
Cluster dengan <b>Previous Purchases rendah + Freq rendah</b> = At-Risk / New → re-aktivasi & onboarding campaign.<br>
Cluster dengan <b>Amount rendah + Freq tinggi</b> = Frequent Low-Spender → upsell & cross-sell campaign.""")
            elif opt_key == "B":
                insight("""Cluster dengan <b>Amount tinggi + Rating tinggi + Promo rendah</b> = Premium Satisfied → fokus pada experience & eksklusivitas.<br>
Cluster dengan <b>Amount rendah + Promo tinggi</b> = Price-Sensitive → threshold-based promo untuk dorong upsell.<br>
Cluster dengan <b>Freq tinggi + Rating rendah</b> = Frequent Dissatisfied → prioritaskan service recovery.""")
            elif opt_key == "C":
                insight("""Cluster dengan <b>Usia muda + Freq tinggi + Prev Purchases rendah</b> = Young Active → digital campaign, social media, loyalty onboarding.<br>
Cluster dengan <b>Usia tua + Prev Purchases tinggi + Amount tinggi</b> = Mature Loyal → personal touch, program VIP, komunikasi email.<br>
Cluster dengan <b>Amount rendah + Freq rendah</b> = Dormant → re-aktivasi campaign dengan insentif kuat.""")

    # Tab Komparasi
    with tab_comp:
        section("Perbandingan Ketiga Opsi Clustering")

        sil_a = silhouette_score(clust_results["A"]["X_sc"], clust_results["A"]["labels"])
        sil_b = silhouette_score(clust_results["B"]["X_sc"], clust_results["B"]["labels"])
        sil_c = silhouette_score(clust_results["C"]["X_sc"], clust_results["C"]["labels"])

        m1, m2, m3 = st.columns(3)
        m1.markdown(kpi(f"{sil_a:.4f}", f"Opsi A — k={clust_results['A']['k']}", "#66c2a5",
                        "RFM-Proxy · CRM"), unsafe_allow_html=True)
        m2.markdown(kpi(f"{sil_b:.4f}", f"Opsi B — k={clust_results['B']['k']}", "#fc8d62",
                        "Behavioral · Promosi"), unsafe_allow_html=True)
        m3.markdown(kpi(f"{sil_c:.4f}", f"Opsi C — k={clust_results['C']['k']}", "#8da0cb",
                        "Demo-Behavioral · Targeting"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section("Distribusi Ukuran Cluster per Opsi")
        c1, c2, c3 = st.columns(3)
        for col, opt_key, color in [(c1,"A","#66c2a5"),(c2,"B","#fc8d62"),(c3,"C","#8da0cb")]:
            lbl_col = f"Cluster_{opt_key}"
            sizes   = df[lbl_col].value_counts().sort_index()
            fig, ax = base_fig(4, 3.5)
            bars = ax.bar([f"C{i}" for i in sizes.index], sizes.values,
                          color=[PAL_HEX[i] for i in sizes.index], edgecolor="white", width=0.6)
            ax.set_title(f"Opsi {opt_key} — {clust_results[opt_key]['label']}", fontweight="bold")
            ax.set_ylabel("Jumlah Pelanggan")
            ax.spines[["top","right"]].set_visible(False)
            for bar, val in zip(bars, sizes.values):
                pct = val/len(df)*100
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                        f"{val}\n({pct:.1f}%)", ha="center", fontsize=8, fontweight="bold")
            plt.tight_layout(); col.pyplot(fig); plt.close()

        section("Rekomendasi Penggunaan")
        r1, r2, r3 = st.columns(3)
        r1.markdown("""**🎯 Opsi A — RFM-Proxy**
Gunakan untuk **CRM & Loyalty Program**.
Menentukan prioritas retensi berdasarkan riwayat & nilai transaksi. Paling dekat dengan framework RFM klasik.""")
        r2.markdown("""**🎨 Opsi B — Behavioral**
Gunakan untuk **Strategi Promosi & Pricing**.
Mendesain campaign promo tepat sasaran berdasarkan pola belanja dan price sensitivity.""")
        r3.markdown("""**📡 Opsi C — Demo-Behavioral**
Gunakan untuk **Targeting & Personalisasi Channel**.
Menentukan channel komunikasi sesuai (digital vs email vs in-store) berdasarkan profil usia & perilaku.""")

        insight("""Silhouette score ketiga opsi relatif rendah karena <b>distribusi dataset yang uniform</b> — tidak ada natural cluster yang tajam. Ini bukan kegagalan algoritma, melainkan cerminan karakteristik intrinsik data.
Kualitas clustering akan meningkat signifikan jika dataset diperkaya dengan fitur seperti <b>income, CLV, dan timestamp transaksi</b>.""")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KESIMPULAN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Kesimpulan & Saran":
    st.title("📝 Kesimpulan & Saran Strategis")

    section("Ringkasan Temuan EDA")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**📊 Kualitas & Profil Dataset**
- 3.900 transaksi, 17 fitur, **0 missing value, 0 duplikat**
- Semua variabel numerik terdistribusi **uniform** — bukan normal

**👥 Profil Pelanggan**
- Pria ~68%, usia merata 18–70 tahun, 27% berlangganan
- Shipping Type & Payment Method merata di semua pilihan

**💳 Pola Spending**
- Avg/Median **$60/transaksi**, range uniform $20–$100
- Tidak ada variabel yang memengaruhi spending secara signifikan
- **Footwear & Clothing tertinggi** ($60–$63); **Outerwear terendah** ($57–$58)
- Fall = spending tertinggi, Summer terendah — selisih hanya $4
        """)
    with c2:
        st.markdown("""
**📅 Frekuensi & Revenue**
- Frekuensi terbanyak: **Quarterly (1.147)**, Weekly terkecil (539)
- Revenue terbesar: **Quarterly=$68.859**, Weekly terkecil ($31.786)

**🔗 Korelasi Penting**
- Purchase Amount: korelasi ≤ |0.03| → **spending tidak dapat diprediksi**
- Subscription↔Promo (0.70), Gender↔Promo (0.60), Gender↔Subscription (0.42)

**🎫 Efektivitas Promo & Free Shipping**
- Promo: No=$60.1 vs Yes=$59.3 → tidak meningkatkan nilai transaksi
- Free Shipping: dipilih merata tanpa segmen dominan

**🔵 Clustering**
- Silhouette rendah karena distribusi uniform — wajar untuk data ini
- Tiga opsi tersedia untuk CRM, Promosi, dan Targeting
        """)

    section("Saran Strategis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🔍 A. Pengayaan Data**
- Income level / price sensitivity
- Browsing session & click-through rate
- Cart abandonment history
- Customer Lifetime Value historis
- Timestamp transaksi untuk tren waktu
        """)
    with col2:
        st.markdown("""
**🎯 B. Insentif & Segmen**
- Redesain promo: **threshold-based** (belanja >$80 dapat diskon)
- Free Shipping sebagai insentif minimum pembelian
- Fokus retensi: **Quarterly & Fortnightly** (revenue terbesar)
- Re-aktivasi: segmen **Weekly** — volume rendah, potensi naik
        """)
    with col3:
        st.markdown("""
**📦 C. Produk & Musim**
- **Footwear & Clothing** = prioritas upsell & cross-sell
- **Outerwear** perlu evaluasi strategi harga
- Manfaatkan momentum **Fall** — spending tertinggi
- Bundle item kurang laku dengan item populer musiman
        """)

    section("Keterbatasan Analisis")
    st.markdown("""
- Distribusi uniform → **model prediktif sangat terbatas** tanpa fitur tambahan
- Tidak ada timestamp transaksi → analisis tren waktu tidak dapat dilakukan
- Clustering silhouette rendah → butuh fitur yang lebih diskriminatif
- Korelasi Gender/Subscription/Promo perlu divalidasi lebih lanjut
    """)
