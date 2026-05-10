import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

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
    df["Age Group"] = pd.cut(df["Age"], bins=[17, 25, 35, 45, 55, 70],
                              labels=["18-25", "26-35", "36-45", "46-55", "56+"])
    df["Spending Tier"] = pd.qcut(df["Purchase Amount"], q=3,
                                   labels=["Low", "Mid", "High"])
    df["Loyalty Tier"] = pd.qcut(df["Previous Purchases"], q=3,
                                  labels=["New", "Regular", "Loyal"])
    return df

df = load_data()
PAL  = sns.color_palette("Set2", 8)
CMAP = "YlOrRd"
URUTAN_FREQ = ["Weekly", "Fortnightly", "Monthly", "Quarterly", "Annually"]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def kpi(val, label, color, delta=""):
    return f"""<div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val" style="color:{color}">{val}</div>
        <div class="kpi-delta">{delta}</div>
    </div>"""

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight-box">📌 {text}</div>', unsafe_allow_html=True)

def base_fig(w=6, h=3.8):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax

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
        "📝 Kesimpulan & Saran",
    ])
    st.markdown("---")
    st.markdown("**Filter Global**")
    gender_filter  = st.multiselect("Gender", df["Gender"].unique().tolist(),
                                     default=df["Gender"].unique().tolist())
    season_filter  = st.multiselect("Musim", df["Season"].unique().tolist(),
                                     default=df["Season"].unique().tolist())
    age_min, age_max = st.slider("Rentang Usia",
                                  int(df["Age"].min()), int(df["Age"].max()), (18, 70))
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} records | 17 fitur")

mask = (df["Gender"].isin(gender_filter) &
        df["Season"].isin(season_filter) &
        df["Age"].between(age_min, age_max))
dff = df[mask]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Spending Patterns")
    st.caption("Exploratory Data Analysis — Shopping Behavior Dataset")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown(kpi(f"{len(dff):,}", "Total Transaksi", "#6366f1",
                    f"{len(dff)/len(df)*100:.0f}% dari total"), unsafe_allow_html=True)
    c2.markdown(kpi(f"${dff['Purchase Amount'].mean():.0f}", "Avg Spending",
                    "#10b981", f"Median ${dff['Purchase Amount'].median():.0f}"),
                unsafe_allow_html=True)
    c3.markdown(kpi(f"${dff['Purchase Amount'].sum():,.0f}", "Total Revenue",
                    "#f59e0b", "seluruh transaksi"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{dff['Review Rating'].mean():.2f} ★", "Avg Rating",
                    "#ef4444", "dari 5.0"), unsafe_allow_html=True)
    c5.markdown(kpi(f"{(dff['Subscription Status']=='Yes').mean()*100:.0f}%",
                    "Subscriber Rate", "#8b5cf6",
                    f"{(dff['Subscription Status']=='Yes').sum():,} pelanggan"),
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    section("Distribusi Purchase Amount & Revenue per Kategori")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = base_fig()
        ax.hist(dff["Purchase Amount"], bins=30, color=PAL[0], edgecolor="white", linewidth=0.6)
        ax.axvline(dff["Purchase Amount"].mean(),  color="red",  linestyle="--",
                   linewidth=1.8, label=f"Mean=${dff['Purchase Amount'].mean():.0f}")
        ax.axvline(dff["Purchase Amount"].median(), color="navy", linestyle=":",
                   linewidth=1.8, label=f"Median=${dff['Purchase Amount'].median():.0f}")
        ax.set_title("Distribusi Purchase Amount", fontweight="bold")
        ax.set_xlabel("USD"); ax.set_ylabel("Frekuensi")
        ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight("Distribusi <b>uniform sempurna $20–$100</b>. Mean=59.8 ≈ Median=60.0 → distribusi simetris tanpa skewness. Tidak ada harga pembelian yang lebih populer dari harga lainnya.")

    with c2:
        cat_rev = dff.groupby("Category")["Purchase Amount"].sum().sort_values()
        fig, ax = base_fig()
        bars = ax.barh(cat_rev.index, cat_rev.values, color=PAL[:len(cat_rev)], edgecolor="white")
        ax.set_title("Total Revenue per Kategori", fontweight="bold")
        ax.set_xlabel("Total Revenue (USD)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for bar in bars:
            ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight("<b>Clothing</b> mendominasi total revenue karena volume transaksi terbesar (1.737 item). Bukan karena harga tertinggi — semua kategori rata-rata $57–$63 per transaksi, dengan <b>Outerwear justru terendah ($57)</b>.")

    section("Heatmap Rata-rata Spending: Kategori × Musim")
    pivot = dff.pivot_table(values="Purchase Amount", index="Category",
                             columns="Season", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=CMAP, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Avg Purchase (USD)"})
    ax.set_title("Heatmap Rata-rata Spending per Kategori & Musim", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("<b>Footwear-Fall tertinggi (63.7)</b>. <b>Outerwear adalah baris terendah</b> di hampir semua musim (54.6–59.8). Range keseluruhan hanya $9 → tidak ada pola musiman yang dramatis. Spring adalah musim dengan spending paling rendah di hampir semua kategori.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMOGRAFI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Demografi":
    st.title("👤 Analisis Demografi Pelanggan")

    section("4.1 — Distribusi Variabel Numerik")
    cols_num = st.columns(4)
    num_info = [
        ("Age",                "Usia Pelanggan (tahun)", PAL[0]),
        ("Purchase Amount",    "Jumlah Pembelian (USD)", PAL[1]),
        ("Review Rating",      "Rating Ulasan (1–5)",    PAL[2]),
        ("Previous Purchases", "Riwayat Pembelian",      PAL[3]),
    ]
    for col, (var, label, color) in zip(cols_num, num_info):
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.hist(dff[var], bins=25, color=color, edgecolor="white", linewidth=0.5, alpha=0.9)
        ax.axvline(dff[var].mean(),   color="red",  linestyle="--", linewidth=1.6,
                   label=f"Mean={dff[var].mean():.1f}")
        ax.axvline(dff[var].median(), color="navy", linestyle=":",  linewidth=1.6,
                   label=f"Median={dff[var].median():.1f}")
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylabel("Frekuensi", fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); col.pyplot(fig); plt.close()

    insight("""<b>Age</b>: Uniform merata 18–70 tahun, Mean=44.1 ≈ Median=44.0 — tidak ada kelompok usia dominan.&nbsp;|&nbsp;
<b>Purchase Amount</b>: Uniform $20–$100, Mean=59.8 ≈ Median=60.0 — simetris sempurna, tanpa skewness.&nbsp;|&nbsp;
<b>Review Rating</b>: Distribusi <b>flat/uniform</b> di rentang 2.5–5.0, Mean=Median=3.7 — tidak ada skewness, pelanggan memberi rating merata di semua level (bukan mayoritas rating tinggi seperti dugaan awal).&nbsp;|&nbsp;
<b>Previous Purchases</b>: <b>Hampir uniform</b> 0–50, Mean=25.4 ≈ Median=25.0 — tidak ada dominasi pelanggan baru maupun lama.""")

    section("4.2 — Distribusi Variabel Kategorikal")
    cat_cols = ["Gender", "Category", "Season", "Subscription Status",
                "Shipping Type", "Payment Method", "Size", "Frequency of Purchases"]
    row1 = st.columns(4)
    row2 = st.columns(4)
    for ax_col, col in zip(row1 + row2, cat_cols):
        if col == "Frequency of Purchases":
            vc = dff[col].value_counts().reindex(URUTAN_FREQ).dropna()
        else:
            vc = dff[col].value_counts()
        fig, ax = plt.subplots(figsize=(4, 3))
        bars = ax.bar(vc.index, vc.values, color=PAL[:len(vc)], edgecolor="white")
        ax.set_title(col, fontsize=8, fontweight="bold")
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.spines[["top","right"]].set_visible(False)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    str(int(bar.get_height())), ha="center", fontsize=6, color="#333")
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()

    insight("""<b>Gender</b>: Pria=2.652 (~68%) vs Wanita=1.248 (~32%).&nbsp;|&nbsp;
<b>Category</b>: Clothing terbanyak (1.737), diikuti Accessories (1.240).&nbsp;|&nbsp;
<b>Season</b>: Sangat merata (955–999) — tidak ada musim dominan.&nbsp;|&nbsp;
<b>Subscription</b>: No=2.847 (73%) vs Yes=1.053 (27%).&nbsp;|&nbsp;
<b>Shipping</b>: <b>Sangat merata semua 6 jenis (627–675)</b> — tidak ada tipe dominan.&nbsp;|&nbsp;
<b>Payment</b>: <b>Sangat merata semua metode (612–677)</b> — tidak ada preferensi dominan.&nbsp;|&nbsp;
<b>Size</b>: M mendominasi (1.755), L urutan kedua (1.053).&nbsp;|&nbsp;
<b>Frequency</b>: <b>Quarterly terbanyak (1.147)</b>, diikuti Fortnightly (1.089). <b>Weekly terendah (539)</b> — mayoritas pelanggan belanja bulanan atau per 3 bulan, bukan mingguan.""")

    section("Distribusi Usia, Gender & Kelompok Usia")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig, ax = base_fig(5, 3.8)
        ax.hist(dff["Age"], bins=25, color=PAL[0], edgecolor="white")
        ax.axvline(dff["Age"].mean(), color="red", linestyle="--",
                   label=f"Mean={dff['Age'].mean():.1f}")
        ax.set_title("Distribusi Usia", fontweight="bold")
        ax.set_xlabel("Usia"); ax.set_ylabel("Count"); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        gender_c = dff["Gender"].value_counts()
        fig, ax = base_fig(5, 3.8)
        ax.pie(gender_c, labels=gender_c.index, autopct="%1.1f%%",
               colors=PAL[:2], startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Distribusi Gender", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c3:
        age_g = dff["Age Group"].value_counts().sort_index()
        fig, ax = base_fig(5, 3.8)
        bars = ax.bar(age_g.index, age_g.values, color=PAL[:len(age_g)], edgecolor="white")
        ax.set_title("Pelanggan per Kelompok Usia", fontweight="bold"); ax.set_ylabel("Count")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(int(bar.get_height())), ha="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Distribusi usia <b>uniform merata</b> — tidak ada peak di kelompok tertentu. Semua kelompok usia hadir dalam proporsi seimbang (~700–800 per grup). Pria ~68% mendominasi secara konsisten.")

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

    insight("Avg spending <b>flat ~$58–$62</b> di semua kelompok usia untuk kedua gender — usia dan gender sama-sama bukan prediktor spending. Selisih antar kelompok usia tidak melebihi $4 → tidak bermakna secara praktis.")

    section("Matriks Segmentasi: Spending Tier × Loyalty Tier")
    pivot_seg = dff.pivot_table(values="Customer ID", index="Loyalty Tier",
                                columns="Spending Tier", aggfunc="count", observed=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.heatmap(pivot_seg, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax,
                    cbar_kws={"label": "Jumlah Pelanggan"})
        ax.set_title("Matriks Segmentasi (Count)", fontweight="bold")
        ax.set_xlabel("Spending Tier"); ax.set_ylabel("Loyalty Tier")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        pivot_seg.plot(kind="bar", stacked=True, figsize=(6, 3.5),
                       color=PAL[:3], edgecolor="white")
        plt.title("Komposisi Segmen per Loyalty Tier", fontweight="bold")
        plt.xlabel("Loyalty Tier"); plt.ylabel("Jumlah Pelanggan")
        plt.xticks(rotation=0)
        plt.legend(title="Spending Tier", fontsize=9); plt.tight_layout()
        st.pyplot(plt.gcf()); plt.close()

    insight("Distribusi <b>sangat merata</b> di semua 9 sel (407–470) → loyalitas tidak memengaruhi tier spending. Insight penting: <b>New-High Spender (462)</b> hampir sama besar dengan Loyal-High Spender (428) — pelanggan baru tidak lebih sedikit berbelanja besar dibanding pelanggan lama.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POLA SPENDING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Pola Spending":
    st.title("💳 Pola Spending Pelanggan")

    section("5.1 — Rata-rata Spending vs Variabel Kategorikal")
    cat_targets = [
        ("Gender",              "Spending per Gender"),
        ("Category",            "Spending per Kategori"),
        ("Season",              "Spending per Musim"),
        ("Subscription Status", "Subscriber vs Non-Subscriber"),
    ]
    cols_biv = st.columns(4)
    for ax_col, (col, title) in zip(cols_biv, cat_targets):
        order = dff.groupby(col)["Purchase Amount"].mean().sort_values(ascending=False).index
        grp   = dff.groupby(col)["Purchase Amount"]
        means = grp.mean().reindex(order)
        sems  = grp.sem().reindex(order)
        fig, ax = plt.subplots(figsize=(4, 3.5))
        bars = ax.bar(means.index, means.values, color=PAL[:len(means)], edgecolor="white")
        ax.errorbar(range(len(means)), means.values, yerr=sems.values * 1.96,
                    fmt="none", color="#444", capsize=4, linewidth=1.5)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_ylabel("Avg Purchase (USD)")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.set_ylim(0, means.max() * 1.3)
        for bar, val in zip(bars, means.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.4,
                    f"${val:.0f}", ha="center", fontsize=8, fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()

    insight("""<b>Gender</b>: Pria dan wanita identik $60 — gender bukan penentu spending.&nbsp;|&nbsp;
<b>Kategori</b>: <b>Footwear & Clothing tertinggi ($60)</b>, <b>Outerwear terendah ($57)</b> — berlawanan dari dugaan awal, Outerwear bukan kategori premium.&nbsp;|&nbsp;
<b>Musim</b>: Fall tertinggi ($62), Summer terendah ($58) — selisih kecil, tidak ada seasonality signifikan.&nbsp;|&nbsp;
<b>Subscription</b>: No=$60 vs Yes=$59 — hampir identik, status berlangganan tidak mendorong spending lebih tinggi.""")

    section("5.2 — Heatmap Spending: Kategori × Musim")
    pivot_hm = dff.pivot_table(values="Purchase Amount", index="Category",
                                columns="Season", aggfunc="mean").round(1)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot_hm, annot=True, fmt=".1f", cmap=CMAP, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Avg Purchase (USD)"})
    ax.set_title("Heatmap Rata-rata Spending: Kategori × Musim", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("<b>Footwear-Fall (63.7)</b> adalah sel tertinggi. <b>Outerwear adalah baris terendah</b> di hampir semua musim — min. $54.6 (Spring). Range keseluruhan hanya $9 ($54.6–$63.7) → tidak ada kombinasi kategori-musim yang berbeda secara dramatis. Musim Spring paling rendah di hampir semua kategori.")

    section("5.3 — Distribusi Spending: Musim & Metode Pembayaran")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = base_fig(6, 4)
        sns.violinplot(data=dff, x="Season", y="Purchase Amount",
                       palette="Set2", inner="quartile", ax=ax)
        ax.set_title("Purchase Amount per Musim", fontweight="bold")
        ax.set_xlabel("Musim"); ax.set_ylabel("Purchase Amount (USD)")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        order_pay = dff.groupby("Payment Method")["Purchase Amount"].median().sort_values(ascending=False).index
        fig, ax = base_fig(6, 4)
        sns.boxplot(data=dff, x="Payment Method", y="Purchase Amount",
                    order=order_pay, palette="Set3", ax=ax)
        ax.set_title("Purchase Amount per Metode Pembayaran", fontweight="bold")
        ax.tick_params(axis="x", rotation=30); ax.set_xlabel("Metode Pembayaran")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Violin plot: bentuk <b>hampir identik</b> di semua musim → distribusi spending benar-benar konsisten sepanjang tahun. Box plot: <b>semua metode pembayaran memiliki IQR sangat serupa</b> (sekitar $40–$80) dan median ~$60 — tidak ada metode pembayaran yang mencerminkan pola spending berbeda.")

    section("5.4 — Correlation Matrix")
    df_corr = dff.copy()
    for col in ["Gender", "Subscription Status", "Promo Code Used"]:
        df_corr[col] = df_corr[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})
    num_c = ["Age", "Purchase Amount", "Review Rating", "Previous Purchases",
             "Gender", "Subscription Status", "Promo Code Used"]
    corr = df_corr[num_c].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Korelasi Pearson"})
    ax.set_title("Correlation Matrix — Variabel Numerik & Biner", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("""<b>Purchase Amount ≤ |0.03|</b> dengan semua variabel → spending tidak dapat diprediksi dari fitur yang tersedia.
Korelasi bermakna yang terbaca: <b>Subscription Status ↔ Promo Code Used (0.70)</b> — subscriber sangat cenderung pakai promo.
<b>Gender ↔ Promo Code Used (0.60)</b> — pria lebih banyak menggunakan promo code.
<b>Gender ↔ Subscription Status (0.42)</b> — pria lebih cenderung berlangganan.
Age, Review Rating, dan Previous Purchases saling independen (korelasi mendekati 0).""")

    section("5.5 — Scatter Matrix Variabel Numerik")
    num_scatter = ["Age", "Purchase Amount", "Review Rating", "Previous Purchases"]
    pd.plotting.scatter_matrix(dff[num_scatter], figsize=(10, 8), alpha=0.2,
                                hist_kwds={"bins": 20, "color": PAL[0], "edgecolor": "white"},
                                color=PAL[1], diagonal="hist")
    plt.suptitle("Scatter Matrix — Variabel Numerik", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(); st.pyplot(plt.gcf()); plt.close()
    insight("Seluruh pairwise plot menunjukkan sebaran acak tanpa pola linear — keempat variabel benar-benar independen. Histogram diagonal mengkonfirmasi semua variabel <b>terdistribusi uniform/flat</b>. Ini menjelaskan mengapa korelasi mendekati nol di semua pasangan variabel.")

    section("6.1 — Spending Heatmap: Kategori × Musim per Gender")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    fig.suptitle("Avg Spending: Kategori × Musim per Gender", fontsize=12, fontweight="bold")
    for ax, gender in zip(axes, ["Male", "Female"]):
        pivot = (dff[dff["Gender"] == gender]
                 .pivot_table(values="Purchase Amount", index="Category",
                              columns="Season", aggfunc="mean"))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap=CMAP, linewidths=0.5,
                    ax=ax, vmin=50, vmax=75, cbar_kws={"label": "Avg (USD)"})
        ax.set_title(f"Gender: {gender}", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("""Pola kedua gender sangat mirip → gender bukan faktor segmentasi spending yang bermakna.
<b>Pria</b>: nilai tertinggi <b>Footwear-Fall ($65)</b>. Outerwear paling rendah (55–59).
<b>Wanita</b>: nilai tertinggi <b>Accessories-Summer ($63)</b> dan Clothing-Fall ($62). Outerwear wanita juga rendah (55–62).
Tidak ada kategori yang secara konsisten mendominasi di kedua gender. <b>Outerwear bukan kategori premium</b> dari sisi spending rata-rata.""")

    section("6.2 — Dampak Promo Code terhadap Spending")
    c1, c2, c3 = st.columns(3)
    with c1:
        promo_avg = dff.groupby("Promo Code Used")["Purchase Amount"].mean()
        fig, ax = base_fig(4, 3.5)
        bars = ax.bar(promo_avg.index, promo_avg.values,
                      color=[PAL[3], PAL[0]], edgecolor="white", width=0.5)
        ax.set_title("Avg Spend: Promo Code", fontweight="bold")
        ax.set_ylabel("Avg Purchase (USD)"); ax.set_ylim(55, 65)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                    f"${bar.get_height():.1f}", ha="center", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        promo_cat = dff.groupby(["Category", "Promo Code Used"])["Purchase Amount"].mean().unstack()
        fig, ax = base_fig(5, 3.5)
        promo_cat.plot(kind="bar", ax=ax, color=PAL[:2], edgecolor="white", rot=0)
        ax.set_title("Avg Spend per Kategori & Promo", fontweight="bold")
        ax.set_xlabel("Kategori"); ax.set_ylabel("Avg Purchase (USD)")
        ax.legend(title="Promo Code", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c3:
        df_fp = dff[dff["Frequency of Purchases"].isin(URUTAN_FREQ)]
        freq_promo = (df_fp.groupby(["Frequency of Purchases", "Promo Code Used"])
                     .size().unstack().reindex(URUTAN_FREQ))
        fig, ax = base_fig(5, 3.5)
        freq_promo.plot(kind="bar", ax=ax, color=PAL[:2], edgecolor="white", rot=30)
        ax.set_title("Penggunaan Promo per Frekuensi", fontweight="bold")
        ax.set_xlabel("Frekuensi"); ax.set_ylabel("Jumlah Transaksi")
        ax.legend(title="Promo Code", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("""No=$60.1 vs Yes=$59.3 → <b>promo code tidak meningkatkan nilai transaksi</b>, selisih hanya $0.8. Konsisten di semua kategori.
Panel frekuensi: di <b>semua kelompok frekuensi</b>, transaksi tanpa promo selalu lebih banyak (~60:40) → promo digunakan kurang dari separuh pelanggan di semua segmen.
Strategi promo perlu didesain ulang — misalnya dengan threshold minimum pembelian agar benar-benar mendorong peningkatan nilai transaksi.""")

    section("6.3 — Spending per Frekuensi Pembelian")
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
        rev_freq = (df_freq2.groupby("Frequency of Purchases")["Purchase Amount"]
                    .sum().reindex(URUTAN_FREQ))
        fig, ax = base_fig(6, 4)
        bars = ax.bar(rev_freq.index, rev_freq.values, color=PAL[:5], edgecolor="white")
        ax.set_title("Total Revenue per Frekuensi Pembelian", fontweight="bold")
        ax.set_ylabel("Total Revenue (USD)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                    f"${bar.get_height():,.0f}", ha="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("""Median spending (~$60) <b>identik</b> di semua kelompok frekuensi — seberapa sering pun pelanggan berbelanja, nilai per transaksi tetap sama.
Total revenue: <b>Quarterly=$68.859 terbesar</b>, diikuti Fortnightly=$65.207. <b>Weekly justru terkecil ($31.786)</b> — meskipun namanya "weekly", justru kelompok dengan volume transaksi paling sedikit di dataset ini.
Segmen paling berharga dari sisi total revenue: <b>Quarterly dan Fortnightly</b>.""")

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
               wedgeprops={"edgecolor": "white", "linewidth": 1.5})
        ax.set_title("Komposisi Penjualan per Kategori", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        top_items = dff["Item Purchased"].value_counts().head(10)
        fig, ax = base_fig(5, 4)
        ax.barh(top_items.index[::-1], top_items.values[::-1], color=PAL[2], edgecolor="white")
        ax.set_title("Top 10 Item Terlaris", fontweight="bold")
        ax.set_xlabel("Jumlah Transaksi")
        for i, v in enumerate(top_items.values[::-1]):
            ax.text(v + 1, i, str(v), va="center", fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("<b>Clothing</b> mendominasi volume (44.5%), diikuti Accessories (31.8%). Dominasi Clothing di total revenue murni karena volume transaksi terbesar, bukan nilai per item — karena rata-rata harga per item semua kategori sangat serupa ($57–$63).")

    section("6.5 — Rata-rata Spending per Item (Top 5 per Kategori)")
    top_sp = (dff.groupby(["Category", "Item Purchased"])["Purchase Amount"]
               .mean().reset_index()
               .sort_values(["Category", "Purchase Amount"], ascending=[True, False]))
    categories = sorted(dff["Category"].unique())
    cols_items = st.columns(len(categories))
    for ax_col, cat in zip(cols_items, categories):
        sub = top_sp[top_sp["Category"] == cat].head(5)
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        ax.barh(sub["Item Purchased"][::-1], sub["Purchase Amount"][::-1],
                color=PAL[:5], edgecolor="white")
        ax.set_title(cat, fontweight="bold", fontsize=10)
        ax.set_xlabel("Avg USD")
        for bar in ax.patches:
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    f"${bar.get_width():.0f}", va="center", fontsize=7)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()

    insight("""Range harga <b>sangat sempit</b> di semua kategori ($57–$63) — tidak ada item yang jauh lebih mahal atau murah.
<b>Clothing</b>: T-shirt tertinggi ($63).&nbsp;|&nbsp;<b>Footwear</b>: Boots tertinggi ($63), Sandals terendah ($58).
<b>Accessories</b>: Scarf & Hat tertinggi ($61) — range paling sempit.&nbsp;|&nbsp;
<b>Outerwear</b>: Coat=$58, Jacket=$57 → <b>seluruh kategori Outerwear terendah</b> dibanding semua kategori lain. Outerwear bukan kategori premium dari sisi nilai transaksi.""")

    section("Distribusi Warna & Ukuran Produk")
    c1, c2 = st.columns(2)
    with c1:
        top_colors = dff["Color"].value_counts().head(10)
        fig, ax = base_fig(6, 3.8)
        ax.bar(top_colors.index, top_colors.values, color=PAL[3], edgecolor="white")
        ax.set_title("Top 10 Warna Favorit", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    with c2:
        size_order = [s for s in ["XS", "S", "M", "L", "XL"] if s in dff["Size"].values]
        size_c = dff["Size"].value_counts().reindex(size_order).dropna()
        fig, ax = base_fig(6, 3.8)
        bars = ax.bar(size_c.index, size_c.values, color=PAL[4], edgecolor="white")
        ax.set_title("Distribusi Ukuran", fontweight="bold")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(int(bar.get_height())), ha="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Ukuran <b>M sangat mendominasi (1.755)</b>, diikuti L (1.053) — distribusi mengikuti pola normal ukuran pakaian populasi umum. XS (429) dan XL (663) paling sedikit diminati.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: KESIMPULAN & SARAN
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
- Pria ~68%, usia merata 18–70 tahun
- 27% pelanggan berlangganan; mayoritas tidak berlangganan
- Ukuran M mendominasi (45%), XL paling sedikit

**💳 Pola Spending**
- Avg/Median **$60/transaksi**, range $20–$100 (uniform sempurna)
- **Tidak ada variabel** yang memengaruhi spending secara signifikan
- **Footwear & Clothing tertinggi** ($60–$63)
- **Outerwear terendah** ($57–$58) — berlawanan dari dugaan awal
- Fall = musim spending tertinggi, Summer terendah — selisih kecil ($4)
        """)
    with c2:
        st.markdown("""
**📅 Frekuensi & Revenue**
- Frekuensi terbanyak: **Quarterly (1.147)**, Weekly terkecil (539)
- Revenue terbesar: **Quarterly=$68.859**, Weekly terkecil ($31.786)
- Median spending **identik ~$60** di semua kelompok frekuensi

**🔗 Korelasi Penting**
- Purchase Amount: korelasi ≤ |0.03| → **spending tidak dapat diprediksi**
- **Subscription ↔ Promo (0.70)**: subscriber lebih sering pakai promo
- **Gender ↔ Promo (0.60)**: pria lebih banyak pakai promo
- **Gender ↔ Subscription (0.42)**: pria lebih cenderung berlangganan

**🎫 Efektivitas Promo**
- No=$60.1 vs Yes=$59.3 → promo **tidak meningkatkan** nilai transaksi
- Distribusi penggunaan promo ~60:40 (no:yes) merata di semua frekuensi
        """)

    section("Saran Strategis")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**🔍 A. Pengayaan Data**

Fitur saat ini tidak prediktif. Rekomendasikan penambahan:
- Income level / price sensitivity
- Browsing session & click-through rate
- Cart abandonment history
- Wishlist / saved items count
- Customer Lifetime Value historis
- Timestamp transaksi untuk analisis tren
        """)
    with col2:
        st.markdown("""
**🎯 B. Strategi Insentif & Segmen**

- Redesain promo: ganti flat discount dengan **threshold-based** ("belanja >$80 dapat diskon")
- Fokus retensi pada segmen **Quarterly** — revenue terbesar
- Campaign re-aktivasi untuk segmen **Weekly** — volume rendah, potensi besar
- Perkuat value proposition subscription — saat ini tidak membedakan spending vs non-subscriber
        """)
    with col3:
        st.markdown("""
**📦 C. Strategi Produk & Musim**

- **Footwear & Clothing** = avg spending tertinggi → prioritas upsell & cross-sell
- **Outerwear** perlu evaluasi strategi harga — avg spending justru terendah
- Manfaatkan momentum **Fall** — spending tertinggi lintas semua kategori
- Bundle **Footwear + Accessories** berpotensi meningkatkan Average Order Value
        """)

    section("Keterbatasan Analisis")
    st.markdown("""
- Semua variabel numerik terdistribusi uniform → **model prediktif akan sangat terbatas** tanpa fitur tambahan
- Tidak tersedia timestamp transaksi → analisis tren waktu tidak dapat dilakukan
- Tidak ada informasi harga asli produk → sulit membedakan pengaruh volume vs nilai
- Korelasi Gender/Subscription/Promo perlu divalidasi lebih lanjut sebelum dijadikan dasar keputusan bisnis
    """)
