import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── ML ─────────────────────────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             accuracy_score, roc_auc_score, roc_curve,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                              GradientBoostingRegressor, GradientBoostingClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVC
from time import time

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Customer Spending Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;700&family=DM+Mono&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f172a 0%, #1e293b 100%);
    color: white;
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stRadio label { color: #94a3b8 !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }

/* KPI Cards */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0,0,0,.06), 0 4px 16px rgba(0,0,0,.06);
    border-left: 4px solid;
    transition: transform .15s;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-val  { font-size: 2rem; font-weight: 700; line-height: 1.1; margin: .3rem 0 .15rem; }
.kpi-label{ font-size: .75rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }
.kpi-delta{ font-size: .78rem; font-weight: 500; margin-top: .2rem; }

/* Section headers */
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #0f172a;
    border-bottom: 2px solid #e2e8f0; padding-bottom: .5rem; margin: 1.5rem 0 1rem;
}

/* Insight boxes */
.insight-box {
    background: #f8fafc; border-radius: 10px; padding: 1rem 1.2rem;
    border-left: 3px solid #6366f1; margin: .8rem 0; font-size: .88rem; color: #334155;
}
.insight-box b { color: #4f46e5; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING & PREPROCESSING
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

@st.cache_data
def prepare_ml(df):
    df_ml = df.drop(columns=["Age Group", "Spending Tier", "Loyalty Tier"]).copy()
    df_ml["Has Promo"] = (df_ml["Promo Code Used"] == "Yes").astype(int)
    df_ml["Age Group Code"] = pd.cut(df_ml["Age"], bins=[17,25,35,45,55,70],
                                      labels=[0,1,2,3,4]).astype(int)
    urutan_freq = ["Weekly","Fortnightly","Monthly","Quarterly","Annually"]
    freq_map = {f: i for i, f in enumerate(urutan_freq[::-1])}
    df_ml["Freq Ordinal"] = df_ml["Frequency of Purchases"].map(freq_map).fillna(0).astype(int)

    cat_cols = df_ml.select_dtypes(exclude="number").columns.tolist()
    for col in cat_cols:
        df_ml[col] = LabelEncoder().fit_transform(df_ml[col].astype(str))

    X_reg = df_ml.drop(columns=["Purchase Amount"])
    y_reg = df_ml["Purchase Amount"]
    median_spend = y_reg.median()
    df_ml["High Spender"] = (y_reg >= median_spend).astype(int)
    X_cls = df_ml.drop(columns=["Purchase Amount", "High Spender"])
    y_cls = df_ml["High Spender"]
    return X_reg, y_reg, X_cls, y_cls

@st.cache_data
def run_regression(X_reg, y_reg):
    X_tr, X_te, y_tr, y_te = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    SCALE = {"Linear Regression", "Ridge Regression", "KNN Regressor"}

    models = {
        "Linear Regression"  : LinearRegression(),
        "Ridge Regression"   : Ridge(alpha=1.0),
        "Decision Tree"      : DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest"      : RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting"  : GradientBoostingRegressor(n_estimators=100, random_state=42),
        "KNN Regressor"      : KNeighborsRegressor(n_neighbors=10),
    }
    if XGB_AVAILABLE: models["XGBoost"]  = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    if LGB_AVAILABLE: models["LightGBM"] = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)

    results = {}
    for name, m in models.items():
        Xtr = X_tr_sc if name in SCALE else X_tr.values
        Xte = X_te_sc if name in SCALE else X_te.values
        t0 = time(); m.fit(Xtr, y_tr); elapsed = time()-t0
        y_pred = m.predict(Xte)
        cv = cross_val_score(m, Xtr, y_tr, cv=5, scoring="r2").mean()
        fi = getattr(m, "feature_importances_", None)
        results[name] = dict(
            MAE=mean_absolute_error(y_te, y_pred),
            RMSE=np.sqrt(mean_squared_error(y_te, y_pred)),
            R2=r2_score(y_te, y_pred), CV_R2=cv, Time=elapsed,
            y_pred=y_pred, y_test=y_te.values,
            feature_importance=pd.Series(fi, index=X_reg.columns) if fi is not None else None,
        )
    return results

@st.cache_data
def run_classification(X_cls, y_cls):
    X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.2, stratify=y_cls, random_state=42)
    sc = StandardScaler()
    X_tr_sc = sc.fit_transform(X_tr); X_te_sc = sc.transform(X_te)
    SCALE = {"Logistic Regression", "KNN Classifier", "SVM"}

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"      : DecisionTreeClassifier(max_depth=6, random_state=42),
        "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting"  : GradientBoostingClassifier(n_estimators=100, random_state=42),
        "KNN Classifier"     : KNeighborsClassifier(n_neighbors=10),
        "SVM"                : SVC(probability=True, random_state=42),
    }
    if XGB_AVAILABLE: models["XGBoost"]  = XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", verbosity=0)
    if LGB_AVAILABLE: models["LightGBM"] = LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)

    results = {}
    for name, m in models.items():
        Xtr = X_tr_sc if name in SCALE else X_tr.values
        Xte = X_te_sc if name in SCALE else X_te.values
        t0 = time(); m.fit(Xtr, y_tr); elapsed = time()-t0
        y_pred = m.predict(Xte)
        y_prob = m.predict_proba(Xte)[:,1]
        cv = cross_val_score(m, Xtr, y_tr, cv=5, scoring="accuracy").mean()
        fi = getattr(m, "feature_importances_", None)
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        results[name] = dict(
            Accuracy=accuracy_score(y_te, y_pred),
            ROC_AUC=roc_auc_score(y_te, y_prob),
            CV_Acc=cv, Time=elapsed,
            y_pred=y_pred, y_test=y_te.values, y_prob=y_prob,
            fpr=fpr, tpr=tpr,
            cm=confusion_matrix(y_te, y_pred),
            feature_importance=pd.Series(fi, index=X_cls.columns) if fi is not None else None,
        )
    return results

# ── Load ───────────────────────────────────────────────────────────────────────
df = load_data()

PAL = sns.color_palette("Set2", 8)
CMAP = "YlOrRd"

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
        "🤖 Machine Learning",
        "🏆 Evaluasi Model",
    ])

    st.markdown("---")
    st.markdown("**Filter Global**")

    gender_filter = st.multiselect("Gender", df["Gender"].unique().tolist(),
                                    default=df["Gender"].unique().tolist())
    season_filter = st.multiselect("Musim", df["Season"].unique().tolist(),
                                    default=df["Season"].unique().tolist())
    age_min, age_max = st.slider("Rentang Usia", int(df["Age"].min()),
                                  int(df["Age"].max()), (18, 70))

    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} records | 17 fitur")

# Apply filters
mask = (
    df["Gender"].isin(gender_filter) &
    df["Season"].isin(season_filter) &
    df["Age"].between(age_min, age_max)
)
dff = df[mask]

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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Customer Spending Patterns")
    st.caption("Exploratory Data Analysis — Shopping Behavior Dataset")

    # KPI row
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.markdown(kpi(f"{len(dff):,}", "Total Transaksi", "#6366f1",
                    f"{len(dff)/len(df)*100:.0f}% dari total"), unsafe_allow_html=True)
    c2.markdown(kpi(f"${dff['Purchase Amount'].mean():.0f}", "Avg Spending",
                    "#10b981", f"Median ${dff['Purchase Amount'].median():.0f}"), unsafe_allow_html=True)
    c3.markdown(kpi(f"${dff['Purchase Amount'].sum():,.0f}", "Total Revenue",
                    "#f59e0b", "seluruh transaksi"), unsafe_allow_html=True)
    c4.markdown(kpi(f"{dff['Review Rating'].mean():.2f} ★", "Avg Rating",
                    "#ef4444", f"dari 5.0"), unsafe_allow_html=True)
    c5.markdown(kpi(f"{(dff['Subscription Status']=='Yes').mean()*100:.0f}%",
                    "Subscriber Rate", "#8b5cf6",
                    f"{(dff['Subscription Status']=='Yes').sum():,} pelanggan"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue & spending distribution
    section("Distribusi Purchase Amount & Revenue per Kategori")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.hist(dff["Purchase Amount"], bins=30, color=PAL[0], edgecolor="white", linewidth=0.6)
        ax.axvline(dff["Purchase Amount"].mean(), color="red", linestyle="--", linewidth=1.8,
                   label=f"Mean=${dff['Purchase Amount'].mean():.0f}")
        ax.axvline(dff["Purchase Amount"].median(), color="navy", linestyle=":", linewidth=1.8,
                   label=f"Median=${dff['Purchase Amount'].median():.0f}")
        ax.set_title("Distribusi Purchase Amount", fontweight="bold")
        ax.set_xlabel("USD"); ax.set_ylabel("Frekuensi")
        ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight("Distribusi mendekati <b>uniform $20–$100</b>. Mean ≈ Median ≈ $60 → simetris, tidak ada skewness.")

    with c2:
        cat_rev = dff.groupby("Category")["Purchase Amount"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(6,3.5))
        bars = ax.barh(cat_rev.index, cat_rev.values, color=PAL[:len(cat_rev)], edgecolor="white")
        ax.set_title("Total Revenue per Kategori", fontweight="bold")
        ax.set_xlabel("Total Revenue (USD)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x:,.0f}"))
        for bar in bars:
            ax.text(bar.get_width()+200, bar.get_y()+bar.get_height()/2,
                    f"${bar.get_width():,.0f}", va="center", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()
        insight("<b>Clothing</b> mendominasi total revenue karena volume transaksi terbesar, bukan harga tertinggi per item.")

    # Heatmap kategori × musim
    section("Heatmap Avg Spending: Kategori × Musim")
    pivot = dff.pivot_table(values="Purchase Amount", index="Category", columns="Season", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=CMAP, linewidths=0.5, ax=ax,
                cbar_kws={"label":"Avg Purchase (USD)"})
    ax.set_title("Rata-rata Spending per Kategori & Musim", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("<b>Outerwear</b> konsisten tertinggi di semua musim → produk premium & perennial. Tidak ada seasonality yang signifikan.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DEMOGRAFI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 Demografi":
    st.title("👤 Analisis Demografi Pelanggan")

    section("Distribusi Usia & Gender")
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.hist(dff["Age"], bins=25, color=PAL[0], edgecolor="white")
        ax.axvline(dff["Age"].mean(), color="red", linestyle="--",
                   label=f"Mean={dff['Age'].mean():.1f}")
        ax.set_title("Distribusi Usia"); ax.set_xlabel("Usia"); ax.set_ylabel("Count")
        ax.legend(fontsize=8); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        gender_c = dff["Gender"].value_counts()
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.pie(gender_c, labels=gender_c.index, autopct="%1.1f%%",
               colors=PAL[:len(gender_c)], startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":1.5})
        ax.set_title("Distribusi Gender")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        age_g = dff["Age Group"].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(5,3.5))
        bars = ax.bar(age_g.index, age_g.values, color=PAL[:len(age_g)], edgecolor="white")
        ax.set_title("Pelanggan per Kelompok Usia"); ax.set_ylabel("Count")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(int(bar.get_height())), ha="center", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Distribusi usia <b>merata 18–70 tahun</b> (uniform). Pria mendominasi ~68%. Tidak ada segmen usia yang secara signifikan berbeda dalam pola belanja.")

    section("Avg Spending per Kelompok Usia & Gender")
    c1, c2 = st.columns(2)
    with c1:
        age_sp = dff.groupby("Age Group", observed=True)["Purchase Amount"].mean()
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.plot(age_sp.index, age_sp.values, marker="o", color=PAL[2], linewidth=2, markersize=7)
        ax.fill_between(range(len(age_sp)), age_sp.values, alpha=0.15, color=PAL[2])
        ax.set_xticks(range(len(age_sp))); ax.set_xticklabels(age_sp.index)
        ax.set_title("Avg Spending per Kelompok Usia", fontweight="bold")
        ax.set_ylabel("Avg Purchase (USD)"); ax.set_ylim(50, 75)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        gen_sp = dff.groupby(["Age Group","Gender"], observed=True)["Purchase Amount"].mean().unstack()
        fig, ax = plt.subplots(figsize=(6,3.5))
        gen_sp.plot(kind="bar", ax=ax, color=PAL[:2], edgecolor="white", rot=0)
        ax.set_title("Avg Spending: Usia × Gender", fontweight="bold")
        ax.set_xlabel("Kelompok Usia"); ax.set_ylabel("Avg Purchase (USD)")
        ax.legend(title="Gender", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Avg spending <b>flat ~$58–$62</b> di semua kelompok usia → usia bukan prediktor spending. Strategi age-based pricing tidak efektif untuk dataset ini.")

    section("Segmentasi Pelanggan: Spending Tier × Loyalty Tier")
    pivot_seg = dff.pivot_table(values="Customer ID", index="Loyalty Tier",
                                columns="Spending Tier", aggfunc="count", observed=True)
    fig, ax = plt.subplots(figsize=(7,3.5))
    sns.heatmap(pivot_seg, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax,
                cbar_kws={"label":"Jumlah Pelanggan"})
    ax.set_title("Matriks Segmentasi Pelanggan", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("Distribusi <b>sangat merata</b> di semua 9 sel → pelanggan loyal tidak otomatis berbelanja lebih banyak per transaksi. Strategi retensi & upsell perlu dipersonalisasi.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: POLA SPENDING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳 Pola Spending":
    st.title("💳 Pola Spending Pelanggan")

    section("Spending per Musim, Kategori, dan Metode Pembayaran")
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(6,4))
        sns.violinplot(data=dff, x="Season", y="Purchase Amount", palette="Set2",
                       inner="quartile", ax=ax)
        ax.set_title("Distribusi Spending per Musim", fontweight="bold")
        ax.set_xlabel("Musim"); ax.set_ylabel("Purchase Amount (USD)")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        order_pay = dff.groupby("Payment Method")["Purchase Amount"].median().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(6,4))
        sns.boxplot(data=dff, x="Payment Method", y="Purchase Amount",
                    order=order_pay, palette="Set3", ax=ax)
        ax.set_title("Spending per Metode Pembayaran", fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Bentuk violin hampir <b>identik</b> antar musim → spending konsisten sepanjang tahun. Median spending serupa di semua metode pembayaran.")

    section("Dampak Promo Code terhadap Spending")
    c1, c2, c3 = st.columns(3)

    with c1:
        promo_avg = dff.groupby("Promo Code Used")["Purchase Amount"].mean()
        fig, ax = plt.subplots(figsize=(4,3.5))
        bars = ax.bar(promo_avg.index, promo_avg.values, color=[PAL[3],PAL[0]], edgecolor="white", width=0.5)
        ax.set_title("Avg Spend: Promo Code", fontweight="bold")
        ax.set_ylabel("Avg (USD)"); ax.set_ylim(50, 70)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"${bar.get_height():.1f}", ha="center", fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        sub_avg = dff.groupby("Subscription Status")["Purchase Amount"].mean()
        fig, ax = plt.subplots(figsize=(4,3.5))
        bars = ax.bar(sub_avg.index, sub_avg.values, color=[PAL[1],PAL[4]], edgecolor="white", width=0.5)
        ax.set_title("Avg Spend: Subscription", fontweight="bold")
        ax.set_ylabel("Avg (USD)"); ax.set_ylim(50, 70)
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
                    f"${bar.get_height():.1f}", ha="center", fontweight="bold")
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        freq_order = ["Weekly","Fortnightly","Monthly","Quarterly","Annually"]
        freq_rev = (dff[dff["Frequency of Purchases"].isin(freq_order)]
                    .groupby("Frequency of Purchases")["Purchase Amount"].sum()
                    .reindex(freq_order))
        fig, ax = plt.subplots(figsize=(4,3.5))
        bars = ax.bar(freq_rev.index, freq_rev.values, color=PAL[:5], edgecolor="white")
        ax.set_title("Total Revenue per Frekuensi", fontweight="bold")
        ax.set_ylabel("Total (USD)"); ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f"${x/1000:.0f}K"))
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Promo code dan subscription <b>tidak meningkatkan spending per transaksi secara signifikan</b>. Weekly & Fortnightly shoppers adalah kontributor revenue terbesar karena frekuensi, bukan nilai transaksi.")

    section("Correlation Matrix")
    df_corr = dff.copy()
    for col in ["Gender","Subscription Status","Promo Code Used"]:
        df_corr[col] = df_corr[col].map({"Yes":1,"No":0,"Male":1,"Female":0})
    num_cols = ["Age","Purchase Amount","Review Rating","Previous Purchases",
                "Gender","Subscription Status","Promo Code Used"]
    corr = df_corr[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(8,5))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                square=True, linewidths=0.5, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"shrink":0.8,"label":"Korelasi Pearson"})
    ax.set_title("Correlation Matrix", fontweight="bold")
    plt.tight_layout(); st.pyplot(fig); plt.close()
    insight("Purchase Amount <b>hampir tidak berkorelasi</b> dengan semua variabel lainnya → pola spending bersifat acak/uniform. Fitur tambahan (income, browsing history) sangat dibutuhkan.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PRODUK & KATEGORI
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📦 Produk & Kategori":
    st.title("📦 Analisis Produk & Kategori")

    section("Distribusi Kategori & Item Terpopuler")
    c1, c2 = st.columns(2)

    with c1:
        cat_c = dff["Category"].value_counts()
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(cat_c, labels=cat_c.index, autopct="%1.1f%%",
               colors=PAL[:len(cat_c)], startangle=90,
               wedgeprops={"edgecolor":"white","linewidth":1.5})
        ax.set_title("Komposisi Penjualan per Kategori", fontweight="bold")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        top_items = dff["Item Purchased"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.barh(top_items.index[::-1], top_items.values[::-1], color=PAL[2], edgecolor="white")
        ax.set_title("Top 10 Item Terlaris", fontweight="bold"); ax.set_xlabel("Jumlah Transaksi")
        for i, v in enumerate(top_items.values[::-1]):
            ax.text(v+1, i, str(v), va="center", fontsize=8)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    section("Avg Spending per Item — Top 5 per Kategori")
    top_sp = (dff.groupby(["Category","Item Purchased"])["Purchase Amount"]
               .mean().reset_index()
               .sort_values(["Category","Purchase Amount"], ascending=[True,False]))
    categories = sorted(dff["Category"].unique())
    cols = st.columns(len(categories))
    for ax_col, cat in zip(cols, categories):
        sub = top_sp[top_sp["Category"]==cat].head(5)
        fig, ax = plt.subplots(figsize=(3.5,3.5))
        bars = ax.barh(sub["Item Purchased"][::-1], sub["Purchase Amount"][::-1],
                       color=PAL[:5], edgecolor="white")
        ax.set_title(cat, fontweight="bold", fontsize=10); ax.set_xlabel("Avg USD")
        for bar in bars:
            ax.text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                    f"${bar.get_width():.0f}", va="center", fontsize=7)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); ax_col.pyplot(fig); plt.close()

    insight("<b>Outerwear</b> adalah kategori premium — hampir semua itemnya >$60. <b>Accessories</b> memiliki range harga sempit dan lebih terjangkau.")

    section("Distribusi Warna & Ukuran")
    c1, c2 = st.columns(2)
    with c1:
        top_colors = dff["Color"].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(6,3.5))
        ax.bar(top_colors.index, top_colors.values, color=PAL[3], edgecolor="white")
        ax.set_title("Top 10 Warna Favorit", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        size_order = [s for s in ["XS","S","M","L","XL"] if s in dff["Size"].values]
        size_c = dff["Size"].value_counts().reindex(size_order).dropna()
        fig, ax = plt.subplots(figsize=(6,3.5))
        bars = ax.bar(size_c.index, size_c.values, color=PAL[4], edgecolor="white")
        ax.set_title("Distribusi Ukuran", fontweight="bold")
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5,
                    str(int(bar.get_height())), ha="center", fontsize=9)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MACHINE LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Machine Learning":
    st.title("🤖 Machine Learning — Prediksi Spending")

    with st.spinner("Training models... (sekali saja, hasil di-cache)"):
        X_reg, y_reg, X_cls, y_cls = prepare_ml(df)
        reg_res  = run_regression(X_reg, y_reg)
        cls_res  = run_classification(X_cls, y_cls)

    task = st.radio("Pilih Task ML", ["Regresi — Prediksi Purchase Amount",
                                       "Klasifikasi — Prediksi High Spender"],
                    horizontal=True)

    if task.startswith("Regresi"):
        section("Hasil Model Regresi")
        best_reg = max(reg_res, key=lambda k: reg_res[k]["R2"])
        df_reg = pd.DataFrame({
            "Model"        : list(reg_res.keys()),
            "MAE (USD)"    : [f"{reg_res[m]['MAE']:.2f}"   for m in reg_res],
            "RMSE (USD)"   : [f"{reg_res[m]['RMSE']:.2f}"  for m in reg_res],
            "R² Score"     : [f"{reg_res[m]['R2']:.4f}"    for m in reg_res],
            "CV R² (5-fold)": [f"{reg_res[m]['CV_R2']:.4f}" for m in reg_res],
            "Time (s)"     : [f"{reg_res[m]['Time']:.2f}"  for m in reg_res],
        }).sort_values("R² Score", ascending=False)
        st.dataframe(df_reg, use_container_width=True)

        st.markdown(f"**🏆 Model terbaik: `{best_reg}` — R² = {reg_res[best_reg]['R2']:.4f}**")
        insight("R² mendekati 0 di semua model → Purchase Amount yang <b>hampir uniform $20–$100</b> sangat sulit diprediksi dari fitur demografis & transaksi saja.")

        c1, c2 = st.columns(2)
        with c1:
            section("Actual vs Predicted")
            y_test = reg_res[best_reg]["y_test"]
            y_pred = reg_res[best_reg]["y_pred"]
            fig, ax = plt.subplots(figsize=(5,4))
            ax.scatter(y_test, y_pred, alpha=0.3, color=PAL[2], s=15)
            mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
            ax.plot([mn,mx],[mn,mx],"r--",linewidth=2, label="Perfect fit")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs Predicted — {best_reg}", fontweight="bold")
            ax.legend(); ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            section("Feature Importance")
            fi_models = {k:v for k,v in reg_res.items() if v["feature_importance"] is not None}
            if fi_models:
                sel = st.selectbox("Pilih model", list(fi_models.keys()))
                fi = fi_models[sel]["feature_importance"].sort_values(ascending=False).head(10)
                fig, ax = plt.subplots(figsize=(5,4))
                ax.barh(fi.index[::-1], fi.values[::-1], color=PAL[0], edgecolor="white")
                ax.set_title(f"Feature Importance — {sel}", fontweight="bold")
                ax.set_xlabel("Importance Score")
                ax.spines[["top","right"]].set_visible(False)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    else:
        section("Hasil Model Klasifikasi")
        best_cls = max(cls_res, key=lambda k: cls_res[k]["ROC_AUC"])
        df_cls = pd.DataFrame({
            "Model"               : list(cls_res.keys()),
            "Accuracy"            : [f"{cls_res[m]['Accuracy']:.4f}" for m in cls_res],
            "ROC-AUC"             : [f"{cls_res[m]['ROC_AUC']:.4f}"  for m in cls_res],
            "CV Accuracy (5-fold)": [f"{cls_res[m]['CV_Acc']:.4f}"   for m in cls_res],
            "Time (s)"            : [f"{cls_res[m]['Time']:.2f}"     for m in cls_res],
        }).sort_values("ROC-AUC", ascending=False)
        st.dataframe(df_cls, use_container_width=True)
        st.markdown(f"**🏆 Model terbaik: `{best_cls}` — ROC-AUC = {cls_res[best_cls]['ROC_AUC']:.4f}**")
        insight("Accuracy & ROC-AUC ≈ 0.50 → semua model setara tebak koin. Butuh fitur lebih informatif (income, browsing history) untuk performa bermakna.")

        c1, c2 = st.columns(2)
        with c1:
            section("Confusion Matrix")
            cm = cls_res[best_cls]["cm"]
            fig, ax = plt.subplots(figsize=(4,3.5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Low","High"], yticklabels=["Low","High"], linewidths=0.5)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(f"Confusion Matrix — {best_cls}", fontweight="bold")
            plt.tight_layout(); st.pyplot(fig); plt.close()

        with c2:
            section("ROC Curve")
            fig, ax = plt.subplots(figsize=(4,3.5))
            for i, (name, res) in enumerate(cls_res.items()):
                lw = 2.5 if name == best_cls else 0.8
                ax.plot(res["fpr"], res["tpr"],
                        label=f"{name} ({res['ROC_AUC']:.3f})",
                        linewidth=lw, color=PAL[i % len(PAL)])
            ax.plot([0,1],[0,1],"k--",linewidth=1,label="Random")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("ROC Curve", fontweight="bold")
            ax.legend(fontsize=6, loc="lower right")
            ax.spines[["top","right"]].set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Evaluasi Model":
    st.title("🏆 Evaluasi & Komparasi Model")

    with st.spinner("Loading model results..."):
        X_reg, y_reg, X_cls, y_cls = prepare_ml(df)
        reg_res  = run_regression(X_reg, y_reg)
        cls_res  = run_classification(X_cls, y_cls)

    section("Radar Chart — Regresi")
    def norm_inv(s): return 1 - (s - s.min()) / (s.max() - s.min() + 1e-9)
    def norm(s):     return     (s - s.min()) / (s.max() - s.min() + 1e-9)

    mnr = list(reg_res.keys())
    mtr = {
        "MAE (inv)" : norm_inv(pd.Series([reg_res[m]["MAE"]   for m in mnr])),
        "RMSE (inv)": norm_inv(pd.Series([reg_res[m]["RMSE"]  for m in mnr])),
        "R²"        : norm    (pd.Series([reg_res[m]["R2"]    for m in mnr])),
        "CV R²"     : norm    (pd.Series([reg_res[m]["CV_R2"] for m in mnr])),
    }
    mlr = list(mtr.keys()); Nr = len(mlr)
    ang_r = np.linspace(0, 2*np.pi, Nr, endpoint=False).tolist(); ang_r += ang_r[:1]
    best_reg = max(reg_res, key=lambda k: reg_res[k]["R2"])

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax.set_thetagrids(np.degrees(ang_r[:-1]), mlr)
        for i, name in enumerate(mnr):
            vals = [mtr[m][i] for m in mlr] + [mtr[mlr[0]][i]]
            lw = 2.5 if name == best_reg else 0.9
            ax.plot(ang_r, vals, lw=lw, label=name, color=PAL[i%len(PAL)])
            ax.fill(ang_r, vals, alpha=0.05, color=PAL[i%len(PAL)])
        ax.set_ylim(0,1); ax.set_title("Model Regresi", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.5,1.1), fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    mnc = list(cls_res.keys())
    mtc = {
        "Accuracy"   : norm(pd.Series([cls_res[m]["Accuracy"] for m in mnc])),
        "ROC-AUC"    : norm(pd.Series([cls_res[m]["ROC_AUC"]  for m in mnc])),
        "CV Acc"     : norm(pd.Series([cls_res[m]["CV_Acc"]   for m in mnc])),
        "Speed (inv)": norm_inv(pd.Series([cls_res[m]["Time"] for m in mnc])),
    }
    mlc = list(mtc.keys()); Nc = len(mlc)
    ang_c = np.linspace(0, 2*np.pi, Nc, endpoint=False).tolist(); ang_c += ang_c[:1]
    best_cls = max(cls_res, key=lambda k: cls_res[k]["ROC_AUC"])

    with c2:
        fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
        ax.set_thetagrids(np.degrees(ang_c[:-1]), mlc)
        for i, name in enumerate(mnc):
            vals = [mtc[m][i] for m in mlc] + [mtc[mlc[0]][i]]
            lw = 2.5 if name == best_cls else 0.9
            ax.plot(ang_c, vals, lw=lw, label=name, color=PAL[i%len(PAL)])
            ax.fill(ang_c, vals, alpha=0.05, color=PAL[i%len(PAL)])
        ax.set_ylim(0,1); ax.set_title("Model Klasifikasi", fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.5,1.1), fontsize=7)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    insight("Area lebih <b>luas & terluar</b> = performa lebih baik. <b>Speed (inv)</b> = semakin cepat training, semakin tinggi skornya. Radar membantu melihat trade-off antar model sekaligus.")

    section("Kesimpulan & Saran")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**📊 Temuan EDA**
- Spending **uniform $20–$100**, mean ≈ median ≈ $60
- Gender, usia, musim, subscription **tidak memengaruhi** spending secara signifikan
- **Outerwear** = kategori premium & perennial
- **Weekly/Fortnightly** shoppers = segmen revenue terbesar
- Promo code **tidak meningkatkan** nilai transaksi
        """)
    with col2:
        st.markdown("""
**🤖 Temuan ML**
- R² ≈ 0 dan ROC-AUC ≈ 0.50 → fitur saat ini tidak prediktif
- Masalah pada **data**, bukan algoritma
- **Random Forest & Gradient Boosting** sedikit lebih baik dari yang lain
- Butuh fitur baru: income level, browsing history, cart abandonment

**💡 Rekomendasi**
- Tambah fitur kontekstual untuk meningkatkan prediksi
- Redesain promo: threshold-based incentive
- Fokus retensi pada pelanggan **Weekly/Fortnightly**
- Implementasikan **Collaborative Filtering** untuk rekomendasi produk
        """)
