# 🛒 Customer Spending Patterns Dashboard

Dashboard interaktif untuk analisis pola belanja pelanggan — dibangun dengan Streamlit.

## Cara Deploy ke Streamlit Community Cloud

### Langkah 1 — Siapkan File
Pastikan 3 file ini ada dalam satu folder:
```
📁 repo-anda/
 ├── app.py
 ├── requirements.txt
 └── shopping_behavior_updated.csv
```

### Langkah 2 — Upload ke GitHub
1. Buat repo baru di https://github.com/new (Public)
2. Upload ketiga file di atas ke repo tersebut

### Langkah 3 — Deploy di Streamlit Cloud
1. Buka https://share.streamlit.io
2. Login dengan akun GitHub
3. Klik **"Create app"**
4. Isi form:
   - **Repository**: pilih repo yang baru dibuat
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Klik **"Deploy!"**

Selesai! App akan live dalam 2–5 menit. 🎉

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fitur Dashboard

| Halaman | Konten |
|---------|--------|
| 📊 Overview | KPI utama, heatmap, distribusi spending |
| 👤 Demografi | Usia, gender, segmentasi pelanggan |
| 💳 Pola Spending | Violin plot, korelasi, dampak promo |
| 📦 Produk & Kategori | Top items, warna, ukuran |
| 🤖 Machine Learning | Regresi & klasifikasi dengan 6–8 model |
| 🏆 Evaluasi Model | Radar chart, kesimpulan & saran |
