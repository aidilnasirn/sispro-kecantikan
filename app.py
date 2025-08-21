# app_streamlit.py
# Streamlit UI untuk BeautyProductRecommendationSystem (versi final dengan jenis kulit lengkap)
from __future__ import annotations

import io
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore")

# =========================
# Util
# =========================
def read_csv_flex(path_or_buffer):
    """Reader fleksibel yang disempurnakan untuk menangani buffer dari Streamlit."""
    try:
        string_data = path_or_buffer.getvalue().decode('utf-8')
        path_or_buffer = io.StringIO(string_data)
    except Exception:
        path_or_buffer.seek(0)

    for delimiter in [',', ';']:
        try:
            path_or_buffer.seek(0)
            return pd.read_csv(path_or_buffer, delimiter=delimiter)
        except Exception:
            continue
    path_or_buffer.seek(0)
    raise ValueError("Tidak dapat membaca file CSV. Pastikan pemisah kolom adalah koma (,) atau titik koma (;).")


def safe_price_bounds(series: pd.Series) -> tuple[int, int]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0, 500000
    vmin = int(s.min())
    vmax = int(s.max())
    if vmin >= vmax:
        return max(0, vmin - 1), vmax + 50000
    return vmin, vmax

# =========================
# Core class
# =========================
class BeautyProductRecommendationSystem:
    def __init__(self):
        self.products_df = None
        self.tfidf_vectorizer = None
        self.similarity_matrix = None

    def _canonicalize_skin_token(self, raw: str) -> str | None:
        if not raw: return None
        t = str(raw).lower().strip()
        t = re.sub(r"\b(kulit|dan)\b", "", t).strip()
        CANON = {"berjerawat","berminyak","kering","sensitif","normal","kombinasi","kusam","semua"}
        if t in CANON: return t
        if "semua" in t: return "semua"
        if "acne" in t or "jerawat" in t: return "berjerawat"
        if "oily" in t or "berminyak" in t: return "berminyak"
        if "dry" in t or "kering" in t: return "kering"
        if "sensitive" in t or "sensitif" in t: return "sensitif"
        if "comb" in t or "kombinasi" in t: return "kombinasi"
        if "dull" in t or "kusam" in t: return "kusam"
        return None

    def _parse_skin_tokens(self, s) -> set[str]:
        if pd.isna(s): return {"semua"}
        text = str(s).lower()
        if "semua" in text: return {"semua"}
        parts = re.split(r"[,|/;&]", text)
        toks = {self._canonicalize_skin_token(p) for p in parts if self._canonicalize_skin_token(p)}
        return toks if toks else {"semua"}

    def _normalize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).lower().strip().replace(" ", "_").replace(".", "") for c in df.columns]

        alias_map = {
            "product_name": "nama_produk", "merek": "brand", "subkategori": "sub_kategori",
            "skin_type": "jenis_kulit_kompatibel", "description": "deskripsi",
            "harga": "harga_idr", "price": "harga_idr", "size": "size_ml",
            "gambar": "url_gambar", "image": "url_gambar", "link_gambar": "url_gambar"
        }
        df.rename(columns=alias_map, inplace=True)

        required_cols = {
            "id", "nama_produk", "brand", "sub_kategori", "manfaat", "jenis_kulit_kompatibel",
            "rating", "deskripsi", "harga_idr", "size_ml", "klaim", "url_gambar"
        }
        for col in required_cols:
            if col not in df.columns:
                df[col] = "" if col not in ["rating", "harga_idr", "size_ml"] else np.nan

        for col in ["rating", "harga_idr", "size_ml"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in list(required_cols):
            if col not in ["rating", "harga_idr", "size_ml", "id"]:
                df[col] = df[col].astype(str).fillna('')
        return df

    def load_and_preprocess_data(self, uploaded_df: pd.DataFrame | None = None):
        if uploaded_df is not None:
            self.products_df = uploaded_df.copy()
        else:
            data = [
            # Skintific
            [1, "Skintific", "Skintific", "Panthenol Cleanser", "Sabun cuci muka", "Kulit jerawat & Kemerahan", 5.00, "Panthenol: menenangkan iritasi kulit", 89000, 120, "Hydrating|Brightening", "https://images.tokopedia.net/img/cache/700/VqbcmM/2023/5/31/c8c5e5f0-5f5e-4b4a-9c4c-8e8f5f5f5f5f.jpg"],
            [1, "Skintific", "Skintific", "Niacinamide Toner", "Sebagai booster pencerah kulit", "Kulit kusam", 4.9, "Toner dengan kandungan Triple Brightening Agents", 150000, 100, "Hydrating|Soothing|Barrier", "https://images.soco.id/b5be06c1-f275-4aa6-a93d-01fcb786e608-.jpg"],
            [1, "Skintific", "Skintific", "Dark Spot Serum", "Memudarkan noda hitam", "Kulit kusam, noda hitam, jerawat", 5.00, "Skintific SymWhite377 Dark Spot Serum", 139000, 20, "Exfoliating|Brightening", "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkxMwaigURX06oCNV3VDVHBnStQgHP3E4AZt-gFmA27DFT0KiUxUP7HpZrKQytkT1X4wk&usqp=CAU"],
            [1, "Skintific", "Skintific", "Moisturizer 5x Ceramide", "Mengontrol minyak & jerawat", "Kulit berminyak dan berjerawat", 4.9, "Skintific 5x Ceramide Barrier Moisture Gel 30g", 139000, 30, "Anti-acne|Oil control", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [1, "Skintific", "Skintific", "Daily Sunscreen", "Menenangkan dan melembabkan", "Kulit berjerawat", 5.00, "The Water-Like Elegant sunscreen yang ringan", 109000, 30, "Hydrating|Soothing", "https://cf.shopee.co.id/file/id-11134207-23030-pz7j7j7j7j7j7j"],
            # Glad2Glow
            [2, "Glad2Glow", "Glad2Glow", "Micellar Water", "Pembersih make up", "Kulit berjerawat", 5.00, "Pembersih makeup yang lembut", 28000, 80, "Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [2, "Glad2Glow", "Glad2Glow", "Face wash", "Sabun cuci muka", "Kulit berjerawat, kering, berminyak", 4.9, "Glad2Glow Tremella Vita B5 Cleanser GEL", 29000, 70, "Gentle Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [2, "Glad2Glow", "Glad2Glow", "Ceramide Moisturizer", "Merawat Skin Barrier", "Kulit kering, berminyak, sensitif", 4.7, "Moisturizer dengan ekstrak blueberry dan ceramide", 34000, 30, "Soothing|Barrier", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [2, "Glad2Glow", "Glad2Glow", "Pomegranate Serum", "Mencerahkan kulit", "Kulit kusam, bertekstur", 4.9, "Serum Niacinamide untuk mencerahkan", 35000, 17, "Brightening|Antioxidant", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # The Originote
            [3, "The Originote", "The Originote", "Micellar Water", "Membersihkan Kotoran & Makeup", "Semua jenis kulit", 5.00, "Micellar water untuk makeup waterproof", 45000, 300, "Cleansing|Oil control", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [3, "The Originote", "The Originote", "Facial Cleanser", "Pembersih Muka", "Semua jenis kulit", 5.00, "The originote Low Ph Cicamide facial wash", 45000, 150, "Gentle Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [3, "The Originote", "The Originote", "Brightening Moisturizer", "Pelembab dan mencerahkan", "Kering, normal, kusam", 5.00, "Moisturizer dengan Hyaluron dan Ceramide", 34000, 50, "Brightening|Barrier", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [3, "The Originote", "The Originote", "Niacinamide serum", "Menjaga kelembapan kulit", "Semua jenis kulit", 5.00, "Serum 10% Niacinamide untuk mencerahkan", 25000, 80, "Hydrating|Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [3, "The Originote", "The Originote", "Cica-B5 Soothing Toner", "Menenangkan kemerahan", "Kulit kering, berjerawat", 5.00, "The originote Cica-B5 Soothing Essence toner", 49000, 80, "Soothing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [3, "The Originote", "The Originote", "Mugwort B3 Clay Stick", "Menenangkan & mengencangkan", "Semua jenis kulit", 5.00, "Mugwort B3 Clay Stick Mask", 50000, 40, "Barrier|Soothing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [3, "The Originote", "The Originote", "Caramella Sunscreen", "Perawatan muka berminyak", "Kulit berminyak, berjerawat", 5.00, "The originote caramella sunscreen SPF 50", 40000, 50, "UV Protection", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # Scarlett
            [4, "Scarlett", "Scarlett", "Facial Wash", "Pembersih Wajah", "Semua jenis kulit", 5.00, "Facial wash mengandung Glutathione & Vitamin E", 50000, 100, "Brightening|Antioxidant", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [4, "Scarlett", "Scarlett", "Acne Essence Toner", "Menenangkan jerawat meradang", "Kulit berjerawat", 5.00, "Toner dengan kandungan Green Tea Water", 60000, 100, "Anti-acne|Soothing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [4, "Scarlett", "Scarlett", "Ceramide Moisturizer", "Merawat Skin Barrier", "Semua jenis kulit", 5.00, "Scarlett whitening 7x ceramide moisturizer", 56000, 20, "Barrier Repair", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [4, "Scarlett", "Scarlett", "Niacinamide Serum", "Serum pencerah untuk pemula", "Kering, normal", 5.00, "Scarlett whitening niacinemide 5% serum", 78000, 15, "Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # MS Glow
            [5, "MS Glow", "MS Glow", "Facial Wash", "Pembersih wajah", "Semua jenis kulit", 5.00, "Pencuci muka untuk membersihkan kotoran", 45000, 50, "Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [5, "MS Glow", "MS Glow", "WhitecallDNA Toner", "Melembabkan & menyamarkan noda", "Semua jenis kulit", 5.00, "Toner diformulasikan dengan whitecelldna", 45000, 50, "Even Tone", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [5, "MS Glow", "MS Glow", "Calm Blemish Moisturizer", "Menenangkan kulit sensitif", "Kulit berjerawat, sensitif", 5.00, "Moisturizer dengan panthenol & centella asiatica", 56000, 40, "Soothing|Anti-acne", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [5, "MS Glow", "MS Glow", "Whitecell DnA Serum", "Mencerahkan & menyamarkan flek", "Semua jenis kulit", 5.00, "Mengandung niacinemide, Glycolic Acid", 35000, 15, "Brightening|Anti-aging", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # Scora
            [6, "Scora", "Scora", "Micellar Cleansing Water", "Mengangkat Makeup & Kotoran", "Kulit sensitif", 5.00, "Scora Gentle and Sooth Micellar water", 67000, 100, "Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [6, "Scora", "Scora", "Gentle Low Ph Cleanser", "Sabun muka oily & acne prone", "Kulit berminyak", 5.00, "Pembersih wajah dengan kandungan pH rendah", 50000, 100, "Oil Control", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [6, "Scora", "Scora", "Arbutin Serum", "Meratakan warna kulit", "Kulit jerawat dan berminyak", 5.00, "Serum All-in-one brightening", 50000, 20, "Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [6, "Scora", "Scora", "Bright me up Sunscreen", "Mencerahkan & melindungi dari UV", "Semua jenis kulit", 5.00, "Diformulasikan dengan teknologi hybrid SPF 40", 50000, 40, "UV Protection", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # Benings
            [7, "Benings", "Bening's Skincare", "Facial Wash", "Mencerahkan & menghilangkan flek", "Kombinasi", 5.00, "Diformulasikan dengan bahan aktif dari Eropa", 100000, 60, "Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [7, "Benings", "Bening's Skincare", "Daily Sunscreen", "Melindungi dari UV & melembabkan", "Kombinasi, kusam, berminyak", 5.00, "Tabir surya harian untuk melindungi kulit", 40000, 50, "UV Protection", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # Wardah
            [8, "Wardah", "Wardah", "Bright+ Tone Up Micellar", "Membersihkan & mencerahkan", "Kulit kusam", 5.00, "Dengan teknologi 3 powerful actions", 40000, 100, "Cleansing|Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [8, "Wardah", "Wardah", "Facial Wash Nature Daily", "Sabun cuci muka lembut", "Semua jenis kulit", 5.00, "Membersihkan tanpa merusak skin barrier", 59000, 50, "Gentle Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [8, "Wardah", "Wardah", "Essence Toner", "Menyamarkan noda hitam", "Semua jenis kulit", 5.00, "Nature daily aloe hydramild essence toner", 43000, 100, "Even Tone", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [8, "Wardah", "Wardah", "Glowshot Day Moisturizer", "Pelembab & pemutih wajah", "Normal, berminyak", 5.00, "Efektif mencerahkan dan terlindungi", 53000, 30, "Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [8, "Wardah", "Wardah", "Acne Calming Sunscreen", "Melindungi & mengatasi jerawat", "Berminyak, berjerawat", 5.00, "Wardah UV Acne calming sunscreen SPF 35", 85000, 35, "UV Protection|Anti-acne", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # Pond's
            [9, "Pond's", "Pond's", "Micellar Miracle Water", "Membersihkan & mengangkat makeup", "Semua jenis kulit", 5.00, "Kekuatan 3-in-1 untuk menghapus makeup", 96000, 100, "Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [9, "Pond's", "Pond's", "Facial Foam", "10x lebih efektif membersihkan pori", "Semua jenis kulit", 5.00, "Sabun cuci muka dengan teknologi D-TOXX", 43000, 100, "Deep Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [9, "Pond's", "Pond's", "Night Serum", "Memudarkan noda hitam", "Semua jenis kulit", 5.00, "Serum dengan NIASORCINOL", 58000, 14, "Brightening|Repairing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [9, "Pond's", "Pond's", "Brightening Moisturizer", "Menjaga skin biome", "Kulit kering, jerawat", 5.00, "Moisturizer ringan dengan PRE-BIOTICS", 32000, 20, "Balancing|Hydrating", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [9, "Pond's", "Pond's", "Moisturizer", "Menenangkan kemerahan kulit", "Semua jenis kulit", 5.00, "Pond's Juice Collection Moisturizer", 25000, 20, "Soothing|Hydrating", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            # Emina
            [10, "Emina", "Emina", "Micellar Water", "Membersihkan debu dan kotoran", "Kering, kusam, berjerawat", 5.00, "Emina fife star micellar water pH 5.5", 89000, 100, "Gentle Cleansing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [10, "Emina", "Emina", "Face Wash", "Mencerahkan & memperkuat skin barrier", "Semua jenis kulit", 5.00, "Emina brightening face wash with niacinemide", 54000, 100, "Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [10, "Emina", "Emina", "Bright Stuff Face Toner", "Membersihkan & mencerahkan kulit", "Semua jenis kulit", 5.00, "Emina bright stuff yang menghidrasi kulit", 23000, 100, "Refreshing|Brightening", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"],
            [10, "Emina", "Emina", "Bright Stuff Face Serum", "Membuat kulit lembab & elastis", "Semua jenis kulit", 5.00, "Serum diformulasikan dengan Oxybiome", 56000, 30, "Hydrating|Glowing", "https://images.tokopedia.net/img/cache/700/hDjmkQ/2023/10/26/1d505370-96f7-4148-8167-bd1c9258ac7f.jpg"]
        ]
            columns = ['id', 'nama_produk', 'brand', 'sub_kategori', 'manfaat', 'jenis_kulit_kompatibel', 'rating', 'deskripsi', 'harga_idr', 'size_ml', 'klaim', 'url_gambar']
            self.products_df = pd.DataFrame(data, columns=columns)

        self.products_df = self._normalize_schema(self.products_df)
        self.products_df.dropna(subset=['nama_produk', 'jenis_kulit_kompatibel', 'sub_kategori'], inplace=True)
        
        text_features = ['jenis_kulit_kompatibel', 'sub_kategori', 'manfaat', 'klaim']
        self.products_df['combined_features'] = self.products_df[text_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1).str.lower()
        
        self.products_df["skin_tokens"] = self.products_df["jenis_kulit_kompatibel"].apply(self._parse_skin_tokens)
        self.products_df.reset_index(drop=True, inplace=True)

    def build_similarity_matrix(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words=None, ngram_range=(1, 2))
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.products_df['combined_features'])
        self.similarity_matrix = cosine_similarity(tfidf_matrix)

    def normalize_skin_type(self, user_skin_type):
        skin_type_mapping = {
            'berjerawat': ['berjerawat', 'jerawat', 'acne'], 'berminyak': ['berminyak', 'oily', 'minyak'],
            'kering': ['kering', 'dry'], 'sensitif': ['sensitif', 'sensitive'],
            'normal': ['normal'], 'kombinasi': ['kombinasi', 'combination'], 'kusam': ['kusam', 'dull']
        }
        user_skin_type_lower = user_skin_type.lower()
        for standard_type, variants in skin_type_mapping.items():
            if any(variant in user_skin_type_lower for variant in variants):
                return standard_type
        return user_skin_type_lower

    def find_compatible_products(self, user_skin_type):
        normalized_skin_type = self.normalize_skin_type(user_skin_type)
        mask = self.products_df["skin_tokens"].apply(lambda s: "semua" in s or normalized_skin_type in s)
        return self.products_df[mask].copy()

    def rank_on_subset(self, subset_indices: list[int], top_n=5):
        if not subset_indices or self.similarity_matrix is None or max(subset_indices) >= len(self.products_df):
            return []
        
        sim_sub_matrix = self.similarity_matrix[np.ix_(subset_indices, subset_indices)]
        avg_sim_scores = sim_sub_matrix.mean(axis=1)
        
        subset_df = self.products_df.loc[subset_indices]
        ratings = pd.to_numeric(subset_df['rating'], errors='coerce').fillna(3.0) / 5.0
        
        final_scores = (avg_sim_scores * 0.6) + (ratings.values * 0.4)
        
        ranked_df = pd.DataFrame({'original_index': subset_indices, 'score': final_scores})
        ranked_df = ranked_df.sort_values(by='score', ascending=False).head(top_n)
        
        top_products = []
        for _, row in ranked_df.iterrows():
            idx = int(row['original_index'])
            top_products.append((idx, row['score'], self.products_df.loc[idx]))
        return top_products

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Sistem Rekomendasi Kecantikan", page_icon="SISKOM", layout="wide")

PINK_AND_CARD_CSS = """
<style>
  :root{--pink-1:#ffe6f1;--pink-3:#ff8ab8;--pink-4:#ff5fa2}
  .pink-bg{background:radial-gradient(900px 400px at 0% 0%, var(--pink-1), transparent 60%)}
  .stButton>button{background:linear-gradient(90deg,var(--pink-4),var(--pink-3));color:#fff;border:none;border-radius:999px;padding:.55rem 1rem}
  .stButton>button:hover{transform:translateY(-1px);box-shadow:0 8px 22px rgba(255,95,162,.25)}
  .chip{display:inline-flex;gap:.35rem;padding:.2rem .6rem;border-radius:999px;background:#fff0f7;border:1px solid #ffd6e8;color:#7d2a4f;font-size:.8rem;font-weight:600;margin-right:.35rem;margin-bottom:.35rem}
  .product-card { background-color: #ffffff; border: 1px solid #f0f0f0; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; height: 100%; display: flex; flex-direction: column; }
  .product-card:hover { transform: translateY(-5px); box-shadow: 0 8px 20px rgba(0,0,0,0.1); }
  .product-card img { width: 100%; height: 180px; object-fit: cover; border-radius: 8px; margin-bottom: 1rem; background-color: #f8f8f8; }
  .product-card .brand { font-size: 0.8rem; color: #888; font-weight: 600; text-transform: uppercase; }
  .product-card h3 { font-size: 1.1rem; font-weight: 700; margin-top: 0.2rem; margin-bottom: 0.5rem; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden; min-height: 2.5em; }
  .product-card .subcategory { font-size: 0.9rem; color: #555; margin-bottom: 0.7rem; font-style: italic; }
  .product-card .price { font-size: 1.2rem; font-weight: 700; color: var(--pink-4); margin-bottom: 0.5rem; }
  .product-card .details { font-size: 0.9rem; color: #555; margin-bottom: 0.5rem; }
  
  /* === PENYESUAIAN CSS DI SINI === */
  .product-card .skin-type { 
    font-size: 0.8rem; 
    color: #7d2a4f; 
    background-color: #fff0f7;
    padding: 0.2rem 0.5rem; 
    border-radius: 5px; 
    margin-bottom: 0.7rem;
    /* Properti pembatas teks dihapus, diganti dengan ini agar bisa wrap */
    white-space: normal; 
    word-wrap: break-word;
  }
  .product-card .manfaat, .product-card .deskripsi { 
    font-size: 0.85rem; 
    color: #666; 
    margin-bottom: 0.5rem; 
  }
  .product-card .score { margin-top: auto; padding-top: 0.5rem; font-size: 0.8rem; color: #aaa; text-align: right; }
</style>
"""
st.markdown(PINK_AND_CARD_CSS, unsafe_allow_html=True)

st.markdown('<div class="pink-bg"><h2>Sistem Rekomendasi Produk Kecantikan</h2></div>', unsafe_allow_html=True)
st.caption("Menggunakan Metode **Machine Learning**.")

recomm = BeautyProductRecommendationSystem()
uploaded = st.sidebar.file_uploader("Unggah dataset.csv Anda", type=["csv"])

try:
    if uploaded is not None:
        up_df = read_csv_flex(uploaded)
        st.sidebar.success(f"CSV terunggah: {uploaded.name} — {len(up_df)} baris")
        recomm.load_and_preprocess_data(uploaded_df=up_df)
    else:
        recomm.load_and_preprocess_data()
    
    recomm.build_similarity_matrix()
    df = recomm.products_df.copy()

    st.sidebar.header("Preferensi Pengguna")
    
    if 'skin_tokens' in df and not df['skin_tokens'].empty:
        all_skin_tokens = set()
        for token_set in df['skin_tokens']:
            all_skin_tokens.update(token_set)
        skin_opts = sorted(list(all_skin_tokens))
        
        skin_display = [s.title() for s in skin_opts]
        if skin_display:
            try:
                default_index = skin_display.index("Semua") if "Semua" in skin_display else 0
                skin_choice = st.sidebar.selectbox("Jenis kulit Anda", skin_display, index=default_index)
                skin_choice_raw = skin_opts[skin_display.index(skin_choice)]
            except (ValueError, IndexError):
                skin_choice = st.sidebar.selectbox("Jenis kulit Anda", skin_display, index=0)
                skin_choice_raw = skin_opts[0] if skin_opts else "semua"
        else:
            skin_choice_raw = "semua"
    else:
        skin_choice_raw = "semua"

    sub_opts = sorted(df["sub_kategori"].dropna().astype(str).str.strip().unique().tolist())
    sub_kat = st.sidebar.selectbox("Sub Kategori (opsional)", ["(Semua)"] + sub_opts, index=0)
    
    brand_opts = ["(Semua)"] + sorted(df["brand"].dropna().astype(str).unique().tolist())
    brand_choice = st.sidebar.selectbox("Brand (opsional)", brand_opts, index=0)
    
    min_rating = st.sidebar.slider("Rating minimum", 0.0, 5.0, 4.0, 0.1)
    lo, hi = safe_price_bounds(df["harga_idr"])
    price_range = st.sidebar.slider("Harga (IDR)", min_value=lo, max_value=hi, value=(lo, hi))
    top_n = st.sidebar.slider("Top-N", 3, 20, 8)

    if st.button("Rekomendasikan", use_container_width=True):
        compatible = recomm.find_compatible_products(skin_choice_raw)
        
        mask = pd.Series(True, index=compatible.index)
        if sub_kat != "(Semua)": mask &= compatible["sub_kategori"].str.strip() == sub_kat
        if brand_choice != "(Semua)": mask &= compatible["brand"].str.strip() == brand_choice
        mask &= compatible["rating"].fillna(0) >= min_rating
        if 'harga_idr' in compatible.columns and not compatible['harga_idr'].empty:
            mask &= compatible["harga_idr"].fillna(-1).between(price_range[0], price_range[1])
        
        filtered = compatible[mask]

        if filtered.empty:
            st.warning("Tidak ada produk yang cocok dengan filter Anda. Coba longgarkan kriteria.")
        else:
            idxs = filtered.index.tolist()
            ranked = recomm.rank_on_subset(idxs, top_n=top_n)
            
            if not ranked:
                st.warning("Tidak ada ranking yang bisa dihitung.")
            else:
                st.subheader(f"Top {len(ranked)} Rekomendasi Produk Untuk Anda")
                cols = st.columns(4)
                for i, (idx, score, prod) in enumerate(ranked):
                    with cols[i % 4]:
                        image_url = prod.get("url_gambar", "https://via.placeholder.com/250?text=No+Image")
                        jenis_kulit = prod.get('jenis_kulit_kompatibel', '')
                        deskripsi = prod.get('deskripsi', '')
                        manfaat = prod.get('manfaat', '')
                        sub_kategori = prod.get('sub_kategori', '')
                        
                        card_html = f"""
                        <div class="product-card">
                            <img src="{image_url}" alt="{prod.get('nama_produk', '')}">
                            <div class="brand">{prod.get('brand', '')}</div>
                            <h3>{prod.get('nama_produk', 'N/A')}</h3>
                            <div class="subcategory">{sub_kategori}</div>
                            <div class="price">Rp {prod.get('harga_idr', 0):,.0f}</div>
                            <div class="details">⭐ {prod.get('rating', 0):.1f} | {prod.get('size_ml', 0):.0f} ml</div>
                            <div class="skin-type"><b>Cocok untuk:</b> {jenis_kulit}</div>
                            <div class="manfaat"><b>Manfaat:</b> {manfaat}</div>
                            <div class="deskripsi"><b>Deskripsi:</b> {deskripsi}</div>
                            <div class="score">Skor: {score:.4f}</div>
                        </div>
                        """
                        st.markdown(card_html, unsafe_allow_html=True)
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat atau memproses data: {e}")
    st.info("Pastikan file CSV yang diunggah valid atau muat ulang halaman untuk menggunakan data default.")