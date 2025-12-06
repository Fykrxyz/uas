import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm, f, t
from scipy import stats
import base64

# --- 1. Konfigurasi Halaman (Harus di paling atas) ---
st.set_page_config(
    page_title="Proyek Akhir Pemkom 2025 - Uji Statistik",
    page_icon="📊",
    layout="wide"
)

# ==============================================================================
# BAGIAN STYLING & BACKGROUND (CSS)
# ==============================================================================

# --- Konfigurasi Nama File Gambar ---
file_background_utama = "uaspemkom.png"    
file_background_sidebar = "uaspemkomlagi.png" 

# --- Fungsi Konversi Gambar ke Base64 ---
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return None

# --- Memuat Gambar ---
img_main_b64 = get_base64_of_bin_file(file_background_utama)
img_sidebar_b64 = get_base64_of_bin_file(file_background_sidebar)

# --- Menyusun CSS Lengkap ---
# Kita menggabungkan CSS tombol lebar dari kode lama dengan styling warna dari kode baru
styles = f"""
<style>
/* =========================================
   A. BACKGROUND & TEKS UTAMA (MAIN AREA)
   ========================================= */
[data-testid="stAppViewContainer"] {{
    background-image: url("data:image/png;base64,{img_main_b64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: black !important; /* Teks utama tetap hitam */
}}

/* Header di Main Area juga hitam */
[data-testid="stAppViewContainer"] h1, 
[data-testid="stAppViewContainer"] h2, 
[data-testid="stAppViewContainer"] h3 {{
    color: black !important;
}}

/* =========================================
   B. BACKGROUND & TEKS SIDEBAR
   ========================================= */
[data-testid="stSidebar"] {{
    background-image: url("data:image/png;base64,{img_sidebar_b64}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}}

/* 1. Warna Teks Biasa di Sidebar (Cream/Putih) */
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {{
    color: #EAF4C5 !important; 
}}

/* 2. KHUSUS JUDUL/HEADER DI SIDEBAR JADI PUTIH BERSIH */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{
    color: white !important;       
    text-shadow: none !important;  
}}

/* =========================================
   C. KUSTOMISASI TOMBOL (BUTTON)
   ========================================= */
/* Target semua tombol (termasuk yang lebar) */
div.stButton > button {{
    width: 100%;                       /* Agar tombol memenuhi lebar kolom (dari kode lama) */
    height: 3em;                       /* Tinggi tombol (dari kode lama) */
    background-color: white !important; /* Background Putih */
    color: black !important;           /* Teks Hitam */
    border: 1px solid #dcdcdc;         
    border-radius: 8px;                
    font-weight: bold;                 
}}

div.stButton > button:hover {{
    background-color: #f0f0f0 !important; 
    border-color: black !important;
    color: black !important;
}}

/* Paksa teks tombol di sidebar tetap HITAM (override warna cream sidebar) */
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stButton > button * {{
    color: black !important;
}}

/* Mengatur warna input text area agar kontras */
.stTextArea textarea {{
    background-color: rgba(255, 255, 255, 0.9);
    color: black;
}}
.stNumberInput input {{
    background-color: rgba(255, 255, 255, 0.9);
    color: black;
}}
</style>
"""

# Menyuntikkan CSS
st.markdown(styles, unsafe_allow_html=True)


# ==============================================================================
# LOGIKA APLIKASI & SESSION STATE
# ==============================================================================

# --- Inisialisasi Session State (Agar data tidak hilang saat pindah tab) ---
# State untuk Uji Proporsi
if 'hasil_1_sampel' not in st.session_state:
    st.session_state['hasil_1_sampel'] = None
if 'hasil_2_sampel' not in st.session_state:
    st.session_state['hasil_2_sampel'] = None

# State untuk Uji F
if "Fh" not in st.session_state:
    st.session_state["Fh"] = 0
if "v1" not in st.session_state:
    st.session_state["v1"] = 0
if "v2" not in st.session_state:
    st.session_state["v2"] = 0
if "Jenis_uji" not in st.session_state:
    st.session_state["Jenis_uji"] = "Two Tail"
if "alpha_f" not in st.session_state:
    st.session_state["alpha_f"] = 0.05

# State untuk Pooled t-test
if 'p_value_pooled' not in st.session_state:
    st.session_state.p_value_pooled = None
if 't_stat_pooled' not in st.session_state:
    st.session_state.t_stat_pooled = 0
if 't_crit_pooled' not in st.session_state:
    st.session_state.t_crit_pooled = 0
if 'df_pooled' not in st.session_state:
    st.session_state.df_pooled = 0
if 'mean1_pooled' not in st.session_state:
    st.session_state.mean1_pooled = 0
if 'mean2_pooled' not in st.session_state:
    st.session_state.mean2_pooled = 0
if 'sd1_pooled' not in st.session_state:
    st.session_state.sd1_pooled = 0
if 'sd2_pooled' not in st.session_state:
    st.session_state.sd2_pooled = 0

# State untuk Welch t-test
if 'res_welch' not in st.session_state:
    st.session_state.res_welch = {"valid": False}
if 'alpha_welch' not in st.session_state:
    st.session_state.alpha_welch = 0.05

# State untuk Paired t-test
if 'p_value_paired' not in st.session_state:
    st.session_state.p_value_paired = None
if 't_stat_paired' not in st.session_state:
    st.session_state.t_stat_paired = 0
if 't_crit_paired' not in st.session_state:
    st.session_state.t_crit_paired = 0
if 'df_paired' not in st.session_state:
    st.session_state.df_paired = 0
if 'mean_d_paired' not in st.session_state:
    st.session_state.mean_d_paired = 0
if 'sd_d_paired' not in st.session_state:
    st.session_state.sd_d_paired = 0


# --- Fungsi Bantuan (Helper Functions) ---
def hitung_p_value_z(z_score, arah):
    if arah == 'two-sided':
        return 2 * (1 - norm.cdf(abs(z_score)))
    elif arah == 'smaller':
        return norm.cdf(z_score)
    elif arah == 'larger':
        return 1 - norm.cdf(z_score)

def tampilkan_kesimpulan_akhir(p_val, alpha, jenis_h0="Hipotesis Nol"):
    st.markdown("---")
    st.subheader("🏁 Kesimpulan Uji Hipotesis")
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Alpha (Taraf Signifikansi)", f"{alpha}")
        with col2:
            is_signif = p_val < alpha
            status = "Tolak H0" if is_signif else "Gagal Tolak H0"
            
            if is_signif:
                st.error(f"Keputusan: {status}")
            else:
                st.success(f"Keputusan: {status}")
    
    penjelasan = (
        f"Karena nilai **P-Value ({p_val:.4f}) < Alpha ({alpha})**, maka kita memiliki cukup bukti statistik untuk **Menolak {jenis_h0}**."
        if is_signif else
        f"Karena nilai **P-Value ({p_val:.4f}) >= Alpha ({alpha})**, maka kita **Tidak Cukup Bukti** untuk menolak {jenis_h0}."
    )
    st.info(penjelasan)

# ==============================================================================
# SIDEBAR (NAVIGASI)
# ==============================================================================
with st.sidebar:
    try:
        # Menampilkan Logo (Opsional, jika link mati tidak akan error fatal)
        st.image("https://upload.wikimedia.org/wikipedia/id/thumb/a/a2/Logo_Unpad.svg/1200px-Logo_Unpad.svg.png", width=100)
    except:
        st.markdown("### UNPAD")
        
    st.title("Proyek Pemkom 2025")
    st.write("**Prodi Statistika FMIPA Unpad**")
    st.markdown("---")
    
    st.write("Silakan pilih menu:")
    menu = st.selectbox(
        "Navigasi Halaman",
        [
            "Halaman Utama", 
            "Uji Proporsi (1 & 2 Sampel)", 
            "Uji Kesamaan Varians (F-Test)",
            "Uji Rata-rata 2 Sampel Independen (Pooled t)",
            "Uji Rata-rata 2 Sampel Independen (Welch t)",
            "Uji Rata-rata 2 Sampel Dependen (Paired t-test)"
        ]
    )
    
    st.markdown("---")
    st.markdown("### Anggota Kelompok:")
    st.text("1. Nama Anggota 1\n2. Nama Anggota 2\n3. Nama Anggota 3\n4. Nama Anggota 4\n5. Nama Anggota 5\n6. Nama Anggota 6\n7. Nama Anggota 7")


# ==============================================================================
# HALAMAN UTAMA
# ==============================================================================
if menu == "Halaman Utama":
    st.title("Alur Pemilihan Uji Statistik")
    st.write("Berikut adalah Flowchart yang digunakan sebagai acuan dalam aplikasi ini:")

    # --- CARA MENAMPILKAN GAMBAR FLOWCHART ---
    try:
        # Pastikan nama file gambarnya benar (huruf besar/kecil berpengaruh)
        st.image("flowchart.png", caption="Diagram Alur Pengerjaan", use_container_width=True)
        
    except Exception as e:
        st.error("Gambar flowchart.png tidak ditemukan. Pastikan sudah di-upload.")
# ==============================================================================
# UJI PROPORSI (1 & 2 SAMPEL)
# ==============================================================================
elif menu == "Uji Proporsi (1 & 2 Sampel)":
    
    # Sub-menu selection using tabs or radio inside the page
    st.title("📊 Uji Proporsi")
    tipe_uji = st.radio("Pilih Tipe Uji:", ["1 Sampel", "2 Sampel"], horizontal=True)
    
    tab_penjelasan, tab_rumus, tab_contoh, tab_kalkulator = st.tabs(
        ["📘 Penjelasan", "vi Rumus", "💡 Contoh", "🧮 Kalkulator"]
    )

    # --- TAB 1: PENJELASAN ---
    with tab_penjelasan:
        st.header(f"Konsep Dasar Uji Proporsi {tipe_uji}")
        if tipe_uji == "1 Sampel":
            st.write("""
            **Uji Proporsi Satu Sampel** digunakan untuk menguji hipotesis mengenai proporsi populasi ($p$)
            berdasarkan data sampel tunggal. Uji ini membandingkan proporsi sampel ($\hat{p}$) dengan 
            nilai acuan tertentu ($p_0$).
            """)
            st.markdown("**Hipotesis:**")
            st.latex(r"H_0: p = p_0")
            st.latex(r"H_1: p \neq p_0 \text{ (Dua sisi), atau } p < p_0, p > p_0")
        else:
            st.write("""
            **Uji Proporsi Dua Sampel Independen** digunakan untuk mengetahui apakah terdapat perbedaan 
            signifikan antara proporsi dua populasi yang berbeda.
            """)
            st.markdown("**Hipotesis:**")
            st.latex(r"H_0: p_1 = p_2 \text{ (Proporsi sama)}")
            st.latex(r"H_1: p_1 \neq p_2 \text{ (Berbeda), atau } p_1 > p_2, p_1 < p_2")

    # --- TAB 2: RUMUS ---
    with tab_rumus:
        st.header("Rumus Statistik Uji")
        if tipe_uji == "1 Sampel":
            st.info("Rumus Z-Test 1 Sampel")
            st.latex(r"Z_{hitung} = \frac{\hat{p} - p_0}{\sqrt{\frac{p_0(1-p_0)}{n}}}")
            st.markdown("""
            **Keterangan Variabel:**
            * $\hat{p} = x/n$ : Proporsi Sampel
            * $p_0$ : Proporsi Hipotesis (Nilai target)
            * $n$ : Jumlah Sampel
            """)
        else:
            st.info("Rumus Z-Test 2 Sampel (Pooled Variance)")
            st.latex(r"Z_{hitung} = \frac{\hat{p}_1 - \hat{p}_2}{\sqrt{\hat{p}_{pool}(1-\hat{p}_{pool}) \left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}")
            st.latex(r"\hat{p}_{pool} = \frac{x_1 + x_2}{n_1 + n_2}")
            st.markdown("""
            **Keterangan Variabel:**
            * $\hat{p}_1, \hat{p}_2$ : Proporsi sampel grup 1 & 2
            * $\hat{p}_{pool}$ : Proporsi gabungan
            * $n_1, n_2$ : Ukuran sampel
            """)

    # --- TAB 3: CONTOH ---
    with tab_contoh:
        st.header("Contoh Perhitungan")
        if tipe_uji == "1 Sampel":
            st.success("""
            **Kasus:** Peneliti menduga kurang dari 50% ($p_0=0.5$) mahasiswa membawa laptop.
            Dari 100 mahasiswa ($n$), ada 45 orang ($x$) membawa laptop.
            * Hipotesis: $H_1: p < 0.5$
            """)
        else:
            st.success("""
            **Kasus:** Apakah proporsi kelulusan Pria ($x_1=30, n_1=100$) berbeda dengan Wanita ($x_2=40, n_2=100$)?
            * Hipotesis: $H_1: p_1 \\neq p_2$
            """)

    # --- TAB 4: KALKULATOR ---
    with tab_kalkulator:
        st.subheader("Input Data")
        col1, col2 = st.columns(2)

        if tipe_uji == "1 Sampel":
            with col1:
                x = st.number_input("Jumlah Sukses (x)", 0, key="x1")
                n = st.number_input("Total Sampel (n)", 1, value=100, key="n1")
            with col2:
                p0 = st.number_input("Hipotesis Awal ($p_0$)", 0.01, 0.99, 0.5, key="p01")
                alpha = st.selectbox("Taraf Signifikansi ($\\alpha$)", [0.01, 0.05, 0.10], index=1, key="alp1")
                arah = st.selectbox("Arah Hipotesis", ["Two-sided (≠)", "Smaller (<)", "Larger (>)"], key="ar1")

            if st.button("Hitung Statistik Uji", key="btn1"):
                if x > n:
                    st.error("Error: x tidak boleh lebih besar dari n")
                else:
                    phat = x/n
                    se = np.sqrt((p0 * (1 - p0)) / n)
                    z = (phat - p0) / se
                    arah_map = {"Two-sided (≠)": "two-sided", "Smaller (<)": "smaller", "Larger (>)": "larger"}
                    
                    st.session_state['hasil_1_sampel'] = {
                        'z': z, 'phat': phat, 'p0': p0,
                        'arah': arah_map[arah], 'alpha': alpha
                    }

        else: # 2 Sampel
            with col1:
                st.markdown("**Sampel 1**")
                x1 = st.number_input("Sukses 1 ($x_1$)", 0, value=30, key="x21")
                n1 = st.number_input("Total 1 ($n_1$)", 1, value=100, key="n21")
            with col2:
                st.markdown("**Sampel 2**")
                x2 = st.number_input("Sukses 2 ($x_2$)", 0, value=40, key="x22")
                n2 = st.number_input("Total 2 ($n_2$)", 1, value=100, key="n22")
                alpha = st.selectbox("Signifikansi ($\\alpha$)", [0.01, 0.05, 0.10], index=1, key="alp2")
                arah = st.selectbox("Arah Hipotesis", ["Two-sided (≠)", "Smaller (<)", "Larger (>)"], key="ar2")

            if st.button("Hitung Statistik Uji", key="btn2"):
                if x1 > n1 or x2 > n2:
                    st.error("Error: Jumlah sukses melebihi sampel")
                else:
                    p1_hat = x1/n1
                    p2_hat = x2/n2
                    p_pool = (x1 + x2) / (n1 + n2)
                    se = np.sqrt(p_pool * (1 - p_pool) * ((1/n1) + (1/n2)))
                    
                    if se == 0:
                        st.error("Standard Error = 0. Data identik sempurna.")
                    else:
                        z = (p1_hat - p2_hat) / se
                        if "Two-sided" in arah: arah_code = "two-sided"
                        elif "Smaller" in arah: arah_code = "smaller"
                        else: arah_code = "larger"
                        
                        st.session_state['hasil_2_sampel'] = {
                            'z': z, 'p1': p1_hat, 'p2': p2_hat,
                            'arah': arah_code, 'alpha': alpha
                        }

        # --- OUTPUT HASIL PROPORSI ---
        hasil = st.session_state[f'hasil_{"1" if tipe_uji == "1 Sampel" else "2"}_sampel']
        
        if hasil:
            st.markdown("---")
            st.markdown("### 📝 Hasil Statistik")
            
            c1, c2, c3 = st.columns(3)
            if tipe_uji == "1 Sampel":
                c1.metric("Proporsi Sampel", f"{hasil['phat']:.4f}")
                c2.metric("Proporsi Target", f"{hasil['p0']:.4f}")
            else:
                c1.metric("Proporsi 1", f"{hasil['p1']:.4f}")
                c2.metric("Proporsi 2", f"{hasil['p2']:.4f}")
            
            c3.metric("Z-Score Hitung", f"{hasil['z']:.4f}")
            
            st.write("") 
            
            if st.button("Lihat Keputusan Uji (P-Value)", key="keputusan_proporsi"):
                p_val = hitung_p_value_z(hasil['z'], hasil['arah'])
                tampilkan_kesimpulan_akhir(p_val, hasil['alpha'], "Hipotesis Nol (H0)")


# ==============================================================================
# UJI KESAMAAN VARIANS (F-TEST)
# ==============================================================================
elif menu == "Uji Kesamaan Varians (F-Test)":
    st.title("Uji Kesamaan Varians (F-test)")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Konsep", "Hipotesis", "Rumus", "Contoh Perhitungan Singkat", "Hasil Uji", "Kriteria Uji"])
    
    with tab1:
        st.subheader("Penjelasan Konsep")
        st.write("""
        Uji F untuk kesamaan varians digunakan untuk menentukan apakah dua populasi memiliki
        varians yang sama atau berbeda secara signifikan. Uji ini membandingkan dua varians
        sampel untuk menilai apakah perbedaan yang terlihat hanya terjadi karena variasi
        acak atau karena varians populasi memang berbeda. Uji F umumnya digunakan ketika
        data berasal dari dua kelompok dan diasumsikan berdistribusi normal.
        """)
        
    with tab2:
        st.header("Hipotesis Uji")
        st.subheader("Two-tail Test")
        st.latex(r"H_0: \sigma_1^2 = \sigma_2^2 \quad")
        st.latex(r"H_1: \sigma_1^2 \neq \sigma_2^2 \quad")
        st.subheader("Lower-tail Test")
        st.latex(r"H_0: \sigma_1^2 \geq \sigma_2^2 \quad")
        st.latex(r"H_1: \sigma_1^2 < \sigma_2^2 \quad")
        st.subheader("Upper-tail Test")
        st.latex(r"H_0: \sigma_1^2 \leq \sigma_2^2 \quad")
        st.latex(r"H_1: \sigma_1^2 > \sigma_2^2 \quad")

    with tab3:
        st.header("Rumus")
        st.subheader("1. Statistik Uji F:")
        st.latex(r"F = \frac{S_1^2}{S_2^2}")
        st.markdown("""
        **Keterangan:**
        * $S_1^2$ = Varians sampel 1
        * $S_2^2$ = Varians sampel 2
        """)
        st.subheader("2. Derajat bebas:")
        st.latex(r"v_1 = n_1 - 1")
        st.latex(r"v_2 = n_2 - 1")

    with tab4:
        st.header("Contoh Perhitungan Singkat")
        st.write("""
        Misalkan diberikan dua kelompok data:
        X1 = [17, 14, 19, 16, 16]
        X2 = [15, 13, 16, 14, 11]
        """)
        st.write("""
        Hasil perhitungan diperoleh:
        * Varians X1 = 3.30
        * Varians X2 = 3.70
        * F-hitung = 3.70 / 3.30 = 1.1212
        * v1 = 4, v2 = 4
        """)

    with tab5:
        st.header("Masukkan Data")
        st.write("Masukkan data X1 dan X2 (dipisahkan koma).")
        x1_input = st.text_area("Data X1 (misal: 10,12,9,15,11)", key="f_x1")
        x2_input = st.text_area("Data X2 (misal: 8,11,7,14,10)", key="f_x2")
        alpha = st.number_input("Masukkan nilai alpha (α):", 0.01, 0.10, 0.05, key="f_alpha")
        jenis_uji = st.selectbox("Pilih Jenis Uji:",["Two Tail", "Upper Tail", "Lower Tail"], key="f_jenis")

        if st.button("Hitung Uji F"):
            try:
                x1 = np.array(list(map(float, x1_input.split(","))))
                x2 = np.array(list(map(float, x2_input.split(","))))
                
                if len(x1) < 2 or len(x2) < 2:
                    st.error("Jumlah data minimal masing-masing 2!")
                else:
                    s1 = np.var(x1, ddof=1)
                    s2 = np.var(x2, ddof=1)
                    
                    Fh = s2 / s1
                    v1 = len(x2) - 1
                    v2 = len(x1) - 1
                    
                    st.session_state["Jenis_uji"] = jenis_uji
                    st.session_state["Fh"] = Fh
                    st.session_state["v1"] = v1
                    st.session_state["v2"] = v2
                    st.session_state["alpha_f"] = alpha
                    
                    st.subheader("Hasil Perhitungan")
                    st.write(f"F-hitung: {Fh:.2f}")
                    st.write(f"Derajat Bebas 1: {v1}")
                    st.write(f"Derajat Bebas 2: {v2}")
            except:
                st.error("Format data tidak valid! Pastikan hanya angka dan koma.")

    with tab6:
        if st.session_state["v1"] > 0: # Cek jika sudah dihitung
            Jenis_uji = st.session_state["Jenis_uji"]
            Fh = st.session_state["Fh"]
            v1 = st.session_state["v1"]
            v2 = st.session_state["v2"]
            alpha = st.session_state["alpha_f"]
            
            if Jenis_uji == "Upper Tail":
                p_value = 1 - f.cdf(Fh, v1, v2)
                Fk_upper = f.ppf(1 - alpha, v1, v2)
                krit = Fh > Fk_upper
                Fk = Fk_upper
                krit_latex = f"{{F_hitung}} > F_{{1-\\alpha, v_1, v_2}} \\quad atau \\quad p\\text{{-value}} < \\alpha"
            elif Jenis_uji == "Lower Tail":
                p_value = f.cdf(Fh, v1, v2)
                Fk_lower = f.ppf(alpha, v1, v2)
                krit = Fh < Fk_lower
                Fk = Fk_lower
                krit_latex = f"{{F_hitung}} < F_{{\\alpha, v_1, v_2}} \\quad atau \\quad p\\text{{-value}} < \\alpha"
            else:  # TWO TAIL
                p_value = 2 * min(f.cdf(Fh, v1, v2), 1 - f.cdf(Fh, v1, v2))
                Fk_upper = f.ppf(1 - alpha / 2, v1, v2)
                Fk_lower = f.ppf(alpha / 2, v1, v2)
                krit = Fh < Fk_lower or Fh > Fk_upper
                krit_latex = f"F_{{hitung}} < F_{{\\alpha/2, v_1, v_2}} \\quad atau F_{{hitung}} > F_{{1-\\alpha/2, v_1, v_2}} \\quad atau \\quad p\\text{{-value}} < \\alpha"

            st.write("### Hasil Perhitungan")
            if Jenis_uji == "Two Tail":
                colA, colB, colC, colD, colE = st.columns(5)
                colA.metric("Jenis Uji", Jenis_uji)
                colB.metric("F-hitung", f"{Fh:.2f}")
                colC.metric("F Upper", f"{Fk_upper:.2f}")
                colD.metric("F Lower", f"{Fk_lower:.2f}")
                colE.metric("p-value", f"{p_value:.2f}")
                st.metric("Alpha", f"{alpha:.2f}")
            else:
                colA, colB, colC, colD, colE = st.columns(5)
                colA.metric("Jenis Uji", Jenis_uji)
                colB.metric("F-hitung", f"{Fh:.2f}")
                colC.metric("F-kritis", f"{Fk:.2f}")
                colD.metric("p-value", f"{p_value:.2f}")
                colE.metric("Alpha", f"{alpha:.2f}")

            st.write("### Kriteria Uji")
            st.write("Tolak H₀ jika:")
            st.latex(krit_latex)
            
            if krit or p_value < alpha:
                keputusan = "Tolak H₀"
                st.error(f"Keputusan: {keputusan}")
                write = "terdapat perbedaan varians yang signifikan antara dua sampel."
            else:
                keputusan = "Gagal Tolak H₀"
                st.success(f"Keputusan: {keputusan}")
                write = "tidak terdapat perbedaan varians yang signifikan antara dua sampel."
            
            st.write("### Kesimpulan")
            st.info(f"Pada taraf signifikansi α = {alpha}, diperoleh keputusan **{keputusan}**, dengan demikian {write}")
        else:
            st.warning("Silakan lakukan perhitungan terlebih dahulu di Tab Hasil Uji.")


# ==============================================================================
# POOLED T-TEST
# ==============================================================================
elif menu == "Uji Rata-rata 2 Sampel Independen (Pooled t)":
    st.title("Uji Rata-rata 2 Sampel Independen (Pooled t-test)")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Konsep", "Hipotesis", "Rumus", "Contoh", "Input Data & Hitung", "Hasil & Kesimpulan"
    ])

    with tab1:
        st.header("Konsep Uji Statistik")
        st.write("""
        Uji Pooled t-test digunakan ketika **dua sampel independen** dibandingkan 
        dan diasumsikan memiliki **variansi yang sama**.
        """)

    with tab2:
        st.header("Hipotesis")
        st.latex(r"H_0 : \mu_1 = \mu_2")
        st.latex(r"H_1 : \mu_1 \neq \mu_2")
        st.write("• μ1 = rata-rata populasi 1, μ2 = rata-rata populasi 2")

    with tab3:
        st.header("Rumus Pooled t-test")
        st.write("1. Pooled Variance (Sp²):")
        st.latex(r"S_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}")
        st.write("2. Statistik Uji t:")
        st.latex(r"t = \frac{\bar{X}_1 - \bar{X}_2}{S_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}}")
        st.write("3. Derajat Bebas (df):")
        st.latex(r"df = n_1 + n_2 - 2")

    with tab4:
        st.header("Contoh Perhitungan Singkat")
        st.write("""
        Sampel 1: X1 = [12, 15, 14, 16, 13] (Mean=14)
        Sampel 2: X2 = [10, 14, 11, 13, 12] (Mean=12)
        Pooled Variance Sp² ≈ 2.5
        t ≈ 3.16
        df = 8
        """)

    with tab5:
        st.header("Input Data & Hitung")
        st.write("Masukkan data sampel 1 dan sampel 2 (dipisahkan koma).")
        x1_input = st.text_area("Data Sampel 1 (misal: 12,15,14,16,13)", key="pool_x1")
        x2_input = st.text_area("Data Sampel 2 (misal: 10,14,11,13,12)", key="pool_x2")
        alpha_pooled = st.number_input("Taraf Signifikansi (α):", 0.01, 0.10, 0.05, step=0.01, key="pool_alpha")

        if st.button("Hitung Pooled t-test"):
            try:
                x1_arr = np.array([float(i) for i in x1_input.split(",")])
                x2_arr = np.array([float(i) for i in x2_input.split(",")])
                
                n1 = len(x1_arr)
                n2 = len(x2_arr)
                mean1 = np.mean(x1_arr)
                mean2 = np.mean(x2_arr)
                sd1 = np.std(x1_arr, ddof=1)
                sd2 = np.std(x2_arr, ddof=1)
                
                # Pooled variance
                Sp2 = ((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1+n2-2)
                Sp = np.sqrt(Sp2)
                
                # t-stat
                t_stat = (mean1 - mean2) / (Sp * np.sqrt(1/n1 + 1/n2))
                df = n1 + n2 - 2
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
                t_crit = stats.t.ppf(1 - alpha_pooled/2, df)
                
                # Simpan session state
                st.session_state.t_stat_pooled = t_stat
                st.session_state.t_crit_pooled = t_crit
                st.session_state.df_pooled = df
                st.session_state.mean1_pooled = mean1
                st.session_state.mean2_pooled = mean2
                st.session_state.sd1_pooled = sd1
                st.session_state.sd2_pooled = sd2
                st.session_state.p_value_pooled = p_val
                st.session_state.alpha_pooled_saved = alpha_pooled
                
                st.success("Perhitungan selesai! Silakan buka tab 'Hasil & Kesimpulan'.")
            except:
                st.error("Format data salah. Pastikan hanya angka dan koma.")

    with tab6:
        st.header("Hasil & Kesimpulan")
        if st.session_state.p_value_pooled is not None:
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Mean Sampel 1", f"{st.session_state.mean1_pooled:.4f}")
            colB.metric("Mean Sampel 2", f"{st.session_state.mean2_pooled:.4f}")
            colC.metric("t-hitung", f"{st.session_state.t_stat_pooled:.4f}")
            colD.metric("p-value", f"{st.session_state.p_value_pooled:.6f}")
            
            st.write(f"df = {st.session_state.df_pooled}")
            st.write(f"SD Sampel 1 ≈ {st.session_state.sd1_pooled:.4f}, SD Sampel 2 ≈ {st.session_state.sd2_pooled:.4f}")
            st.write(f"t-kritis = {st.session_state.t_crit_pooled:.4f}")
            st.markdown("---")
            
            alpha_used = st.session_state.get('alpha_pooled_saved', 0.05)
            if st.session_state.p_value_pooled < alpha_used:
                st.error(f"**Tolak H0** (p-value {st.session_state.p_value_pooled:.6f} < {alpha_used})")
                st.write("Kesimpulan: Terdapat perbedaan signifikan antara kedua sampel.")
            else:
                st.success(f"**Gagal Tolak H0** (p-value {st.session_state.p_value_pooled:.6f} > {alpha_used})")
                st.write("Kesimpulan: Tidak cukup bukti untuk menyatakan perbedaan signifikan.")
        else:
            st.info("Belum ada perhitungan. Masukkan data di tab 'Input Data & Hitung'.")


# ==============================================================================
# WELCH T-TEST
# ==============================================================================
elif menu == "Uji Rata-rata 2 Sampel Independen (Welch t)":
    st.title("Uji Rata-rata Dua Sampel Independen (Welch t-test)")
    st.write("Digunakan untuk membandingkan rata-rata dua populasi yang independen dengan **asumsi varians TIDAK sama**.")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Konsep", "Hipotesis", "Rumus", "Contoh", "Statistik Uji", "Keputusan"
    ])

    with tab1:
        st.header("Penjelasan Konsep")
        st.write("Perbedaan utamanya terletak pada perhitungan **Derajat Bebas (df)** yang menggunakan pendekatan **Welch-Satterthwaite**.")

    with tab2:
        st.header("Hipotesis Uji")
        st.latex(r"H_0 : \mu_1 = \mu_2")
        st.latex(r"H_1 : \mu_1 \neq \mu_2")

    with tab3:
        st.header("Rumus Statistik Uji")
        st.markdown("**1. Hitung Statistik Uji $t'$:**")
        st.latex(r"t' = \frac{(\bar{x}_1 - \bar{x}_2)}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}")
        st.markdown("**2. Hitung Derajat Bebas (df) - Rumus Welch-Satterthwaite:**")
        st.latex(r"df = \frac{\left( \frac{s_1^2}{n_1} + \frac{s_2^2}{n_2} \right)^2}{ \frac{\left( \frac{s_1^2}{n_1} \right)^2}{n_1 - 1} + \frac{\left( \frac{s_2^2}{n_2} \right)^2}{n_2 - 1} }")

    with tab4:
        st.header("Contoh Perhitungan")
        st.write("Kelas A (n=5): [80, 85, 90, 75, 70]. Varians = 62.5")
        st.write("Kelas B (n=6): [60, 62, 58, 65, 61, 60]. Varians = 5.6")
        st.write("Karena varians jauh berbeda, gunakan Welch Test. Hasil: p-value = 0.0045 (Tolak H0).")

    with tab5:
        st.header("Masukkan Data Sampel")
        col1, col2 = st.columns(2)
        with col1:
            x1_input = st.text_area("Data Sampel 1 ($X_1$)", key="welch_x1")
        with col2:
            x2_input = st.text_area("Data Sampel 2 ($X_2$)", key="welch_x2")
        
        alpha_welch = st.number_input("Taraf Signifikansi (alpha):", 0.01, 0.20, 0.05, step=0.01, key="alpha_welch_input")

        if st.button("Hitung Welch t-test"):
            try:
                data1 = np.array([float(x) for x in x1_input.split(",")])
                data2 = np.array([float(x) for x in x2_input.split(",")])
                
                n1, n2 = len(data1), len(data2)
                mean1, mean2 = np.mean(data1), np.mean(data2)
                var1, var2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
                
                if n1 < 2 or n2 < 2:
                    st.error("Setiap sampel minimal harus memiliki 2 data!")
                else:
                    numerator_t = mean1 - mean2
                    se_sq = (var1 / n1) + (var2 / n2)
                    se = np.sqrt(se_sq)
                    t_stat = numerator_t / se
                    
                    df_num = se_sq**2
                    term1 = ((var1 / n1)**2) / (n1 - 1)
                    term2 = ((var2 / n2)**2) / (n2 - 1)
                    df_den = term1 + term2
                    df_welch = df_num / df_den
                    
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df_welch))
                    t_crit = stats.t.ppf(1 - alpha_welch/2, df_welch)
                    
                    st.session_state.res_welch = {
                        "mean1": mean1, "mean2": mean2, "var1": var1, "var2": var2,
                        "n1": n1, "n2": n2, "t_stat": t_stat, "df": df_welch,
                        "p_val": p_val, "t_crit": t_crit, "valid": True
                    }
                    st.session_state.alpha_welch = alpha_welch
                    st.success("Perhitungan selesai! Silakan cek tab 'Keputusan'.")
            except ValueError:
                st.error("Format data salah. Pastikan hanya memasukkan angka dipisah koma.")

    with tab6:
        st.header("Hasil Analisis")
        if st.session_state.res_welch["valid"]:
            res = st.session_state.res_welch
            c1, c2, c3 = st.columns(3)
            c1.metric("t-hitung ($t'$)", f"{res['t_stat']:.4f}")
            c2.metric("Derajat Bebas (df)", f"{res['df']:.4f}")
            c3.metric("p-value", f"{res['p_val']:.4f}")
            
            st.markdown("---")
            st.subheader("Detail Statistik")
            summary_data = {
                "Sampel": ["Kelompok 1", "Kelompok 2"],
                "Jumlah (n)": [res['n1'], res['n2']],
                "Rata-rata": [res['mean1'], res['mean2']],
                "Varians ($s^2$)": [res['var1'], res['var2']]
            }
            st.table(pd.DataFrame(summary_data))
            
            st.markdown("---")
            st.subheader("Keputusan Uji")
            alpha_val = st.session_state.alpha_welch
            
            if res['p_val'] < alpha_val:
                st.error(f"**Keputusan: TOLAK H0**")
                st.write(f"Karena p-value ({res['p_val']:.4f}) < Alpha ({alpha_val}).")
                st.write("**Kesimpulan:** Ada perbedaan rata-rata yang signifikan antara kedua kelompok.")
            else:
                st.success(f"**Keputusan: GAGAL TOLAK H0**")
                st.write(f"Karena p-value ({res['p_val']:.4f}) > Alpha ({alpha_val}).")
                st.write("**Kesimpulan:** Tidak cukup bukti untuk menyatakan perbedaan rata-rata.")
        else:
            st.info("Belum ada data yang dihitung.")


# ==============================================================================
# PAIRED T-TEST
# ==============================================================================
elif menu == "Uji Rata-rata 2 Sampel Dependen (Paired t-test)":
    st.title("Uji Rata-rata Dua Sampel Dependen (Paired t-test)")
    st.write("Digunakan ketika dua data saling berpasangan (paired).")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Konsep", "Hipotesis", "Rumus", "Contoh", "Hitung", "Hasil"
    ])

    with tab1:
        st.header("Penjelasan")
        st.write("""
        Uji dua sampel dependen digunakan ketika **dua data yang dibandingkan berpasangan**,
        artinya setiap subjek diukur dua kali (sebelum–sesudah). 
        Yang diuji adalah **rata-rata selisih D = X₁ – X₂**.
        """)

    with tab2:
        st.header("Hipotesis Uji")
        st.latex(r"H_0 : \mu_D = 0")
        st.latex(r"H_1 : \mu_D \neq 0")

    with tab3:
        st.header("Rumus Paired t-test")
        st.write("1. Selisih tiap pasangan (d): $d_i = X_{1i} - X_{2i}$")
        st.write("2. Rata-rata selisih: $\\bar{X}_d = \\frac{\\sum d_i}{n}$")
        st.write("3. Simpangan baku selisih: $s_d = \\sqrt{\\frac{\\sum(d_i - \\bar{X}_d)^2}{n-1}}$")
        st.write("4. Statistik uji t: $t = \\frac{\\bar{X}_d - \\mu_0}{s_d / \\sqrt{n}}$")
        st.write("5. Derajat bebas: $df = n - 1$")

    with tab4:
        st.header("Contoh Perhitungan singkat")
        st.write("X1 = [10, 12, 9, 15, 11], X2 = [8, 11, 7, 14, 10]")
        st.write("D = [2, 1, 2, 1, 1]. Rata-rata selisih = 1.4. t = 5.72. Kesimpulan: Tolak H0.")

    with tab5:
        st.header("Masukkan Data")
        st.write("Masukkan data X1 dan X2 (dipisahkan koma).")
        x1_input = st.text_area("Data X1 (misal: 10,12,9,15,11)", key="paired_x1")
        x2_input = st.text_area("Data X2 (misal: 8,11,7,14,10)", key="paired_x2")
        alpha_paired = st.number_input("Taraf Signifikansi (α):", 0.01, 0.10, 0.05, step=0.01, key="alpha_paired_input")

        if st.button("Hitung Uji t Paired"):
            try:
                x1_arr = np.array([float(i) for i in x1_input.split(",")])
                x2_arr = np.array([float(i) for i in x2_input.split(",")])
                
                if len(x1_arr) != len(x2_arr):
                    st.error("Error: Jumlah data X1 dan X2 harus sama!")
                else:
                    D_arr = x1_arr - x2_arr
                    n_calc = len(D_arr)
                    df_calc = n_calc - 1
                    mean_D_calc = np.mean(D_arr)
                    sd_D_calc = np.std(D_arr, ddof=1)
                    
                    if sd_D_calc == 0:
                        st.warning("Standar deviasi selisih adalah 0, t-hitung tidak terdefinisi.")
                    else:
                        t_stat_calc = mean_D_calc / (sd_D_calc / np.sqrt(n_calc))
                        p_val_calc = 2 * (1 - stats.t.cdf(abs(t_stat_calc), df_calc))
                        t_crit_calc = stats.t.ppf(1 - alpha_paired/2, df_calc)
                        
                        st.session_state.t_stat_paired = t_stat_calc
                        st.session_state.t_crit_paired = t_crit_calc
                        st.session_state.df_paired = df_calc
                        st.session_state.mean_d_paired = mean_D_calc
                        st.session_state.sd_d_paired = sd_D_calc
                        st.session_state.p_value_paired = p_val_calc
                        st.session_state.alpha_paired = alpha_paired
                        
                        st.success("Perhitungan selesai! Silakan buka tab 'Hasil'.")
            except ValueError:
                st.error("Format data salah. Pastikan hanya angka dan koma.")

    with tab6:
        st.header("Hasil Uji")
        if st.session_state.df_paired > 0 and st.session_state.p_value_paired is not None:
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Rata-rata Selisih", f"{st.session_state.mean_d_paired:.4f}")
            colB.metric("t-hitung", f"{st.session_state.t_stat_paired:.4f}")
            colC.metric("t-kritis", f"{st.session_state.t_crit_paired:.4f}")
            colD.metric("p-value", f"{st.session_state.p_value_paired:.6f}")
            
            st.write(f"Degree of Freedom (df): **{st.session_state.df_paired}**")
            st.write(f"Standar Deviasi Selisih: **{st.session_state.sd_d_paired:.4f}**")
            st.markdown("---")
            
            st.subheader("Keputusan")
            alpha_val = st.session_state.get("alpha_paired", 0.05)
            
            if st.session_state.p_value_paired < alpha_val:
                st.error(f"**Tolak H0** (p-value {st.session_state.p_value_paired:.6f} < {alpha_val})")
                st.write("Kesimpulan: Terdapat perbedaan yang signifikan antara kedua kondisi.")
            else:
                st.success(f"**Gagal Tolak H0** (p-value {st.session_state.p_value_paired:.6f} > {alpha_val})")
                st.write("Kesimpulan: Tidak cukup bukti untuk menyatakan adanya perbedaan signifikan.")
        else:
            st.info("Belum ada data. Silakan masukkan data di tab 'Hitung' dan klik tombol Hitung.")
