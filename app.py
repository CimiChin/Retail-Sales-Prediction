import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Permintaan Produk Fashion",
    page_icon="👕",
    layout="wide"
)

# Fungsi untuk memuat dan cache data
@st.cache_data
def load_data():
    df = pd.read_csv('retail_sales_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    # Mengubah 'Quantity' menjadi kategori untuk klasifikasi
    # 1: Rendah, 2: Sedang, 3: Tinggi
    bins = [0, 1, 2, float('inf')]
    labels = ['Rendah (1)', 'Sedang (2)', 'Tinggi (3+)']
    df['Demand'] = pd.cut(df['Quantity'], bins=bins, labels=labels)
    df.dropna(subset=['Demand'], inplace=True)
    return df

df = load_data()

# Sidebar untuk Navigasi
st.sidebar.title("Navigasi 🧭")
page = st.sidebar.radio("Pilih Halaman:", ["Dataset & EDA", "Pelatihan Model", "Formulir Prediksi"])

# =====================================================================================
# Halaman 1: EDA (Exploratory Data Analysis)
# =====================================================================================
if page == "Dataset & EDA":
    st.title("📊 Analisis Data Eksploratif (EDA)")
    st.markdown("Halaman ini menampilkan analisis awal dari dataset penjualan ritel, dengan fokus pada produk fashion.")

    # Tampilkan data mentah
    if st.checkbox("Tampilkan Dataset Mentah"):
        st.write(df.head())

    # Karakteristik Dataset
    st.header("Karakteristik Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Transaksi", f"{df.shape[0]:,}")
        st.metric("Jumlah Pelanggan Unik", f"{df['Customer ID'].nunique():,}")
    with col2:
        st.metric("Jumlah Kategori Produk", f"{df['Product Category'].nunique()}")
        st.metric("Periode Data", f"{df['Date'].min().strftime('%d %B %Y')} - {df['Date'].max().strftime('%d %B %Y')}")

    st.subheader("Informasi dan Statistik Deskriptif")
    st.text("Informasi Tipe Data:")
    buffer = st.empty()
    # Storing the info in a string and then displaying it
    import io
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    buffer.text(s)
    
    st.text("Statistik Deskriptif untuk Fitur Numerik:")
    st.write(df.describe())

    # Visualisasi
    st.header("Visualisasi Data")
    
    # Filter khusus untuk produk 'Clothing'
    df_clothing = df[df['Product Category'] == 'Clothing']

    # Visualisasi 1: Distribusi Permintaan (Demand) untuk Kategori Pakaian
    fig1 = px.histogram(df_clothing, x='Demand', title='Distribusi Tingkat Permintaan untuk Produk Pakaian',
                        color='Demand', color_discrete_map={'Rendah (1)':'skyblue','Sedang (2)':'royalblue','Tinggi (3+)':'darkblue'})
    st.plotly_chart(fig1, use_container_width=True)

    # Visualisasi 2: Distribusi Umur dan Gender Pelanggan Pakaian
    fig2 = px.histogram(df_clothing, x='Age', color='Gender', title='Distribusi Umur dan Gender Pelanggan Pakaian',
                        barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

    # Visualisasi 3: Tren Penjualan Pakaian dari Waktu ke Waktu
    df_clothing_time = df_clothing.set_index('Date').resample('M')['Total Amount'].sum().reset_index()
    fig3 = px.line(df_clothing_time, x='Date', y='Total Amount', title='Tren Total Penjualan Pakaian Bulanan')
    st.plotly_chart(fig3, use_container_width=True)

# =====================================================================================
# Halaman 2: Pelatihan Model
# =====================================================================================
elif page == "Pelatihan Model":
    st.title("🤖 Hasil Pelatihan Model Machine Learning")
    st.markdown("Halaman ini menunjukkan hasil pelatihan model **K-Nearest Neighbors (KNN)** dan **Naive Bayes** untuk memprediksi tingkat permintaan produk.")

    # Persiapan Data
    features = ['Age', 'Gender', 'Product Category', 'Price per Unit']
    target = 'Demand'

    X = df[features]
    y = df[target]

    # Encoding Fitur Kategorikal
    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    X['Gender'] = le_gender.fit_transform(X['Gender'])
    X['Product Category'] = le_category.fit_transform(X['Product Category'])

    # Standardisasi Fitur Numerik
    scaler = StandardScaler()
    X[['Age', 'Price per Unit']] = scaler.fit_transform(X[['Age', 'Price per Unit']])

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # Pelatihan Model
    knn = KNeighborsClassifier(n_neighbors=5)
    nb = GaussianNB()

    knn.fit(X_train, y_train)
    nb.fit(X_train, y_train)

    # Prediksi
    y_pred_knn = knn.predict(X_test)
    y_pred_nb = nb.predict(X_test)

    # Evaluasi
    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_nb = confusion_matrix(y_test, y_pred_nb)

    # Tampilkan Hasil
    st.header("Hasil Evaluasi Model")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.metric("Akurasi", f"{acc_knn:.2%}")
        st.write("Confusion Matrix:")
        st.write(cm_knn)

    with col2:
        st.subheader("Naive Bayes")
        st.metric("Akurasi", f"{acc_nb:.2%}")
        st.write("Confusion Matrix:")
        st.write(cm_nb)

    st.info("""
    **Penjelasan:**
    - **Akurasi**: Persentase prediksi yang benar dari total data uji. Semakin tinggi, semakin baik.
    - **Confusion Matrix**: Tabel yang menunjukkan performa model. Diagonal utama (kiri atas ke kanan bawah) adalah jumlah prediksi yang benar untuk setiap kelas.
    - Model ini dilatih untuk mengklasifikasikan permintaan menjadi tiga kategori: 'Rendah (1)', 'Sedang (2)', dan 'Tinggi (3+)'.
    """)

# =====================================================================================
# Halaman 3: Formulir Prediksi
# =====================================================================================
elif page == "Formulir Prediksi":
    st.title("📝 Formulir Prediksi Permintaan")
    st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi tingkat permintaan produk menggunakan model yang telah dilatih.")

    # Siapkan data dan model (sama seperti halaman 2, untuk konsistensi)
    features = ['Age', 'Gender', 'Product Category', 'Price per Unit']
    target = 'Demand'
    X = df[features]
    y = df[target]

    le_gender = LabelEncoder().fit(X['Gender'])
    le_category = LabelEncoder().fit(X['Product Category'])
    X['Gender'] = le_gender.transform(X['Gender'])
    X['Product Category'] = le_category.transform(X['Product Category'])
    
    scaler = StandardScaler().fit(X[['Age', 'Price per Unit']])
    X[['Age', 'Price per Unit']] = scaler.transform(X[['Age', 'Price per Unit']])

    # Latih ulang model pada seluruh data untuk prediksi production
    knn_prod = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    nb_prod = GaussianNB().fit(X, y)

    # Buat Formulir
    with st.form("prediction_form"):
        st.header("Masukkan Detail Pelanggan dan Produk")
        
        # Input Fitur
        age = st.slider("Umur Pelanggan", 18, 70, 30)
        gender = st.selectbox("Jenis Kelamin", df['Gender'].unique())
        product_category = st.selectbox("Kategori Produk", df['Product Category'].unique(), index=df['Product Category'].unique().tolist().index('Clothing'))
        price_per_unit = st.number_input("Harga per Unit ($)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)

        # Tombol Submit
        submitted = st.form_submit_button("Prediksi Permintaan")

    if submitted:
        # Proses input
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Product Category': [product_category],
            'Price per Unit': [price_per_unit]
        })

        # Transformasi input
        input_data['Gender'] = le_gender.transform(input_data['Gender'])
        input_data['Product Category'] = le_category.transform(input_data['Product Category'])
        input_data[['Age', 'Price per Unit']] = scaler.transform(input_data[['Age', 'Price per Unit']])

        # Lakukan Prediksi
        pred_knn = knn_prod.predict(input_data)[0]
        pred_nb = nb_prod.predict(input_data)[0]

        # Tampilkan Hasil Prediksi
        st.subheader("Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Prediksi Model KNN")
            st.success(f"Tingkat Permintaan: **{pred_knn}**")
        with col2:
            st.write("#### Prediksi Model Naive Bayes")
            st.success(f"Tingkat Permintaan: **{pred_nb}**")
            
        st.warning("**Disclaimer**: Prediksi ini didasarkan pada data historis dan hanya untuk tujuan demonstrasi. Hasil sebenarnya dapat bervariasi.")
