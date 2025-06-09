import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import warnings
import io

warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Permintaan Produk Fashion",
    page_icon="👕",
    layout="wide"
)

# Fungsi untuk memuat dan cache data dengan kategori yang lebih beragam
@st.cache_data
def load_data():
    df = pd.read_csv('retail_sales_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # REVISI: Membuat kategori permintaan yang lebih beragam menggunakan Kuantil
    # Ini akan membagi data 'Quantity' menjadi 5 kelompok dengan jumlah anggota yang kurang lebih sama.
    try:
        quantiles = pd.qcut(df['Quantity'], q=5, labels=False, duplicates='drop')
        
        # Definisikan label yang lebih deskriptif
        labels = [
            'Permintaan Rendah', 
            'Permintaan Cukup Rendah', 
            'Permintaan Sedang', 
            'Permintaan Tinggi', 
            'Permintaan Sangat Tinggi'
        ]
        
        # Petakan label berdasarkan hasil kuantil
        # Pastikan jumlah label sesuai dengan jumlah kategori unik dari qcut
        unique_quantiles = quantiles.nunique()
        df['Demand'] = pd.qcut(df['Quantity'], q=unique_quantiles, labels=labels[:unique_quantiles])

    except ValueError:
        # Fallback jika qcut gagal (misal, tidak cukup variasi data)
        bins = [0, 1, 2, 3, 4, float('inf')]
        labels = ['Rendah (1)', 'Cukup Rendah (2)', 'Sedang (3)', 'Tinggi (4)', 'Sangat Tinggi (5+)']
        df['Demand'] = pd.cut(df['Quantity'], bins=bins, labels=labels, right=False)

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

    if st.checkbox("Tampilkan Dataset Mentah (dengan kategori permintaan baru)"):
        st.write(df.head())

    st.header("Karakteristik Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Transaksi", f"{df.shape[0]:,}")
        st.metric("Jumlah Pelanggan Unik", f"{df['Customer ID'].nunique():,}")
    with col2:
        st.metric("Jumlah Kategori Produk", f"{df['Product Category'].nunique()}")
        st.metric("Periode Data", f"{df['Date'].min().strftime('%d %B %Y')} - {df['Date'].max().strftime('%d %B %Y')}")

    st.subheader("Informasi dan Statistik Deskriptif")
    buf = io.StringIO()
    df.info(buf=buf)
    st.text(f"Informasi Tipe Data:\n{buf.getvalue()}")
    st.text("Statistik Deskriptif untuk Fitur Numerik:")
    st.write(df.describe())

    st.header("Visualisasi Data")
    
    df_clothing = df[df['Product Category'] == 'Clothing']

    # Visualisasi 1: Distribusi Permintaan (Demand) untuk Kategori Pakaian
    st.subheader("Distribusi Tingkat Permintaan untuk Produk Pakaian")
    fig1 = px.histogram(df_clothing, x='Demand', title='Distribusi Tingkat Permintaan Pakaian',
                        color='Demand',
                        category_orders={"Demand": df_clothing['Demand'].cat.categories.tolist()}) # Urutkan kategori
    st.plotly_chart(fig1, use_container_width=True)

    # Visualisasi 2: Distribusi Umur dan Gender Pelanggan Pakaian
    st.subheader("Distribusi Demografi Pelanggan Pakaian")
    fig2 = px.histogram(df_clothing, x='Age', color='Gender', title='Distribusi Umur dan Gender Pelanggan Pakaian',
                        barmode='group')
    st.plotly_chart(fig2, use_container_width=True)

    # Visualisasi 3: Tren Penjualan Pakaian dari Waktu ke Waktu
    st.subheader("Tren Penjualan Pakaian")
    df_clothing_time = df_clothing.set_index('Date').resample('M')['Total Amount'].sum().reset_index()
    fig3 = px.line(df_clothing_time, x='Date', y='Total Amount', title='Tren Total Penjualan Pakaian Bulanan', markers=True)
    st.plotly_chart(fig3, use_container_width=True)

# =====================================================================================
# Halaman 2: Pelatihan Model
# =====================================================================================
elif page == "Pelatihan Model":
    st.title("🤖 Hasil Pelatihan Model Machine Learning")
    st.markdown("Halaman ini menunjukkan hasil pelatihan model **K-Nearest Neighbors (KNN)** dan **Naive Bayes** untuk memprediksi tingkat permintaan produk.")

    features = ['Age', 'Gender', 'Product Category', 'Price per Unit']
    target = 'Demand'
    X = df[features]
    y = df[target]

    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    X['Gender'] = le_gender.fit_transform(X['Gender'])
    X['Product Category'] = le_category.fit_transform(X['Product Category'])

    scaler = StandardScaler()
    X[['Age', 'Price per Unit']] = scaler.fit_transform(X[['Age', 'Price per Unit']])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    knn = KNeighborsClassifier(n_neighbors=5)
    nb = GaussianNB()
    knn.fit(X_train, y_train)
    nb.fit(X_train, y_train)

    y_pred_knn = knn.predict(X_test)
    y_pred_nb = nb.predict(X_test)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_nb = accuracy_score(y_test, y_pred_nb)
    
    # Membuat confusion matrix lebih mudah dibaca dengan label
    labels = sorted(y.unique())
    cm_knn = pd.DataFrame(confusion_matrix(y_test, y_pred_knn, labels=labels), index=labels, columns=labels)
    cm_nb = pd.DataFrame(confusion_matrix(y_test, y_pred_nb, labels=labels), index=labels, columns=labels)


    st.header("Hasil Evaluasi Model")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.metric("Akurasi", f"{acc_knn:.2%}")
        st.write("Confusion Matrix:")
        st.dataframe(cm_knn)

    with col2:
        st.subheader("Naive Bayes")
        st.metric("Akurasi", f"{acc_nb:.2%}")
        st.write("Confusion Matrix:")
        st.dataframe(cm_nb)

    st.info(f"""
    **Penjelasan:**
    - **Akurasi**: Persentase prediksi yang benar.
    - **Confusion Matrix**: Menunjukkan performa model pada setiap kelas. Baris adalah label aktual, kolom adalah label prediksi.
    - Model ini dilatih untuk mengklasifikasikan permintaan menjadi **{len(labels)} kategori**: {', '.join(labels)}.
    """)

# =====================================================================================
# Halaman 3: Formulir Prediksi
# =====================================================================================
elif page == "Formulir Prediksi":
    st.title("📝 Formulir Prediksi Permintaan")
    st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi tingkat permintaan produk.")

    # Persiapan data dan model untuk prediksi
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

    knn_prod = KNeighborsClassifier(n_neighbors=5).fit(X, y)
    nb_prod = GaussianNB().fit(X, y)

    with st.form("prediction_form"):
        st.header("Masukkan Detail Pelanggan dan Produk")
        
        col1, col2 = st.columns(2)
        with col1:
            age = st.slider("Umur Pelanggan", 18, 70, 30)
            gender = st.selectbox("Jenis Kelamin", df['Gender'].unique(), index=0)
        with col2:
            product_category = st.selectbox("Kategori Produk", df['Product Category'].unique(), index=df['Product Category'].unique().tolist().index('Clothing'))
            price_per_unit = st.number_input("Harga per Unit ($)", min_value=5.0, max_value=500.0, value=50.0, step=5.0)

        submitted = st.form_submit_button("Prediksi Permintaan")

    if submitted:
        input_data = pd.DataFrame({
            'Age': [age], 'Gender': [gender],
            'Product Category': [product_category], 'Price per Unit': [price_per_unit]
        })

        input_data['Gender'] = le_gender.transform(input_data['Gender'])
        input_data['Product Category'] = le_category.transform(input_data['Product Category'])
        input_data[['Age', 'Price per Unit']] = scaler.transform(input_data[['Age', 'Price per Unit']])

        pred_knn = knn_prod.predict(input_data)[0]
        pred_nb = nb_prod.predict(input_data)[0]

        st.subheader("📈 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Prediksi Model KNN", value=pred_knn)
        with col2:
            st.metric(label="Prediksi Model Naive Bayes", value=pred_nb)
            
        st.success("Prediksi berhasil dibuat berdasarkan input yang Anda berikan.")
        st.warning("**Disclaimer**: Prediksi ini didasarkan pada data historis dan hanya untuk tujuan demonstrasi. Hasil sebenarnya dapat bervariasi.")
