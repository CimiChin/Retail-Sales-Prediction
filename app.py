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
import numpy as np

warnings.filterwarnings('ignore')

# Konfigurasi Halaman
st.set_page_config(
    page_title="Prediksi Permintaan Produk Fashion",
    page_icon="👕",
    layout="wide"
)

# Fungsi untuk memuat dan cache data dengan kategori & mapping kuantitas
@st.cache_data
def load_data():
    df = pd.read_csv('retail_sales_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Membuat kategori permintaan yang lebih beragam menggunakan Kuantil
    try:
        quantiles = pd.qcut(df['Quantity'], q=5, labels=False, duplicates='drop')
        labels = [
            'Permintaan Rendah', 'Permintaan Cukup Rendah', 'Permintaan Sedang', 
            'Permintaan Tinggi', 'Permintaan Sangat Tinggi'
        ]
        unique_quantiles = quantiles.nunique()
        df['Demand'] = pd.qcut(df['Quantity'], q=unique_quantiles, labels=labels[:unique_quantiles])
    except ValueError:
        bins = [0, 1, 2, 3, 4, float('inf')]
        labels = ['Rendah (1)', 'Cukup Rendah (2)', 'Sedang (3)', 'Tinggi (4)', 'Sangat Tinggi (5+)']
        df['Demand'] = pd.cut(df['Quantity'], bins=bins, labels=labels, right=False)

    df.dropna(subset=['Demand'], inplace=True)
    
    # REVISI: Membuat mapping dari Kategori Permintaan ke Kuantitas Median
    # Ini akan kita gunakan untuk memberikan estimasi jumlah.
    demand_quantity_map = df.groupby('Demand')['Quantity'].median().to_dict()

    return df, demand_quantity_map

df, demand_quantity_map = load_data()

# Sidebar untuk Navigasi
st.sidebar.title("Navigasi 🧭")
page = st.sidebar.radio("Pilih Halaman:", ["Dataset & EDA", "Pelatihan Model", "Formulir Prediksi"])

# =====================================================================================
# Halaman 1: EDA (Exploratory Data Analysis)
# =====================================================================================
if page == "Dataset & EDA":
    st.title("📊 Analisis Data Eksploratif (EDA)")
    st.markdown("Halaman ini menampilkan analisis awal dari dataset penjualan ritel.")
    
    if st.checkbox("Tampilkan Dataset Mentah (dengan kategori permintaan)"):
        st.write(df.head())
        
    st.subheader("Estimasi Kuantitas per Kategori Permintaan")
    st.info("Berdasarkan data historis, berikut adalah estimasi (median) jumlah unit untuk setiap kategori permintaan:")
    st.json({k: int(v) for k, v in demand_quantity_map.items()})

    st.header("Karakteristik Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Jumlah Transaksi", f"{df.shape[0]:,}")
        st.metric("Jumlah Pelanggan Unik", f"{df['Customer ID'].nunique():,}")
    with col2:
        st.metric("Jumlah Kategori Produk", f"{df['Product Category'].nunique()}")
        st.metric("Periode Data", f"{df['Date'].min().strftime('%d %B %Y')} - {df['Date'].max().strftime('%d %B %Y')}")

    st.header("Visualisasi Data")
    df_clothing = df[df['Product Category'] == 'Clothing']

    st.subheader("Distribusi Tingkat Permintaan untuk Produk Pakaian")
    fig1 = px.histogram(df_clothing, x='Demand', title='Distribusi Tingkat Permintaan Pakaian',
                        color='Demand',
                        category_orders={"Demand": sorted(df_clothing['Demand'].unique())})
    st.plotly_chart(fig1, use_container_width=True)

# =====================================================================================
# Halaman 2: Pelatihan Model
# =====================================================================================
elif page == "Pelatihan Model":
    st.title("🤖 Hasil Pelatihan Model Machine Learning")
    st.markdown("Halaman ini menunjukkan hasil pelatihan model untuk memprediksi **kategori** permintaan produk.")

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

# =====================================================================================
# Halaman 3: Formulir Prediksi
# =====================================================================================
elif page == "Formulir Prediksi":
    st.title("📝 Formulir Prediksi Permintaan")
    st.markdown("Isi formulir di bawah ini untuk mendapatkan prediksi **kategori permintaan** beserta **estimasi jumlah unit**.")

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

        input_data_transformed = input_data.copy()
        input_data_transformed['Gender'] = le_gender.transform(input_data_transformed['Gender'])
        input_data_transformed['Product Category'] = le_category.transform(input_data_transformed['Product Category'])
        input_data_transformed[['Age', 'Price per Unit']] = scaler.transform(input_data_transformed[['Age', 'Price per Unit']])

        # Lakukan Prediksi Kategori
        pred_knn_category = knn_prod.predict(input_data_transformed)[0]
        pred_nb_category = nb_prod.predict(input_data_transformed)[0]

        # REVISI: Dapatkan estimasi kuantitas dari mapping
        estimasi_knn_qty = int(demand_quantity_map.get(pred_knn_category, 0))
        estimasi_nb_qty = int(demand_quantity_map.get(pred_nb_category, 0))

        st.subheader("📈 Hasil Prediksi")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                label="Prediksi Model KNN", 
                value=pred_knn_category, 
                delta=f"Estimasi {estimasi_knn_qty} Unit",
                delta_color="off" # Warna delta menjadi netral
            )
        with col2:
            st.metric(
                label="Prediksi Model Naive Bayes", 
                value=pred_nb_category, 
                delta=f"Estimasi {estimasi_nb_qty} Unit",
                delta_color="off" # Warna delta menjadi netral
            )
        
        st.success("Prediksi berhasil dibuat berdasarkan input yang Anda berikan.")
        st.warning("**Disclaimer**: Estimasi jumlah unit adalah nilai median dari data historis untuk kategori yang diprediksi. Ini adalah perkiraan, bukan angka pasti.")
