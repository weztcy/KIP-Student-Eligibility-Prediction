import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle
from sklearn.naive_bayes import GaussianNB

st.markdown("<h1 style='text-align: center; color: #9ABDDC;'>Aplikasi Prediksi Penerima KIP</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.write(' ')
with col2:
    st.image("kip.png")
with col3:
    st.write(' ')

st.markdown("<h5 style='text-align: center; color: #9ABDDC;'>Aplikasi berbasis web ini dirancang untuk memprediksi atau mengklasifikasi siswa yang berhak menerima KIP atau tidak. Data yang digunakan adalah dataset private peserta didik dari SMP Sepuluh November 02 Semarang.</h5>", unsafe_allow_html=True)

st.sidebar.header('Menu Input Data Siswa')

upload_file = st.sidebar.file_uploader("Upload file csv anda disini", type=["csv"])
if upload_file is not None:
    input = pd.read_csv(upload_file)
else:
    def input_user():
        penghasilan = st.sidebar.slider('Berapa penghasilan orang tua siswa?', 0, 10000000, 0)
        peserta_kps = st.sidebar.selectbox('Apakah keluarga anda memiliki KPS?', ('Ya', 'Tidak'))
        alat_transportasi = st.sidebar.selectbox('Apa alat transportasi yang digunakan siswa ke sekolah?', ('Jalan kaki', 'Angkutan umum', 'Kendaraan pribadi'))
        jumlah_saudara_kandung = st.sidebar.slider('Berapa jumlah saudara kandung siswa?', 0, 5, 0)
        jarak_rumah_ke_sekolah = st.sidebar.slider('Berapa jarak rumah siswa ke sekolah?', 0, 5, 0)
        rekomendasi_bsm = st.sidebar.selectbox('Apakah siswa mendapatkan rekomendasi BSM?', ('Ya', 'Tidak'))
        data = {
            'penghasilan': penghasilan,
            'peserta_kps': peserta_kps,
            'alat_transportasi': alat_transportasi,
            'jumlah_saudara_kandung': jumlah_saudara_kandung,
            'jarak_rumah_ke_sekolah': jarak_rumah_ke_sekolah,
            'rekomendasi_bsm': rekomendasi_bsm
        }
        fitur = pd.DataFrame(data, index=[0])
        return fitur
    input = input_user()

dataset = pd.read_csv('dataset.csv')
data = dataset.drop(columns=['penerima_kip'])
df = pd.concat([input, data], axis=0)

encode = ['peserta_kps', 'alat_transportasi', 'rekomendasi_bsm']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df = df[:1]

st.write(' ')
st.write(' ')
st.markdown("<h3 style='color: #F78119;'>Data Siswa</h3>", unsafe_allow_html=True)
if upload_file is not None:
    st.write(df)
else:
    st.write('Saat ini memakai sample input karena tidak ada file CSV yang diupload.')
    st.write(df)

load_model = pickle.load(open('modelNBC_kip.pkl', 'rb'))

predict = load_model.predict(df)
predict_proba = load_model.predict_proba(df)

st.markdown("<h3 style='color: #F78119;'>Keterangan Label Kelas</h3>", unsafe_allow_html=True)
penerima_kip = np.array(['Tidak', 'Ya'])
st.write(penerima_kip)

st.markdown("<h3 style='color: #F78119;'>Hasil Prediksi atau Klasifikasi</h3>", unsafe_allow_html=True)
st.write(penerima_kip[predict])

st.markdown("<h3 style='color: #F78119;'>Probabilitas Hasil Prediksi atau Klasifikasi</h3>", unsafe_allow_html=True)
st.write(predict_proba)