INI ADALAH KODE UNTUK MELAKUKAN PELATIHAN MODEL SARIMA DENGAN MENGGUNAKAN DATASET SELAMA 3 BULAN DENGAN TOTAL 90 HARI (89 data, data ke 90)

Ini adalah kode untuk melakukan pelatihan model machine learning SARIMA dengan meggunakan Dataset selama 3 bulan dengan total 90 data (89 data untuk training, data ke 90 puluh untuk pembandingan hasil

Dependencies : 
joblib==1.1.0
matplotlib==3.4.3
numpy==1.21.1
pandas==1.3.1
scipy==1.7.1
statsmodels==0.12.2
scikit-learn==0.24.2

IMPORTANT :
dataset yang digunakan telah dimodifikasi agar sesuai dengan format yang direkomendasikan oleh PANDAS terutama DATETIME, mohon ikuti format yang telah ada sebagai contoh

NOTE :
#SARIMA MULTI CORE ADALAH VERSI SARIMA.PY yang menggunakan pararel processing yang mempercepat proses training model
# dengan menggunakan semua core yang ada di CPU, namun juga menggunakan memory yang lebih besar(tergantung data yang di training) 
# biasanya penggunaan ram akan penuh sebelum penggunaan CPU mencapai 100%., dan ada kemungkinan gagal jika memory tidak cukup