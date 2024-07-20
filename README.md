# Submission 2: Stroke Prediction Machine Learning Pipeline
Nama: Annisa Mufidatun Sholihah

Username dicoding: annisams11

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset/data) |
| Masalah | Stroke adalah penyebab kematian kedua secara global, bertanggung jawab atas sekitar 11% dari total kematian. Masalah yang diangkat adalah prediksi apakah seorang pasien mungkin mengalami stroke berdasarkan parameter input seperti jenis kelamin, usia, berbagai penyakit, dan status merokok. |
| Solusi machine learning | Solusi yang akan dibuat adalah model machine learning berbasis neural network untuk memprediksi kemungkinan stroke pada pasien. Model ini akan menggunakan berbagai fitur seperti jenis kelamin, usia, riwayat hipertensi, penyakit jantung, status pernikahan, tipe pekerjaan, tipe tempat tinggal, tingkat glukosa rata-rata, indeks massa tubuh (BMI), dan status merokok. |
| Metode pengolahan | Metode pengolahan data yang digunakan meliputi transformasi fitur-fitur kategorikal menjadi one-hot encoding dan normalisasi fitur-fitur numerik ke rentang [0, 1]. Proses ini dilakukan menggunakan TensorFlow Transform (TFT). Pipeline TFX yang digunakan mencakup komponen-komponen berikut: `CsvExampleGen`, `StatisticsGen`, `SchemaGen`, `ExampleValidator`, `Transform`, `Trainer`, `Evaluator`, dan `Pusher`. |
| Arsitektur model | Arsitektur model yang digunakan adalah jaringan saraf tiruan dengan beberapa lapisan tersembunyi. Model ini menggunakan Keras Tuner untuk mengatur jumlah unit pada setiap lapisan serta tingkat pembelajaran yang optimal. Terdapat tiga lapisan tersembunyi dengan jumlah unit yang bervariasi antara 128, 256, dan 512 untuk lapisan pertama, 32, 64, dan 128 untuk lapisan kedua, serta 8, 16, dan 32 untuk lapisan ketiga. Lapisan output menggunakan aktivasi sigmoid untuk prediksi biner. |
| Metrik evaluasi | Metrik yang digunakan untuk mengevaluasi performa model adalah akurasi biner. Selain itu, digunakan juga metrik `AUC`, `Precision`, `Recall`, dan `ExampleCount` untuk evaluasi yang lebih komprehensif. Metrik akurasi biner memiliki ambang batas (threshold) yang memastikan bahwa akurasi model harus lebih dari 0.5 dan peningkatan absolut minimal 0.0001. |
| Performa model | Model yang dibuat menunjukkan performa yang baik dengan akurasi validasi yang tinggi pada data evaluasi. Model ini dilatih menggunakan optimasi Adam dengan hyperparameter terbaik yang dipilih oleh Keras Tuner. |
| Opsi deployment | Model yang telah dilatih akan dideploy sebagai layanan web menggunakan TensorFlow Serving. Model ini dapat diakses melalui endpoint REST API yang memungkinkan pengguna untuk mengirimkan data pasien dan menerima prediksi stroke secara real-time. |
| Web app | [stroke-model](https://stroke-pred-deploy-production.up.railway.app/v1/models/stroke-model/metadata) |
| Monitoring | Monitoring dilakukan untuk memastikan model yang telah dideploy berfungsi dengan baik dan memberikan prediksi yang akurat. Monitoring dilakukan dengan menggunakan prometheus. |
