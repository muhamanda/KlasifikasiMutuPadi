# **Predictive Analytics - Klasifikasi Mutu Padi**

## **Domain Proyek**

Beras merupakan salah satu bahan pokok paling penting di dunia untuk konsumsi manusia yang berasal dari padi. Di negara negara asia yang penduduknya padat, khususnya Bangladesh, Myanmar, Kamboja, Cina, Indonesia, Korea, laos Filipina, Sri langka, thailand dan vietnam, beras juga merupakan salah satu makanan pokok. Sebanyak 75 % masukan harian masyarakat di negara-negara asia tersebut berasal dari beras. Lebih dari 50 % penduduk dunia tergantung pada beras sebagai sumber kalori utama(Resita wahyu dianti, 2010)

Padi organik juga memiliki gizi yang cukup tinggi untuk di konsumsi oleh masyarakat. Pertumbuhan yang sangat pesat dari akumulasi data mutu padi organik dari seorang expert telah menciptakan kondisi kaya akan data tapi minim informasi. Jika mutu padi organik diketahui maka pihak dinas pertanian dapat melakukan penentuan harga yang sesuai dengan mutu dan juga meningkatkan kepercayaan mitra yang berkerja sama dengan pihak dinas pertanian. Oleh karena itu maka di pandang perlu untuk melakukan sebuah penelitian dalam mencari pola dari mutu padi organik dengan menggunakan machine learning dengan tujuan agar pengklasifikasian tidak lagi sekedar hanya menggunakan perkiraan semata tapi menggunakan data pengalaman dari seorang expert yang sudah di extrak dan menjadi acuan utama dalam menentukan klasifikasi mutu dari sebuah padi organik. 

## **Business understanding**

Dinas pertanian memiliki data terkait padi organik, namun data tersebut
masih sulit untuk melihat pola mutu padi organik secara menyeluruh, untuk kepentingan pengembangan selanjutnya maka dibutuhkan sebuah model yang bisa mengklasifikasi data mutu padi organik di Dinas pertanian,
hal ini diperlukan agar data yang ada bisa lebih bermanfaat lagi bukan hanya sekedar
tumpukan data yang minim informasi.

### Problem Statements
*   Dari serangkaian fitur yang ada, fitur apa yang paling berpengaruh terhadap mutu padi organik?
*   Apa mutu padi organik dengan karakteristik atau fitur tertentu?  

### Goals
*   Mengetahui fitur yang paling berkorelasi dengan mutu padi oragnik.
*   Membuat model machine learning yang dapat memprediksi mutu padi organik seakurat mungkin berdasarkan fitur-fitur yang ada.

### Metodologi
Prediksi mutu/kualitas adalah tujuan yang ingin dicapai. Seperti yang kita tahu, mutu merupakan variabel diskrit. Dalam predictive analytics, saat membuat prediksi variabel diskrit artinya Anda sedang menyelesaikan permasalahan klasifikasi. Oleh karena itu, metodologi pada proyek ini adalah: membangun model klasifikasi dengan mutu padi organik sebagai target.

### Metrik
Metrik digunakan untuk mengevaluasi seberapa baik model Anda dalam memprediksi mutu. Untuk kasus klasifikasi, metrik yang biasanya digunakan adalah accuracy.

## **Data understanding**

Data yang Anda gunakan pada proyek kali ini adalah Data Mutu Padi Organik yang dapat  diunduh di [Drive saya](https://drive.google.com/file/d/1AnABSK_LiYWYobo4eD1fCxiqM63GjYQv/view?usp=sharing).

Dataset ini memiliki **4.952** sampel padi dengan berbagai karakteristik dan grade mutu. Karakteristik yang dimaksud di sini adalah fitur non-numerik seperti varietas, warna, rasa, teknik, musim, dan penyakit, serta fitur numerik seperti Panjang, Besar, dan PH.

### Data Loading

Supaya isi dataset lebih mudah dipahami, kita perlu melakukan proses loading data terlebih dahulu. Dataset yang akan kita gunakan bernama Data Mutu Padi Organik.csv.

Memberikan informasi sebagai berikut:

*   Ada 4.952 baris (records atau jumlah pengamatan) dalam dataset.
*   Terdapat 10 kolom yaitu: varietas, panjang,	besar,	warna,	rasa,	teknik,	musim,	penyakit,	PH, dan	grade mutu.

### Exploratory Data Analysis - Deskripsi Variabel

Exploratory data analysis (EDA) merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data.

Secara umum, Anda dapat melakukan proses EDA untuk menjawab beberapa pertanyaan berikut:
*   Apa saja jenis variabel pada dataset?
*   Apakah ada missing value?
*   Bagaimana distribusi variabel dalam dataset?

Dalam menjawab pertanyaan-pertanyaan di atas, Anda perlu melakukan beberapa hal pada data.

### Exploratory Data Analysis - Menangani Missing Value dan Outliers

Beberapa pengamatan dalam satu set data kadang berada di luar lingkungan pengamatan lainnya. Pengamatan seperti itu disebut outlier. Outliers adalah sampel yang nilainya sangat jauh dari cakupan umum data utama. Ia adalah hasil pengamatan yang kemunculannya sangat jarang dan berbeda dari data hasil pengamatan lainnya. 

Ada beberapa teknik untuk menangani outliers, antara lain:
*   Hypothesis Testing
*   Z-score method
*   IQR Method

Pada kasus ini, Anda akan mendeteksi outliers dengan teknik visualisasi data (boxplot). Kemudian, Anda akan menangani outliers dengan teknik IQR method. 

### Exploratory Data Analysis - Univariate Analysis

Selanjutnya, kita akan melakukan proses analisis data dengan teknik Univariate EDA. Pertama, Anda bagi fitur pada dataset menjadi dua bagian, yaitu numerical features dan categorical features.

## **Data preparation**

### Encoding Data Kategori

Untuk melakukan proses encoding fitur kategori, salah satu teknik yang umum dilakukan adalah teknik one-hot-encoding. Sebagai langkah awal, kita akan bagi data menjadi fitur (x) dan target (y) karena penangan encoding terhadap data kategorinya berbeda.

### Pembagian dataset dengan fungsi train_test_split dari library sklearn

Ketahuilah bahwa setiap transformasi yang kita lakukan pada data juga merupakan bagian dari model. Karena data uji (test set) berperan sebagai data baru, kita perlu melakukan semua proses transformasi dalam data latih. Inilah alasan mengapa langkah awal adalah membagi dataset sebelum melakukan transformasi apa pun. Tujuannya adalah agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. 

Proses scaling pada seluruh dataset membuat model memiliki informasi mengenai distribusi pada data uji. Informasi tentang data uji (yang seharusnya tidak dilihat oleh model) turut diikutsertakan dalam proses transformasi data latih. Oleh karena itu, kita akan melakukan proses scaling secara terpisah antara data latih dan data uji. 

### Standarisasi

Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. 

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

## **Modeling**

Pada tahap ini, kita akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, kita akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. 

### K-Nearest Neighbor

KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. 

Meskipun algoritma KNN mudah dipahami dan digunakan, ia memiliki kekurangan jika dihadapkan pada jumlah fitur atau dimensi yang besar.

### Random Forest

Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Ide dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian.

Ada dua teknik pendekatan dalam membuat model ensemble, yaitu bagging dan boosting. Jangan bingung dulu dengan istilah ini ya. Kita akan bahas satu per satu. 

Bagging atau bootstrap aggregating adalah teknik yang melatih model dengan sampel random.

### Boosting Algorithm

Sebagai model ensemble, keduanya terdiri dari beberapa model yang bekerja secara bersama-sama. Pada teknik bagging, setiap model dilatih secara paralel. Sedangkan, pada teknik boosting, model dilatih secara berurutan atau dalam proses yang iteratif. 

Seperti namanya, boosting, algoritma ini bertujuan untuk meningkatkan performa atau akurasi prediksi. Caranya adalah dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner).

## **Evaluation**

Metrik yang akan kita gunakan pada prediksi ini adalah accuracy. Namun, sebelum menghitung nilai accuracy dalam model, kita perlu melakukan proses scaling fitur numerik pada data uji.
