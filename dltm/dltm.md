
# notes
catatan pikiran selama pengerjaan submission [kelas dicoding membangun proyek deep learning tingkat mahir](https://www.dicoding.com/academies/818-membangun-proyek-deep-learning-tingkat-mahir).

*Last update: 2026/03/16*


## custom dan distributed training dengan tensorflow


### automatic differentiation dengan tf.GradientTape

tf.GradientTape itu gunanya untuk merekam isian fungsi yang mau kita turunkan selaku gradient descent. kalau ga pakai ini, jadinya ga bisa kerekam. python biasa ga bisa merekam. contohnya biar kebayang gini deh:

```
x = 3
y = x ** 2
gradient(y, x) = gradient(3 ** 2, x) = turunan 9 terhadap x di (x = 3) = 0
```

padahal yang kita pengen itu sebenernya ini kan:

```
gradient(y, x) = gradient(x ** 2, x) = turunan x^2 terhadap x di (x = 3) = 6
```

python biasa ga bisa ngerekam x ** 2, soalnya udah otomatis ngehitung x ** 2 pakai nilai x yang udah dideklarasikan di line sebelumnya. 

makanya, kita perlu tf.GradientTape(). tujuannya biar kerekam si x nya, bukannya malah ke-assign duluan value-nya sebelum diturunkan. 

kode lengkap:

```
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y = np.array([6.0, 4.0, 2.0, 0,0, -2.0, -4.0, -6.0], dtype=float)

weight = tf.Variable(random.random(), trainable=True)
bias = tf.Variable(random.random(), trainable=True)
   
def loss(y_true, y_pred):
	return tf.abs(y_true - y_pred)
	
def fit_data(x, y):
	with tf.GradientTape(persistent=True) as tape:
		y_pred = weight * x + bias
		loss = loss(y, y_pred)

	w_gradient = tape.gradient(loss, weight)
	b_gradient = tape.gradient(loss, bias)

	weight.assign_sub(w_gradient * learning_rate)
	bias.assign_sub(b_gradient * learning_rate)
```

nanti kalau udah diturunkan, si nilai gradient itu dipakai untuk memperbarui weight maupun bias. itu pakai assign_sub. sub di sini akronim dari subtract. jadi si weight atau bias lamanya disubtract sama hasil gradient descent yang udah dikomputasi sebelumnya.

persistent=True biar rekaman si gradienttape nya ga mati. kalau di-assign False, rekamannya mati setelah udah nurunin untuk pertama, si w_gradient, dan pas mau nurunin untuk bias b_gradient, jadi ga ada lagi riwayat rekamannya dan jadi ga bisa nurunin yang sesuai kita ekspektasikan deh.



### custom training loop




## submission dltm
sebuah catatan pikiran untuk pengerjaan submission dicoding kelas membangun proyek deep learning tingkat mahir

**alur kerja:** 
1. eda dataset
	- heatmap
	- analisis dekomposisi
	- analisis lag acf dan pacf
2. normalisasi / standardisasi data
3. bangun window dataset
4. bangun pipeline data pakai tf.data.Dataset
5. bagi data jadi data train - validation - test
6. feature engineering dengan rolling statistic
7. bangun baseline model lstm dasar
8. bangun custom layer dense
9. bangun custom layer multi-head attention
10. bangun custom layer bebas: dropout/normalization/activation function (elu/leaky relu/dll)
11. bangun model seq2seq lstm teacher forcing multistep 24 step pakai functional api / model subclassing, dan pakai custom layer yang udah dibangun
12. bangun custom loss mae dan modifikasi fungsi lebih lanjut
13. bangun custom callback early stopping dan modifikasi fungsi lebih lanjut
14. bangun custom training pakai custom loss dan callback yang udah dibangun
15. jalankan training pakai custom training
16. bangun inferensi untuk baseline lstm
17. bangun inferensi autoregressive untuk lstm seq2seq
18. inferensi kedua model pakai data test, tampilkan dalam bentuk chart


**ekspektasi output per kriteria:**
- **kriteria 1**
	- Heatmap korelasi
	- Analisis dekomposisi (skilled)
	- Uji ACF dan PACF (advanced)
	- Feature engineering Rolling Statistic (advanced)
	- Bangun pipeline dataset pakai tf.data.Dataset, bagi jadi data train-val-test (skilled)
	- Bangun baseline model lstm dasar dan latih pakai model.fit()
- **kriteria 2**
	- Output kerjaan dari Kriteria 2 adalah arsitektur model LSTM Seq2Seq yang mengadopsi strategi Teacher Forcing dan pastiin dia memenuhi multi-step time series forecasting sebanyak 24 step.
	- Kalau mau mencapai kriteria Basic, cukup bangun pakai Functional API aja udah memenuhi. Bikin juga satu layer Dense dari nol, pakai Custom Layer.
	- Kalau mau mencapai kriteria Skilled, ga hanya pakai Functional API, tapi, tunjukkan juga kalau model dibangun menggunakan Model Subclassing. Terus, lengkapi juga arsitekturnya dengan menambahkan layer Multi-Head Attention di baseline model LSTM dari Kriteria 1, sama ke model Seq2Seq LSTM nya.
	- Kalau mau mencapai Advanced, itu si layer Multi-Head Attention nya dibangun ulang dari nol pakai Custom Layer, terus terapin ke arsitektur baseline sama seq2seq modelnya. Tambahin juga satu custom layer lainnya, bebas, contoh yang bisa dibangun: Dropout layer, Normalization layer, Activation function layer (Elu, Leaky Relu, dll).
- **kriteria 3**
	- Output kerjaan dari Kriteria 3 adalah Custom Training buatan sendiri yang memanfaatkan tf.GradientTape untuk proses perhitungan gradiennya, yang mana hasil pelatihan dari Custom Training tersebut digunakan untuk melakukan inferensi prediksi dengan teknik Autoregressive. Loop Training dari Custom Training yang dibangun juga harus menampilkan jumlah epoch, loss, dan val loss selama berjalan.
	- Untuk kriteria Skilled, loss nya harus Custom, bangun dari nol. Pakai MAE. Bikin juga callback early stopping custom, bangun dari nol juga. Pasang keduanya di Custom Training yang udah dibangun.
	- Untuk kriteria Advanced, loss custom yang udah dibikin tadi harus dimodif lagi alias di-upgrade, dia harus bisa nambahin weight loss pada horizon atau step yang lebih jauh. Terus juga custom callback nya juga di-upgrade supaya bisa mengurangi learning rate secara bertahap saat validation loss nya stagnan selama beberapa epoch. Pasang kedua custom-an upgrade-an tadi di Custom Training yang udah dibangun. Sama, selain itu ada ketentuan tambahan juga untuk memenuhi kriteria Advanced: performa model custom Seq2Seq LSTM saat dievaluasi pada data test di bawah 0.015 MAE (kondisi sebelum di-inverse scaled).

### eda

- Apa yang perlu dicari tahu dari data untuk kepentingan time series forecasting?
    - Datanya stasioner ga?
        - Stasioner maksudnya stabil. Kek misal, dari waktu ke waktu, kenaikan rata-ratanya terus konstan atau stabil yang ga fluktuatif gonjang ganjing dangdut.
        - Kalau sekiranya ternyata stasioner, berarti bisa dipertimbangkan untuk tetap pakai model statistik sederhana aja kek ARIMA, biar hemat komputasi.
        - Tapi kalau ternyata ga stasioner, masih bisa pakai model statistik biasa kalau memang mau, tapi, datanya perlu di'rata'kan dulu.
        - Kalau ga mau ribet ngeratain manual, berarti pakai deep learning. Karena deep learning ahlinya mempelajari pola gaje yang sulit ditebak.
        - Kenapa perlu tau stasioner atau ngga? Soalnya model statistik berdiri di atas asumsi dataset yang stasioner. Ga stasioner? Ya shikatanai perlu pertimbangkan pakai model lain yang lebih mahal macam deep learning berarti.
        - Tool buat meriksa kestasioneran data: uji ADF.
    - Ada tren ga? Kalau ada, naik atau turun atau dua-duanya?
        - Tool buat meriksanya: **analisis dekomposisi**, yang memecah grafik data asli menjadi 3 komponen: tren, seasonal, residual. 
	        - analisis dekomposisi itu untuk ngeliat, apakah ada tren di data time series kita atau ngga. yang dilakukan dekomposisi adalah memisahkan pola tren, seasonal, dan noise dari data. jadi dari yang tadinya itu bentuk grafik datanya masih pure nyampur dalam satu kesatuan, setelah didekomposisi, grafiknya jadi kepisah 3: satu yang nangkap tren, satu yang nangkap seasonal, satunya lagi yang nangkap noise (untuk deteksi outlier biasanya).
	        - cara melakukan analisis dekomposisi, pakai ini: from statsmodels.tsa.seasonal import seasonal_decompose (ada di modul eksplorasi data dalam time series).
	        - Kenapa yang diminta adalah analisis dekomposisi pada data target? Kenapa ga untuk data fitur juga? --> Karena yang perlu kita tau pola trennya adalah data target, karena itu yang mau kita prediksi. Dalam forecasting, yang wajib dipahami strukturnya adalah variabel yang akan diprediksi, yaitu target. Kalau fitur, sifatnya kan ga selalu kayak target, kayak, boleh jadi ada yang kategorik, jadi mau dianalisis dekomposisi juga memang ga cocok.
	        - Esensinya apa dari melakukan analisis dekomposisi ini? Esensinya adalah dengan kita tahu pola / tren dari data target, kita jadi bisa punya pegangan untuk menentukan window size, lag, maupun jenis model yang akan digunakan untuk forecasting.
        - Esensi dari mengetahui tren, bisa untuk pertimbangan buat bikin fitur baru buat dimasukkan ke model. Contoh penerapan cara bikinnya: [https://www.kaggle.com/code/ryanholbrook/trend](https://www.kaggle.com/code/ryanholbrook/trend)
        - Contoh lainnya, dari hasil tahu ada pola musiman yang ditangkap dari grafik seasonal hasil dekomposisi, kita bisa pertimbangkan buat bikin fitur fourier. Sumber bacaan: [https://www.kaggle.com/code/ryanholbrook/seasonality](https://www.kaggle.com/code/ryanholbrook/seasonality)
    - Ada hubungan ga antara data hari ini dengan data kemarin atau minggu lalu atau lalu lalu? Kalau ada, seberapa jauh ke belakang masih relate?
        - Tool meriksa: **uji ACF dan PACF**.
        - Contoh lag plot: [![lag plot](https://camo.githubusercontent.com/aa28964d05455ba6e4fbb393b22785eb06f554d26205086af6d553d0afbc23f4/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f6b6167676c652d6d656469612f6c6561726e2f696d616765732f487672626f79612e706e67)](https://camo.githubusercontent.com/aa28964d05455ba6e4fbb393b22785eb06f554d26205086af6d553d0afbc23f4/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f6b6167676c652d6d656469612f6c6561726e2f696d616765732f487672626f79612e706e67)
        - Uji lag ACF dan atau PACF, maupun lag plot, itu lebih diperuntukkan untuk dependensi linear, bukan nonlinear. Kalau mau mastiin linear atau ngga nya, liat dari lag plot untuk beberapa lag (lag 1, lag 2, dst). Kalaupun ternyata mayoritasnya ga linear, gapapa juga kalau tetep mau pakai lag sebagai fitur, cuman ditransform dulu jadi linear, atau pakailah model yang mendukung kenonlinearan itu. Contoh PACF: [![pacf](https://camo.githubusercontent.com/0eac2f4cfebb28807003054901fdb145150a8c3f7e592155a3372cc0beff7617/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f6b6167676c652d6d656469612f6c6561726e2f696d616765732f366e54653934452e706e67)](https://camo.githubusercontent.com/0eac2f4cfebb28807003054901fdb145150a8c3f7e592155a3372cc0beff7617/68747470733a2f2f73746f726167652e676f6f676c65617069732e636f6d2f6b6167676c652d6d656469612f6c6561726e2f696d616765732f366e54653934452e706e67)
		- Cara interpretasi PACF: yang masih di luar area biru muda, itu berarti si fitur lag itu masih belum berkorelasi antar sesamanya, alias masih ada kontribusi yang signifikan jika dilibatkan. Untuk case contoh di atas, berarti lag 1 sampai lag 6 bisa banget dijadiin fitur. Tapi kalau lag 7 udah ngga, soalnya dia udah masuk area biru muda. Gimana dengan lag 11? Kalau itu dia false positive, dicurigainya.
        - Hasil uji acf dan pacf ini, selain untuk menentukan window size, juga bisa dijadikan pertimbangan untuk bikin fitur lag. Nilai di tanggal sebelumnya, entah itu nilai di satu hari sebelumnya, atau bahkan nilai di beberapa hari sebelumnya, itu bisa dijadikan fitur untuk melatih model forecasting time series. Fitur ini, dinamakan fitur lag. Kalau antara lag dan nilai sekarang itu berkorelasi, maka itu artinya si fitur lag ini bagus buat dijadikan fitur untuk melatih model. Masalahnya, kan lag itu bisa lag 1 (nilai di 1 hari sebelumnya), lag 2 (nilai di 2 hari sebelumnya), atau bahkan lag 1000000. Yang jadi pertanyaan, gimana cara menentukan lag terbaik untuk dijadikan fitur? Caranya dengan menggunakan uji lag ACF, atau dengan PACF. Sumber bacaan terkait: [https://www.kaggle.com/code/ryanholbrook/time-series-as-features](https://www.kaggle.com/code/ryanholbrook/time-series-as-features)
        - ACF, akronimnya AutoCorrelation Function. Dia itu lag plot yang kulampirin contohnya di atas.
		- Bedanya ACF dan PACF, kalau ACF itu dia ngasitau korelasi antara lag dengan si data dari sejumlah lag yang bersangkutan, tanpa misahin dampak dari lag-lag sebelumnya / di antara si lag itu dengan si data observasi. Kalau PACF, dia bener-bener pure ngasitau korelasi antar lag dan data observasi, yang bener-bener bersih tanpa ada keterlibatan dari lag-lag sebelumnya.
		- Contohnya, kalau ACF, si korelasi antara lag 2 dengan data observasi itu bisa jadi masih ada bagian lag 1 yang "ikut campur". Sementara kalau PACF, itu si lag 1 nya udah ga ada ikut-ikut kecampur lagi di korelasi antara lag 2 dan data observasi.
		- Dalam konteks machine learning, ACF berguna untuk melihat, apakah data time series yang kita gunakan itu relevan atau ga kalau pakai lag buat dijadiin fitur. Kalau sekiranya ternyata dari ACF itu ga keliat ada korelasi sama sekali antar data tanggal observasi dengan data lag nya, berarti bisa disimpulkan kalau lag ga worth it buat dijadikan fitur untuk melatih model.
		- Sementara PACF, faedahnya dalam konteks machine learning adalah untuk menghindari redundansi fitur. Idenya, kita ga mau pakai lag yang ternyata udah ga ngasih informasi fresh lagi. Ga fresh ini artiannya kalau dilihat dari konteks ACF, itu ga fresh yang dikarenakan korelasi yang terbentuk itu ternyata sebenernya mayoritas itu korelasi antar lag sebelumnya, bukan dari si lag yang sekarang. Nah kalau pakai PACF, kita bisa tau, oh, si lag ini ternyata udah ga ngasih informasi apa-apa lagi, soalnya udah diwakilkan lag sebelumnya. ACF, artinya, dalam hal ini bisa kita bayangkan sebagai hubungan data antar waktu ke waktu yang korelasinya itu mengandung unsur efek domino. Kalau PACF udah ga ada efek domino lagi karena pengaruh antar domino nya terhapuskan.
		- Kalau misalkan kita mau menentukan window size untuk LSTM Seq2Seq, artinya yang kita lihat itu ACF, bukan PACF, soalnya yang mau kita tahu itu, sampai seberapa jauh ke belakang data masih punya informasi yang berguna? Atau dengan kata lain, sampai sejauh mana efek domino itu ada?
		- Better ambil window size yang seterkecil mungkin, atau ambil yang gede setergede mungkin? Yang kecil. Soalnya kalau yang besar, mahal komputasi dan risiko overfitting sebab model terlalu belajar dari data yang banyak + noise. Terus juga, kalau pilih yang besar itu ibarat, suruh aja LSTM nya ngelatih dirinya sendiri dengan semua data, ngapain repot-repot nentuin window size lagi.
		- Jadi selain bisa untuk menentukan window size, ACF itu ibarat untuk lihat gambaran besar, buat tau, lag itu worth it atau ga buat dijadiin fitur. Sementara PACF lebih ke untuk nentuin, sampai lag berapa yang bener-bener worth it buat dijadiin fitur.
- Dari hasil menjawab tiga pertanyaan di atas, kita bisa melakukan feature engineering untuk menghasilkan fitur yang relevan dengan keadaan dataset, supaya mempermudah model dalam mempelajari data yang dimasukkan. Ingat, bikin yang relevan aja, jangan semua-semua dicemplungin jadi fitur. Contoh, kalau misalkan ga ada hubungan berarti dari hasil uji ACF dan PACF, berarti ga perlu bikin fitur lag untuk dicemplungin ke model.

### feature engineering

- Gimana caranya bikin fitur baru pakai Rolling Statistic? Seperti apa bentuknya?
    - Cara bikin fitur baru pakai Rolling Statistic dijelasin di modul Feature Engineering dalam Time Series. Esensinya untuk bikin fitur yang ngasitau "alert" atau gambaran dari keadaan data di suatu waktu, menggunakan statistik deskriptif dari kompilasi sejumlah window size data sebelum data ybs. Contohnya: rolling mean, rolling std, rolling min/max.

### preprocessing data

#### normalisasi & standardisasi
- Alasan dari kenapa normalisasi dan standardisasi ga cocok untuk memperbaiki skala data untuk keperluan time series forecasting, adalah karena hasil skala dari kedua metode tersebut jadi bikin tren hilang, jadi bikin ngilangin pola penting karena datanya jadi "rata".
- Selain itu, alasan lainnya kenapa normalisasi dan standardisasi ga cocok dipakai untuk penskalaan data time series (jika, katakan, diterapkan sebelum membagi dataset menjadi data train dan data test) adalah, karena kedua metode tersebut menggunakan statistik deskriptif (mean, std) dari keseluruhan data. Dan karena menggunakan itu, alhasil datanya jadi ngintip. Ngintipnya pas melakukan perhitungan si mean dari seluruh data tadi. Dan, jadinya bocor deh (yang mana dinamakan sebagai data leakage).
- Solusi dari keterbatasan normalisasi dan standardisasi untuk penskalaan data time series forecasting: percentage change transformation. Jadi dari yang tadinya datanya itu dalam bentuk nilai konkrit sebenarnya kayak tanggal segini harga jual 5000, tanggal besoknya harga jualnya 6000, setelah pakai penskalaan data pakai percentage change transformation, datanya sekarang jadi dalam bentuk, tanggal segini terjadi kenaikan/penurunan sebesar sekian persen, tanggal besoknya sekian persen, dst. Dan kekuatan dari persentase itu, dia sifatnya "universal", yang mana sejalan dengan tujuan penskalaan data dengan normalisasi / standardisasi. Terlebih, metode ini dijamin ga bikin data leakage karena rumus perhitungannya itu cukup melibatkan data tanggal hari h dengan data tanggal hari setelahnya (h+1). Jadi, ga ada lagi deh yang namanya ga sengaja ngintip data dari data masa depan.
- Tapi sempet baca juga kalau ternyata percentage change transformation itu kurang tepat buat multivariate time series forecasting, soalnya kalau multivariate itu kan dia pakai lebih dari satu fitur, dan antar fitur nya itu kemungkinan besar ada lah sedikitnya saling mempengaruhinya. Alhasil kalau kita pakai percentage change transformation, pola hubungan relatifnya jadi bisa bias. Jadi untuk multivariate time series buat ngerjain Kriteria 1, mending pakai normalisasi / standardisasi setelah datanya di-split jadi data train-test-val. Ingat, setelah ya, jangan sebelum, biar datanya ga ngintip.

#### window & horizon
- Window size menggambarkan ukuran seberapa jauh waktu data yang mau kita libatkan untuk ngeramal besaran data setelahnya, sekaligus untuk ngasih label. Contohnya, misalkan datanya itu hari 1: 20rb, hari 2: 15rb, hari 3: 25 rb. Kalau window size nya 2, berarti hari 1 dan 2 dipakai buat ngelabelin data hari 3, terus hari 2 dan 3 dipakai buat ngelabelin data hari 4, dst.
- Bedanya window untuk lstm seq2seq dengan lstm biasa, kalau lstm biasa, dia window nya terdiri dari input dan target. Kalau seq2seq, terdiri dari encoder input, decoder input, dan decoder target.
- Contohnya: misal untuk case univariate kita punya dataset mini [1, 2, 3, 4, 5].
- Terus horizon yang kita mau misalkan 2 (2 step), dengan ukuran window 2.
- Berarti window untuk lstm biasa gini:
    - window 1: input: [1, 2], target: [3, 4]
    - window 2: input: [2, 3], target: [4, 5]
- Sementara window untuk lstm seq2seq gini:
    - window 1: encoder input: [1, 2], decoder target: [3, 4], decoder input: [2, 3]
    - window 2: encoder input: [2, 3], decoder target: [4, 5], decoder input: [3, 4]
- Jadi yang jadi input untuk lstm seq2seq adalah encoder input dan decoder input, dan yang jadi targetnya adalah decoder target.
- Untuk case multivariate, prosesnya ga beda-beda banget dengan univariate. Yang bedain cuma array datasetnya. Misal kalau univariate kan gini [1, 2, 3]. Nah kalau multivariate, katakan 2 kolom, jadinya gini datasetnya \[[1, 11], [2, 12], [3, 13]].
- Contoh, kita punya dataset mini: \[[1, 11], [2, 12], [3, 13], [4, 14], [5, 15]].
- Lalu horizonnya misalkan 2, dengan window size 2 juga.
- Berarti untuk lstm biasa, window nya jadi gini:
    - window 1: input: \[[1, 11], [2, 12]], target: \[[3, 13], [4, 14]]
    - window 2: input: \[[2, 12], [3, 13]], target: \[[4, 14], [5, 15]]
- Dan untuk lstm seq2seq, windownya gini:
    - window 1: encoder input: \[[1, 11], [2, 12]], decoder target: \[[13], [14]], decoder input: \[[12], [13]]
    - window 2: encoder input: \[[2, 12], [3, 13]], decoder target: \[[14], [15]], decoder input: \[[13], [14]]
- Decoder input dan decoder targetnya hanya diambil dari satu fitur yang ingin diprediksi (akatakan dalam hal ini kolom terakhir, yang valuenya 11, 12, 13, 14, 15). Namanya many to one. Dan itu gapapa dan memang sengaja untuk keperluan time series forecasting.
- Link notebook kaggle untuk gambaran pembuatan window dataset: https://www.kaggle.com/code/dinanabb/window-dataset-time-series/


### deep learning

- pada dasarnya, neural network itu fungsi dengan input sebanyak jumlah fitur yang digunakan dari dataset, dengan output sebanyak jumlah target yang ingin diketahui / diklasifikasikan. parameternya seluruh weight dan bias dari setiap hubungan neuron antar layer.
- cost function, dalam hal ini, adalah fungsi yang inputnya justru adalah parameter dari fungsi neural network ybs, yaitu seluruh weight dan bias dalam satu batch training, dengan output nya berupa 1 angka yang merepresentasikan seberapa buruk / baik model dalam menyesuaikan parameter weight maupun bias untuk tugas pembelajaran. parameternya sejumlah batch training yang dilakukan model.
- jadi supaya tahu apakah model sudah benar-benar berhasil belajar atau belum, artinya metrik yang perlu dilihat adalah output dari cost function. tujuan kita mendapatkan angka output seminimal mungkin. gimana caranya? --> pakai gradient descent dan backpropagation
- kenapa ga langsung di-adjust aja weight dan bias nya, toh udah ada data target berlabel pas training? --> kalau neural networknya udah bukan linear lagi, ga bisa. 
![cost function](https://raw.githubusercontent.com/dinanabila/dicoding-submission/refs/heads/main/dltm/img/260307.jpg)
- jadi perlu yang namanya metode backpropagation untuk menyesuaikan weight dan bias baru sebesar gradient descent setiap batch training selesai. dinamakan backpropagation karena proses buat mencari gradient descent dilakukan secara mundur dari layer output ke layer input.
- proses perhitungan gradient descent dilakukan dengan memanfaatkan kalkulus turunan multivariabel. angka yang dijadikan besaran gradient descent untuk menyesuaikan weight baru adalah turunan dari cost function terhadap weight.
- cost function sendiri adalah jabaran dari kompilasi rangkaian dari alur fungsi neural network, mulai dari input, lalu hidden layer disertai fungsi aktivasi, hingga ke output, yang lalu kemudian dikurangi dengan nilai seharusnya, dan kemudian dikuadratkan.
- nama lain cost function --> loss function
- Contoh hubungan epoch dan mini batch gradient descent: 
![batch mini](https://raw.githubusercontent.com/dinanabila/dicoding-submission/refs/heads/main/dltm/img/260314.jpg)
- Contoh simulasi batch gradient descent. 
![batch simulation](https://raw.githubusercontent.com/dinanabila/dicoding-submission/refs/heads/main/dltm/img/260315.jpg)
- Weight dan bias yang digunakan dalam satu batch beneran sama. Dari hasil forward propagation, didapat kompilasi hasil prediksi dari data-data dalam satu batch yang sama menggunakan weight dan bias yang sama, yang dari situ diakumulasikan menjadi satu loss function kek mse. MSE nya kemudian digunakan untuk menghitung gradient descent untuk memperbarui weight dan bias.
- SGD --> batch size = 1,
- Batch Gradient Descent --> batch size = jumlah seluruh data di data train
- Mini Batch Gradient Descent --> batch size = sejumlah ukuran sampel yang ditentukan
- Biasanya hasil learning curve dari SGD kan gradakan, sementara hasil dari full batch mulus. Itu bisa terjadi karena di full batch, model bisa full lihat keadaan data dan jadi langsung tahu titik paling minimum secara global, sehingga jadi ga perlu kena noise lagi kayak SGD yang cuma bisa nebak-nebak karena cuma bisa lihat dari satu sampel data aja.

### arsitektur model

#### lstm vs rnn
- LSTM dikembangkan dari RNN, dan dibuat dengan tujuan mengatasi kekurangan RNN.
- RNN digunakan untuk membuat model yang mampu mempelajari hal secara sekuensial. Contohnya, memahami kalimat yang terdiri dari kata-kata yang saling berurutan dan membentuk konteks.
- Cara kerjanya gimana? Jadi dia masukin inputnya secara bertahap ke satu unit RNN nya. Dari satu tahap ke tahap selanjutnya, input masuk bertahap bersamaan dengan hasil output dari tahap-tahap sebelumnya. Terus begitu sampai inputnya habis, jadilah hasilnya adalah kompilasi dari informasi-informasi dari seluruh tahap, untuk kemudian diproses di back propagation dan diproses untuk menghasilkan weight dan bias baru untuk satu unit itu (yang sama untuk setiap timestep karena masih dalam satu unit) jika masih diperlukan untuk batch pelatihan selanjutnya.
- Keterbatasan RNN adalah dia tidak mampu mempelajari sekuensial yang sangat panjang. Contoh: paragraf.
- Kenapa? Soalnya dia ga punya sistem untuk menanggulangi masalah gradien memori lama yang makin mengecil seiring bertambahnya timestep, apalagi yang sangat panjang.
- LSTM punya. Atau varian lainnya, kek GRU, juga punya untuk mengatasi masalah itu.


#### baseline lstm dasar
- Baseline model LSTM itu sampai sebatas gimana baseline yang bener-bener baseline?
    - Sumber bacaan: [https://towardsdatascience.com/baseline-models-your-guide-for-model-building-1ec3aa244b8d/](https://towardsdatascience.com/baseline-models-your-guide-for-model-building-1ec3aa244b8d/)
    - Baseline model itu model yang arsitekturnya simpel sesimpel-simpelnya simpel. Bentuknya bisa berupa single neuron, atau linear model. Esensi bikin baseline model itu sebagai pembanding dengan model lainnya yang arsitekturnya lebih kompleks. katakan kita pengen mengembangkan model dengan menambahkan arsitektur lagi, terus, buat tau itu efektif atau ga, caranya dengan membandingkan performanya dengan baseline model. selain itu, karena kesimpelan dan sifat ringannya, baseline model biasanya juga dipakai buat ngenilai di awal eksperimen, kalau, data yang kita pakai itu udah berkualitas atau belum. "For deep learning projects, consider using a neural network with a simpler architecture."
    - Sumber bacaan ke-2: [https://www.kaggle.com/code/ryanholbrook/a-single-neuron](https://www.kaggle.com/code/ryanholbrook/a-single-neuron)
    - Though individual neurons will usually only function as part of a larger network, it's often useful to start with a single neuron model as a baseline. Single neuron models are linear models.
    - contoh:
        ```
        from tensorflow import keras
        from tensorflow.keras import layers
        
        # Create a network with 1 linear unit
        model = keras.Sequential([
            layers.Dense(units=1, input_shape=[3])
        ])
        ```


#### lstm seq2seq
- Model LSTM Seq2Seq terdiri dari encoder dan decoder. Keduanya dibangun menggunakan hidden layer LSTM.
- Untuk proses pelatihan model, alurnya dimulai dari memproses encoder_input ke dalam encoder di hidden layer LSTM, yang nantinya diatur sedemikian rupa agar keluarannya berupa dua hidden state yaitu state_c dan state_h. Output encoder diabaikan, karena pakai teacher forcing. Ingat, karena ini LSTM, jadinya encoder_input nya dimasukkan dan diproses satu per satu secara bertahap. Begitupula dengan pemrosesan decoder_input di tahap selanjutnya:
- Selanjutnya state_c dan state_h diteruskan ke hidden layer LSTM selanjutnya: decoder. Kedua state tersebut dimasukkan bersamaan dengan decoder input. Output dari decoder inilah hasil forecasting yang kemudian dievaluasi untuk dihitung gradiennya jika error masih tinggi alias jika model belum cukup belajar.
- Bagaimana dengan decoder_target yang juga sudah dibentuk bersamaan dengan encoder_input dan decoder_input dari proses pembuatan window dataset sebelumnya? Kalau itu dipakainya saat inferensi dan evaluasi model, bukan saat pelatihan model.
- **Proses inferensi lstm seq2seq** lebih panjang dibanding arsitektur model lstm seq2seq itu sendiri. Beberapa bagian dari hasil model yang sudah dilatih, itu diambil sebagai balok penyusun proses inferensi. Proses garis besarnya:
	- Bikin model baru khusus encoder, cukup ambil dari layer encoder dari arsitektur model yang udah dilatih sebelumnya.
	- Bikin juga model baru khusus decoder, ambil juga dari arsitektur model sebelumnya.
	- Ambil sampel 1 batch dari test_ds buat dipakai untuk inferensi.
	- Bangun alur untuk melaksanakan proses inferensi. Alurnya:
	    - Masukkan encoder_input dari sampel test_ds ke model encoder, proses (predict).
	    - Ambil state dari hasil predict encoder sebelumnya, juga ambil satu nilai terakhir dari encoder_input ybs.
	    - Masukkan state dan nilai terakhir dari encoder_input tadi ke model decoder, proses (predict).
	    - Hasil predict tadi, si output hasil decoder beserta state terbarukannya, dipakai lagi untuk memprediksi step selanjutnya. Ulang terus sampai sejumlah horizon yang ditetapkan.
	    - Ini yang membedakan antara arsitektur model untuk proses pelatihan, dengan arsitektur saat inferensi. Kalau pas pelatihan pakai teacher forcing yang model decodernya disuguhkan ground truth berupa sejumlah potongan contekan dari decoder_target val_ds tanpa nilai akhir, nah kalau pas inferensi, si decodernya ga dikasih contekan lagi, dan hanya mengandalkan hasil output dari prediksi timestep sebelumnya beserta state terbarukannya. Sebutannya autoregressive.
	    - Autoregressive maksudnya adalah setiap langkah prediksi itu bergantung dengan hasil dari prediksi sebelumnya. Untuk case time series forecasting lstm seq2seq kali ini, kita ga pakai model.predict() buat melakukan inference. Jadi perlu bikin fungsi baru buat fasilitasin proses prediksi berbasis autoregressive.
	    - Dan karena dia autoregressive, jadinya memang perlu dibuat terpisah modelnya antara encoder dan decoder, soalnya kita perlu proses loop biar si decodernya bisa memproses hasil terbarukannya bertahap satu per satu, ga sekaligus untuk horizon yang ditetapkan.
- Ada ngaruhnya ga kalau ubah jumlah unit dari layer encoder maupun decoder di arsitektur model lstm seq2seq untuk pelatihan, ke proses inferensinya? Ga ada, ga ada konfigurasi lebih lanjut yang diperlukan. Encoder dan decoder untuk proses inferensi boleh plek ketiplek aja dari model aslinya.
- Dari lstm seq2seq ini jadi tau kalau input itu ternyata ga harus diproses secara berurutan dari satu hidden layer ke hidden layer setelahnya.
- Contohnya kayak si decoder_input. Dia baru dimasukkan setelah encoder selesai memproses seluruh encoder_input, dan decoder_input ini dimasukkannya bukan ke hidden layer pertama lagi, melainkan langsung ke hidden layer kedua: decoder lstm layer.

#### teacher forcing
- Teacher Forcing itu salah satu strategi pelatihan yang dapat diterapkan untuk membangun model LSTM Seq2Seq. Ibaratnya, anggaplah si model kita ini si murid. Dengan Teacher Forcing, si murid ini diajari dengan cara seperti guru mengajari murid dengan cara menyuapi jawabannya langsung tanpa membuat si murid mencari tahu jawabannya sendiri secara mandiri. Tujuannya biar proses belajar tetap aman terkendali.
- Contoh teknisnya, kalau tanpa teacher forcing, si murid alias si model ini bakal ngeprediksi hari +2 pakai hasil prediksi hari+1, lalu hari +3 pakai hasil prediksi hari+2. Jadi kek dilepas gitu aja tanpa dituntun guru. Akibatnya apa. Kalau dia salah prediksi satu aja, misal, di hasil prediksi hari+2 hasil error kek mae nya tinggi banget, imbasnya jadi nular ke hasil prediksi untuk hari-hari setelahnya.
- Nah kalau pakai teacher forcing, si model bakal ngeprediksi pakai jawaban asli dari setiap langkah sebagai inputnya. Jadi kayak ngasih contekan yang bener.

#### attention
- Attention dipakai supaya bisa bikin model bisa lebih fokus ke hal yang lebih penting, alias, bagian data yang paling relevan. 
- Attention terdiri dari 3 bagian: query, key, dan value.
- Ketiganya diperoleh dari hasil mengalikan input dataset dengan matriks weight query, key, dan value yang diperbarui berkala setiap training loop backpropagation.
- Untuk peran masing-masingnya, mungkin lebih kebayang kalau lihat jawaban pertanyaan tanggal 5 lalu di poin selanjutnya.
- Tujuan dari attention adalah untuk memperoleh konteks antar data input, agar bisa memperoleh hasil yang lebih akurat.
- Multi-head attention bisa dibilang layer berlapis dengan weight key, query, dan value-nya tersendiri (setiap head punya weight yang berbeda).
- Kalau dalam konteks time series forecasting, ibaratnya head pertama perannya sebagai ahli pengamat pola musiman. Head kedua ahlinya mengamati pola mingguan, head ketiga harian. Dll. Cuma gambaran, ga berarti gitu untuk case benerannya.
- Kenapa yang jadi query untuk attention itu decoder outputs, dan kenapa yang jadi key dan value itu encoder outputs? Dan kenapa key dan value nya sama?
	- Decoder output dijadikan query sebab dari hasil prediksi decoder itu ingin dicaritahu, kira-kira ada ga nih hubungan antara hasil prediksi suatu timestep dengan timestep lainnya. Kemudian, seluruh timestep hasil prediksi di decoder output bakal diperbarui agar menyesuaikan dengan keadaan 'konteks' data.
	- Ga sampai di situ, sekarang pertanyaan berlanjut ke: ada pola yang sekiranya bisa mempengaruhi hasil prediksi dari decoder output tadi ga, kalau meninjau dari hasil pembelajaran nya encoder?
	- Untuk menjawab itu, encoder output diproses melalui proses transformasi hingga menjadi key, dan dari situ diproses dengan decoder output yang sudah diproses menjadi query melalui proses dot product. Interpretasinya, semakin besar nilainya, semakin ada hubungan signifikan antara pola data dari masa lalu dengan hasil prediksi yang sedemikian sehingga kerasa perlu buat model untuk 'memperbarui' hasil prediksi dari decoder output.
	- Value berperan sebagai 'cerita' dari pola yang ada dari encoder output selaku key, makanya diprosesnya pakai encoder output, bukan decoder output.
	- Di penghujung, value diproses bersama hasil dot product key dan query. Sebagai gambaran, bayangin satu timestep aja dari decoder output, bukan semua. Bayangin, setiap hasil dari dot product key-query dari suatu timestep itu dikalikan dengan value dari setiap timestep encoder output dari hasil dot product key-value timestep yang sama. Nah itu semua nantinya dijumlahkan, lalu hasil itulah yang akhirnya ibarat kek gradient descent yang memperbarui hasil prediksi timestep decoder output ybs.
	- Proses ini sebutannya cross-attention, bukan self-attention, soalnya key dan query nya ga berasal dari input yang sama: key nya dari encoder output, query nya dari decoder output.

#### custom layer
- Meskipun Tensorflow udah nyediain layer bawaan buat langsung kita comot dan pakai, di lapangannya, boleh jadi justru kita perlu layer yang lebih spesifik tujuan penggunaannya. Dan itu, bisa kita wujudkan dengan membuat custom layer.
- Gimana cara membuat ulang layer dari nol dengan custom layer? --> Caranya dengan membuat class baru yang mewarisi kelas Layer dari Tensorflow. Lalu class baru itu kita isi dengan fungsi-fungsi sesuai dengan kebutuhan layer yang kita inginkan. Untuk detail caranya ada di submodul Kustomisasi Layer dengan Subclassing.

#### functional api
- Kalau yang biasanya dipakai untuk belajar di kelas pemula itu, namanya sequential API. Contohnya:
    ```
    # sequential api
    
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Dense
    
    seq_model = Sequential([
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
        ])
    ```
    
- dan kalau contoh di atas diadaptasi jadi dalam bentuk Functional API, jadinya gini:
    
    ```
    # functional api
    
    from tensorflow.keras import Model
    from tensorflow.keras.layers import Dense, Input
    
    #Inisiasi input layer
    input_layer = Input(shape=(4,))
    
    #Susun layer yang terhubung hingga ke output layer
    dense_1 = Dense(128, activation='relu')(input_layer)
    dense_2 = Dense(64, activation='relu')(dense_1)
    output_layer = Dense(1, activation='sigmoid')(dense_2)
    
    #Deklarasikan model dengan input dan output
    model = Model(inputs=input_layer, outputs=output_layer)
    ```
    
- Terus bedanya di mana? Di tingkat kefleksibelannya. Kalau Functional API, dia itu antara layernya itu bisa disambung-sambungin dengan fleksibel / modular, ga kayak Sequential API yang saklek harus berurutan antar layer yang udah dideklarasikan. Liat deh, itu contohnya aja keliatan dari susunan layer kalau pakai functional api. Fleksibel gitu dia ibarat kalau fungsi di matematika itu f(g(h(x))), terus bisa disesuaiin jadi f(h(g(x))), atau bahkan bisa juga g(f(x)) aja. Caranya, dengan bikin layer-layer output yang dipengenin, terus tarok di parameter outputs nya Model().
- Dari situ kita tau kalau functional api itu, dia input nya bisa lebih dari satu, outputnya juga bisa lebih dari satu. Tuh liat aja parameter dari Model(). Namanya inputs kan, bukan input? Terus juga outputs, bukan output.

#### model subclassing
- Ini versi lebih fleksibelnya lagi dari Functional API. Konsepnya beda-beda tipis sama custom layer. Dengan menerapkan model subclassing, kita jadi bisa bangun arsitektur model yang lebih custom lagi, jauh lebih custom dari yang ditawarkan Functional API.

#### insight dari trial error pembangunan arsitektur model
- Awalnya itu kan aku pakai 500 window, soalnya berdasarkan hasil uji acf, lag nya itu masih bagus di 500an. Window 500 mungkin memang gede, tapi kupikir gapapa karena kan jumlah row datasetnya juga banyak, 50ribuan. Tapi ternyata pas modelnya dilatih, LAMA BANGET. Dan jadi overfitting parah.
- Jadi kukurangi window nya jadi 111 aja. Terus batch_size juga kukurangi dari ribuan, jadi 100an aja karena nge-crash kalau ribuan.
- Sejauh ini aman walau masih overfitting. Yang penting proses training nya ga selama yang awal.
- kwargs itu penting. Tadinya kuanggap sepele jadi sempet ga kupake pas bangun custom layer dkk. TAPI TERNYATA PENTING. Pokoknya kalau mau nge-inherit class lain tapi ragu bakal ada yang ketinggal dideklarasiin, pakai kwargs aja biar pasti dan ga ada yang ketinggal lagi.


### training

- Kalau yang biasanya itu kan kalau mau ngelatih model itu pakainya sintaks model.fit() sama model.compile() tanpa bener-bener merhatiin apa-apa aja yang sebenernya terjadi di baliknya. Pokoknya, yang penting, ngelatih aja. Udah.
- Proses pelatihan deep learning itu pada dasarnya loop. Makanya dinamakan Loop Training.
- Kalau dipelajari lagi, proses pelatihan deep learning itu kan terdiri dari beberapa proses. Dimulai dari input data, terus latih modelnya pakai input data tadi dengan weight awal, lalu hitung loss nya, abistu hitung gradien dari fungsi loss terhadap parameter model, baru deh abis itu perbarui bobot berdasarkan gradien yang udah dihitung tadi, dan balik lagi nge-loop ke awal ke proses pelatihan lagi sampai si loss nya itu cilik secilik-ciliknya yang diinginkan a.k.a sampai optimal.
- Terus kenapa banget kita masih perlu meng-custom proses training ini pakai Custom Training? Ya ga ada ruginya sih. Ibarat nih, misalkan katakan selama ini kamu pakai mobil yang mesin + tool kemudinya emang udah dirancang dari sananya dari pabrik dan kamu tinggal pake, nah, kali ini kamu rakit mobilnya sendiri itu include si daleman-daleman mesinnya beserta tool kemudinya kek setirnya, persenelingnya, remnya dkk nya, itu semua kamu yang atur dah. Biar apa banget? Biar sesuai sama yang benar-benar kita butuhkan. Kek, misalkan kamu selama ini ga puas sama kinerja wiper bawaan pabrik karena kurang mengibas air hujan, nah karena kamu yang kali ini ngerakit sendiri, kamu juga yang bisa atur tu wiper mau gerak gimana biar mata bebas dari air hujan, lebih bebas dari wiper mobil hasil produksi bawaan pabrik. Kek misalkan juga ada sesuatu di mesin yang kamu pengen improve, misal, kamu pengen si mesin nya ga kentut-kentut tiap pindah perseneling. Nah itu kamu juga bisa atur di rakitannya biar ga kentut lagi. Jadi, antara hasil rakitan dengan hasil versi pabrikan memang bisa sama-sama jalan dengan tujuan yang sama, tapi bedanya hasil rakitanmu lebih bisa memenuhi yang kamu butuhkan.

#### tf.GradientTape
- Ini untuk fasiitasin proses perhitungan gradien di custom training model. Jadi kalau pakai ini, kita ga perlu bikin lagi rumus turunan secara manual.
- Gimana cara membangun custom training dengan tf.GradientTape? --> Kaitkan tf.GradientTape dengan proses perhitungan loss dan weight sama bias awalnya. Dibikin dalam satu fungsi. Detail contohnya ada di submodul Automatic Differentiation dengan tf.GradientTape.

#### early stopping
- Selama proses training, ada baiknya bikin chart learning curve kek gini: ![learning curve](https://raw.githubusercontent.com/dinanabila/dicoding-submission/refs/heads/main/dltm/img/260311.png)
- Tujuannya supaya ada gambaran, model ada indikasi underfitting atau overfitting atau ngga.
- Underfitting terjadi ketika data yang digunakan untuk melatih model masih terlalu sedikit dibandingkan apa yang ingin diuji dengan data test. Ibaratnya, siswa yang belum bisa menggeneralisir apa yang sudah dipelajarinya untuk digunakan di kehidupan nyata.
- Overfitting terjadi ketika model terlalu memegang teguh apa yang dipelajarinya dari data, sehingga ketika disuguhkan data test baru yang belum pernah dilihatnya, dia tetap saklek dan jadinya ga sesuai yang diharapkan.
- Callbacks gunanya untuk ngasih batasan kapan berhentinya proses training. Indikatornya bisa pakai early stopping, bisa pakai dropout, bisa pakai dll.
- Salah satu cara menangani overfitting adalah dengan melakukan early stopping.
- Early stopping adalah salah satu metode regularisasi untuk mengatasi overfitting. Ibaratnya rem yang menghentikan proses training jika validation loss di learning curve sudah tidak ada indikasi menurun lagi dan justru malah menaik lagi. Metode regularisasi lainnya yang terkenal adalah dropout.

#### dropout
- Dropout cara kerjanya mematikan sejumlah neuron secara acak selama proses training. Tujuannya untuk menghindari overfitting.
- Dropout digunakan untuk mengatasi overfitting. Kalau diingat lagi, suatu model bisa mengalami overfitting karena terlalu saklek pas belajar, dalam hal ini, weights nya terlalu presisi terhadap data training.
- Terlalu presisinya weights itu bisa terjadi, boleh jadi karena dalam sekali training epoch, kebetulan modelnya udah dapat kombinasi neuron yang bikin loss nya turun, bahkan sebelum banyak epoch terlalui.
- Kadang atau biasanya, di antara kombinasi itu ada neuron 'andalan' yang menurunkan loss sendiri tanpa melibatkan neuron lainnya. Ibarat neuron yang nguli kerja kelompok seorang diri tanpa melibatkan neuron-neuron sekelompoknya untuk berkontribusi, sehingga ketika mereka terpecah dan menjalani hidup masing-masing, neuron-neuron yang ga dapat kesempatan nguli itu ga bisa melakukan hal sepresisi ketika mereka masih bersama neuron kuli yang belajar semuanya sendiri tanpa ngajak belajar bareng itu.
- Akibatnya, model udah keliatan pintar duluan (padahal belum) sebelum benar-benar memberi kesempatan untuk neuron-neuron lainnya dan juga si neuron kuli itu untuk belajar bareng dan menyesuaikan sinergi lagi.
- Dengan mematikan neuron secara acak, proses training model jadi bisa dilakukan dengan lebih bervariasi, sehingga tidak ada neuron yang terlalu dominan, dan semua neuron punya kesempatan yang sama untuk belajar, alias punya kesempatan yang sama untuk diperbarui weight nya dengan lebih adil dan merata.
- Untuk case RNN, termasuk halnya lstm seq2seq, pastikan agar dropout tidak mematikan neuron dalam satu sequence yang sama, dalam hal ini, satu neuron RNN, sebab bisa mengacaukan ingatan RNN dalam pola yang diperoleh dari setiap timestep. Ibaratnya satu timestep dimatikan aja, ya pola nya jadi keputus dan model jadi ga tau pola sebenarnya gimana.

#### insight dari trial error proses training
- Masalah pertama pas nge-inferensi pakai data test itu, yang ketauan itu si jarak antara hasil prediksi dengan data aktual itu beda jauh. Biar ga jomplang banget lagi di awal, itu si learning rate di optimizer adam itu dikecilin, sama, jumlah head dan lstm unit key dim di multi head attention nya kunaikin. Jadinya rada mendingan dan ga terlalu jomplang lagi, terus ga terlalu overfitting lagi antara loss dan val loss nya.
- Masalah kedua, itu si hasil prediksi pas inferensi naik udah kayak eksponensial. Padahal harusnya stabil. Pas kuubah dari normalisasi ke standardisasi, jadinya ga gitu lagi. Masih ga stabil, tapi udah rada lumayan.
- Masalah ketiga, performa modelnya belum mencapai harapan. Masih perlu diutak-atik lagi, mungkin bisa dari arsitektur modelnya.



### yang belum dicaritahu

- Dalam konteks time series forecasting lstm seq2seq, attention-nya berarti harus gitu arsitekturnya? Atau, boleh ga kalau attention nya ditaruh di masing-masing encoder dan decoder. Jadi self-attention jatuhnya. Kira-kira gimana kalau gitu?
- Kalau dataset nya udah dipreprocessing jadi window dataset, masih perlu bikin fitur lag ga? Atau malah jadinya redundan?
- Dalam konteks time series forecasting lstm seq2seq, attention-nya berarti harus gitu arsitekturnya? Atau, boleh ga kalau attention nya ditaruh di masing-masing encoder dan decoder. Jadi self-attention jatuhnya. Kira-kira gimana kalau gitu?
