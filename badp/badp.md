### note
catatan pikiran selama pengerjaan submission [kelas dicoding belajar analisis data untuk pemula](https://www.dicoding.com/academies/906-belajar-analisis-data-untuk-pemula). 

### spreadsheet:
hapus duplikat datanya setelah isi semua null aja, soalnya takutnya ada row yang nilainya null tapi row lain yang sebenarnya identik itu ga null. kalau kayak gitu kan jadinya ga dianggap identik, dan jadinya kurang bersih deh.

eh tapi ga usah ding. sama aja, mau setelah atau sebelum untuk case dataset ini. 

### dashboard
pertimbangkan
- gunakan filter periode yang bisa diklik pengguna, lalu kemudian seluruh data di dashboard menyesuaikan dengan filter periode yang dipilih
- kalau bisa bikin pembaca ga mesti hover area tertemtu untuk tau data persisnya, pertimbangkan untuk taruh nilainya secara eksplisit. tapi pertahankan juga supaya ga 'bulky'
- beri judul representatif di setiap chart
- beri label teks deskripsi singkat di titik data yang ingin di-highlight (contoh: label 'Penjualan Tertinggi' di titik data dengan penjualan tertinggi)
- mainkan warna. beri warna menonjol sendirian untuk titik data yang ingin di-highlight
- beri warna berbeda dari warna background dashboard pada teks deskripsi masing-masing chart
- untuk line chart yang hasilnya patah-patah kecil-kecil kusut, bisa coba haluskan jadi kayak kurva gelombang supaya lebih terlihat naik turunnya

### widget orange:
list seluruh widget yang diperlukan untuk memenuhi kriteria submission: 
- file
- outliers
- select columns
- data table
- data sample
- continuize
- gradient boosting
- random forest
- test and score
- predictions


### optional dari baca-baca web orange:
- pakai rank untuk feature selection best nya, mungkin ini bisa bantu meningkatkan performa model


### alur preprocessing di orange:
ini rada ngecoh, alurnya. baca docs di bagian ini: https://orange3.readthedocs.io/projects/orange-visual-programming/en/latest/widgets/evaluate/testandscore.html#preprocessing-for-predictive-modeling


### reproducibility model
untuk model kek random forest atau gradient boost, jangan lupa centang replicable training supaya hasil evaluasi metrik di widget test and score dan predictions tetap sama kalau dijalankan ulang. 