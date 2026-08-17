### note:
catatan pikiran selama pengerjaan submission [kelas dicoding belajar fundamental deep learning](https://www.dicoding.com/academies/185-belajar-fundamental-deep-learning) proyek klasifikasi gambar. 


### alur kerja:
1. dapetin dataset. dataset nya harus dalam bentuk yang **belum** terpisah jadi train-test-validation set. 
2. liat distribusi datanya. imbang ga antar label nya? kek misalkan kalau gambar dengan label pneumonia lebih banyak daripada gambar dengan label normal, itu artinya masih jomplang. kalau ada yang jomplang, harus kita seimbangkan, caranya dengan data augmentation, dan ini ada banyak sih cara yang bisa dipilih. data augmentation itu ngakalin gimana menggandakan gambar tapi hasil gambar nya ga sama persis dengan gambar referensi, tapi tetep senada, sehingga bisa diikutkan ke dataset. 
3. setelah udah dipastikan seimbang, bagi dataset jadi train-test-validation set (data splitting). nah cara baginya itu, berdasarkan contoh dari modul dicoding, dia itu kan karena sumber datasetnya itu udah terpisah jadi folder train-test-val, jadinya dia itu pertamanya banget nge-combine dulu semuanya ke dalam satu folder dataset. terus abis itu, dipakein train_test_split buat mecah test dan train. terus, dari situ buat dapetin val nya, itu pakai imagedatagenerator, pakai dataset train buat mecahnya jadi train-val. terus baru deh dari situ dikelola masing-masing train-val-test nya pakai flow_from_directory.
4. terus kita normalisasikan masing-masing dataset yang udah di-split tadi, pakai kelas ImageDataGenerator punya nya Keras
5. bangun model. udah dikasih contohnya 8 arsitektur model di notebook punya dicoding. yang perlu di-notice, di arsitektur ini perlu dipahami kalau di dalamnya itu ada yang namanya pooling layer, conv2d. terus kalau sequential itu kek kerangka deh kayaknya. di tahap ini juga implementasikan callback kalau mau
6. bikin grafik evaluasi terhadap akurasi dan loss model (kodenya ada di notebook dicoding)
7. simpan model ke dalam format savedmodel, tf-lite, dan tfjs
8. inferensi pakai salah satu model yang udah di-save tadi (savedmodel, tf-lite, tfjs)