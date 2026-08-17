# notes
catatan pikiran selama pengerjaan submission [kelas dicoding belajar fundamental deep learning](https://www.dicoding.com/academies/185-belajar-fundamental-deep-learning) proyek analisis sentimen. 

Langkah-langkah ngerjain submisison analisis sentimen:
1. scraping datanya
2. bersihin dan preprocess datanya
3. label kan datanya
4. ekstraksi fitur
5. latih model
6. evaluasi model
7. inferensi

## Scraping Dataset

ini kalau di modulnya itu dia pakai library google_play_scraper, buat scrape ulasan dari aplikasi yang ada di playstore. library ini memungkinkan kita buat ambil ulasan berdasarkan negara tempat tinggal pengguna maupun bahasa ulasan yang digunakan. terus bisa juga kustom, mau ambil berapa ulasan (jadi boleh ga semua). urutannya juga bisa dikustom, mau ambil ulasan the most relevan bisa, mau diurutkan ke newest juga bisa. 

kuncinya: baca dokumentasi google_play_scraper biar kebayang apa aja yang bisa dilakukannya, dan gimana outputnya o_o

## Pemrosesan Data

ini lebih ke manipulasi string. pelajari lagi aja sintaks python nya. 
sama, cukup dimengerti aja tahapan-tahapan yang diperlukan untuk preprocess datanya (kek stemming, dkk nya)

## Pelabelan

pelabelan itu maksudnya ngasih label, misal untuk konteks analisis sentimen ini, berarti labelnya: positif/netral/negatif. tujuannya, biar model punya target, jadi kek supervised learning klasifikasi gitu yang harus ada fitur target nya. 

pelabelan ini pakai data lexicon (biasanya pakai dari repo orang, bisa dicari sesuai bahasa dan konteksnya, itu kalau indo. tapi kalau inggris, aku di submission itu pakai opinion_lexicon nya nltk. itu didownload aja di bagian import library). terus juga buat ngasih label berdasarkan lexicon ini, caranya dengan bangun loop dan kondisional sendiri berdasarkan teks yang udah kita preprocess sebelumnya. 

contohnya, kalau terdapat kata di ulasan yang ada di lexicon positif, hitung skor jadi +=1, sebanyak kata yang muncul lagi di lexicon positif. Begitupula kalau ada kata yang muncul di lexicon negatif, skornya jadi -=1. 

itu terus di-loop buat ngitung skor setiap ulasan, terus baru deh dari situ di loop lagi buat bikin kolom baru yang isinya label, dengan kondisi jika skor < 0, berarti negatif, kalau > = berarti positif, kalau = 0 berarti negatif. 

## Ekstraksi Fitur

ekstraksi fitur gunanya untuk ngubah teks jadi numerik, supaya bisa dimengerti model. caranya? tergantung keadaan teks-teks di dataset itu, dan tergantung metode yang dipakai (tf-idf, bow, word embedding, masing-masing beda). tapi sebenernya cukup ngertiin konsep generalnya aja kalau mau sekedar kodenya bisa jalan. tinggal pakai library aja soalnya udah bisa jalan

## Model

ini kan di submission nya itu ada kriteria opsional ngelatih model deep learning. nah kalau aku itu di submission sebelumnya pakai model deep learning biasa: MLP. 

di modul dicoding nya sendiri terkait nlp ini, sebenernya udah diperkenalkan lstm dan gru, tapi belum ada contoh kodenya di modulnya. 

jadinya belum kebayang gimana kalau pakai lstm / gru. 

tapi lagian pakai MLP aja gapapa, malah justru nyambung karena aku udah punya dataset yang udah dalam bentukan yang udah di ekstraksi fitur pakai bow/tfidf. soalnya juga kan di kriteria utama nya itu boleh pakai model machine learning tradisional kayak random forest. 

jadi pakai MLP juga nyambung, malah memang harusnya pakai itu sih kayaknya (?)