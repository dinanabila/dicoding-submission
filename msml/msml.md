# notes
catatan pikiran selama pengerjaan submission [kelas dicoding membangun sistem machine learning](https://www.dicoding.com/academies/713-membangun-sistem-machine-learning).

tahapannya:
- **KRITERIA 1: MELAKUKAN EKSPERIMEN TERHADAP DATASET PELATIHAN**
	- eda dataset.
	- preprocessing dataset di notebook.
	- kode preprocessing yang di notebook, dipindahin ke file python. jadiin script python.
	- tarok file python preprocessing nya ke repo github.
	- bikin workflow pakai github action.
	- yey jadi deh preprocessing otomatis tiap ada perubahan dataset :D
- **KRITERIA 2: MEMBANGUN MODEL MACHINE LEARNING**
	- hm initu langsung bikin script kode untuk ngelatih model ml pakai dataset hasil preprocessing. jadi bikin script kode nya langsung dalam berkas python.
	- script kodenya ini include kode untuk integrasiin hasil log pelatihan model ke mlflow, baik via lokal, maupun via online (dagshub).
	- oh ya, implementasi autolog vs manual logging juga dilakukan di tahap ini biar tau bedanya. 
	- sama bedanya pakai hyperparameter tuning sama ngga, juga diliat gimana bedanya hasil track di mlflow nya.
- **KRITERIA 3: MEMBUAT WORKFLOW CI**
	- integrasikan kode model mlflow dari kriteria 2 ke workflow ci nya github action, terus hubungkan juga ke docker container. tujuannya biar, hm, baca ini deh: https://www.dicoding.com/academies/713/tutorials/42161 biar tau manfaat mlflow, maupun docker, maupun keduanya ketika digabungkan untuk bekerjasama.
	- sepenangkapanku, mlflow memang bisa memfasilitasi pembungkusan model beserta artifaknya ke docker, dalam bentuk image. jadi developer tinggal pakai aja sebungkusan dari docker itu tanpa harus penyesuaian secara manual lagi dari awal.
- **KRITERIA 4: MEMBUAT SISTEM MONITORING DAN LOGGING**
	- ini ngaitin mlflow / docker images ke sistem monitoring grafana dan prometheus.
	- tujuannya biar terpantau kalau ada 'kejanggalan' selama model dijalankan di tahap produksi.
	- prometheus gunanya untuk mencatat, tapi dia ga punya dashboard interaktif. jadi dia dikolaborasikan dengan grafana, karena grafana menyediakan itu. terus juga, grafana juga menyediakan fitur alerting otomatis, kek ke email, atau ke chat discord.


