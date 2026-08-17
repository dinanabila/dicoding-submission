# notes
Catatan pikiran selama pengerjaan submission [kelas dicoding pengembangan generative ai berbasis llm](https://www.dicoding.com/academies/857-pengembangan-generative-ai-berbasis-llm).


### Yang mana yang dipilih untuk di fine-tuning? Base model atau Instruct Model? Masing-masing trade-off nya?

Untuk case submission ini, better pakai base model. Soalnya biar bisa keliatan kemampuan model dalam berbahasa indonesia. Kalau instruct model sebenarnya valid juga. Tapi untuk konteks belajar fine tuning yang bisa bikin model llm itu jadi bisa bercakap sesuai dataset yang diwajibkan, base model lebih masuk akal. Kalau pakai instruct model, dalam konteks dataset ini, jatuhnya kek ngajarin apa yang udah dibisa model. Redundan.

### Cara tau kapasitas komputasi? Terkait VRAM dsb?


### Memangnya instruct model bisa di SFT kan lagi? Dalam case seperti apa contohnya? Dan dengan dataset seperti apa?

Bisa. Contohnya misalnya mau memoles lagi kecenderungan model dalam menjawab. Kek, misal model instruct nya kan basically memang udah bisa ngobrol biasa. Tapi kita bisa SFT kan lagi jadi model yang terbiasa ngobrol pakai bahasa gaul, misalkan. Biar apa, yaaaa, biar modelnya asikan dipakai anak zaman now.

### Apa banget bedanya model 4B, 8B, 16B, dst?