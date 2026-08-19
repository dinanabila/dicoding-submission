# notes
Catatan pikiran selama pengerjaan submission [kelas dicoding pengembangan generative ai berbasis llm](https://www.dicoding.com/academies/857-pengembangan-generative-ai-berbasis-llm).

## MODEL

### Concept

#### Yang mana yang dipilih untuk di fine-tuning? Base model atau Instruct Model? Masing-masing trade-off nya?

Untuk case submission ini, better pakai base model. Soalnya biar bisa keliatan kemampuan model dalam berbahasa indonesia. Kalau instruct model sebenarnya valid juga. Tapi untuk konteks belajar fine tuning yang bisa bikin model llm itu jadi bisa bercakap sesuai dataset yang diwajibkan, base model lebih masuk akal. Kalau pakai instruct model, dalam konteks dataset ini, jatuhnya kek ngajarin apa yang udah dibisa model. Redundan.

#### Cara tau kapasitas komputasi? Terkait VRAM dsb?

Untuk sekarang, cukup liat dari TPU gratisan di colab aja wkwk. 16 apa 15gb kalau ga salah. Dengan kapasitas segitu, udah bisa sft qlora model 8b paling banternya.

#### Memangnya instruct model bisa di SFT kan lagi? Dalam case seperti apa contohnya? Dan dengan dataset seperti apa?

Bisa. Contohnya misalnya mau memoles lagi kecenderungan model dalam menjawab. Kek, misal model instruct nya kan basically memang udah bisa ngobrol biasa. Tapi kita bisa SFT kan lagi jadi model yang terbiasa ngobrol pakai bahasa gaul, misalkan. Biar apa, yaaaa, biar modelnya asikan dipakai anak zaman now.

#### Apa banget bedanya model 4B, 8B, 16B, dst?

Bedanya di jumlah parameter yang dimiliki model. Semakin banyak parameternya, umumnya semakin tinggi kapasitas model dalam mengaitkan hubungan antar konteks. B = billion. 4B = sekitar 4 miliar parameter berarti. 

#### Gimana cara menentukan model yang paling tepat berdasarkan dataset yang digunakan?

Selain juga menimbang dari segi size model (4B/8B/dst), cari juga base model yang dilatih pakai dataset multilingual yang include bahasa yang sama dengan bahasa yang dipakai di dataset. 

Dalam case dataset berbahasa indonesia yang diwajibkan, karena jumlah baris data nya ga terlalu banyak (50k rows), dan karena sifatnya hanya sebatas input-output yang ga terlalu kompleks, jadi sebaiknya memang pilih model yang dari sananya udah bisa meng-handle bahasa indonesia, seengganya yang udah terpapar walau ga 'native speaker'. 

Jadi objektifnya adalah fokus mengajarkan model gaya chat instruction dengan memanfaatkan sft, bukannya membuang sebagian besar kapasitas sft untuk sekedar mengajari model bahasa indonesia dari nol. Kalau itu, udah tugas orang lain pas melakukan pretraining base model. Keknya gitu (?)


#### Gimana cara tahu bahasa yang dibisa model?

Lihat dari model card di platform kek huggingface. Bisa juga lihat dari training data yang digunakan untuk melatih modelnya. 

Jadinya untuk model yang kupakai untuk submission ini: https://huggingface.co/unsloth/Qwen3-4B-Base

### Plan

Untuk model, pilih Qwen3-4B karena udah support multilingual. 4B udah termasuk 'menengah', yang mana kapasitas modelnya ga terlalu simpel, tapi juga ga terlalu kompleks berat. Pas untuk keterbatasan komputasi yang tapinya tetap ingin memperjuangkan performa model ._.

Model yang sama ini kulatih dua kali dengan dua konfigurasi berbeda. Yang satu konfigurasinya serba murah serba ringan. Yang satunya lagi yang bagus, sebisa dan semampunya batasan resource komputasi yang ada. 


## DATASET

### Concept

#### Gimana cara melakukan format dataset sesuai model yang digunakan? Ada ketentuan tertentu yang harus dipenuhi ga? Taunya dari mana dan gimana?
