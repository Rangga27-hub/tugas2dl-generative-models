# PRD Tugas Take-Home Deep Learning D6421 Sesi 07
## Studi komparatif model generatif dan batasan MLP Classifier

**Mahasiswa** Komang Rangga Hariwijaya · NIM dan kelas diisi sebelum pengumpulan  
**Sumber persyaratan** Panduan Tugas Take-Home D6421, §§1–7, enam halaman  
**Luaran** Laporan PDF ilmiah 10–12 halaman dan repositori GitHub publik  
**Target pengerjaan** 6 jam sebagai target kerja. Ukur durasi aktual training dan sampling DDPM sebelum mengunci jadwal. Tenggat resmi mengikuti forum Binusmaya.

## 1. Tujuan dan kriteria selesai

Bandingkan enam arsitektur pada satu benchmark dan jawab Q1–Q6 dengan hasil pengujian. Jalankan lima eksperimen MLP untuk menguji secara terpisah keterbatasan transposisi bobot, kandungan informasi fitur klasifikasi, dan sampling dari ruang laten.

| Luaran | Kriteria penerimaan |
|---|---|
| Enam model | Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, dan DDPM dilatih dengan split bersama, seed tetap, serta loss dan checkpoint yang terdokumentasi. |
| Eksperimen MLP | Akurasi test ≥88%. Kelima tahap dijalankan. Frozen decoder dibandingkan dengan Vanilla AE berlatent 64. |
| Evaluasi | Tabel enam model berisi dimensi laten, detik/epoch, latensi batch 64, MSE, PSNR, SSIM, FID, dan catatan stabilitas. Tandai N/A sesuai jenis model. |
| Visual | Lima figur wajib lengkap, bernomor, tajam, berlabel, dan dianalisis. |
| Laporan | PDF 10–12 halaman, tujuh diagram arsitektur, formulasi loss, eksperimen MLP, hasil, Q1–Q6, dan argumen industri. |
| Repositori | Publik, tautan aktif di halaman judul, skrip PyTorch modular, requirements.txt, README, dan commit bertahap sesuai proses aktual. |

## 2. Kunci desain eksperimen

- **Dataset usulan** Pakai Fashion-MNIST sesuai rekomendasi dosen. Jika batas waktu ketat dan benchmark belum dikunci, MNIST merupakan pilihan sah dengan kompleksitas lebih rendah. Pilih satu sebelum menjalankan eksperimen final dan gunakan untuk semua model serta MLP.
- **Split dan data loader** Buat split 80/10/10 dari 70.000 citra grayscale 28×28, yaitu 56.000 train, 7.000 validation, dan 7.000 test. Catat cara membentuk split dari dataset torchvision dan simpan indeksnya. Gunakan batch 128 sebagai titik awal (256 juga diperbolehkan), shuffle train saja. Simpan seed 42 untuk Python, NumPy, PyTorch, dan loader yang mengacak data.
- **Rentang piksel** Gunakan [0,1] sebagai standar evaluasi. AE, VAE, dan MLP dapat berangkat dari [0,1]. Jika GAN memakai tanh atau DDPM dilatih pada [−1,1], petakan data pada jalur model tersebut dan konversi keluaran ke [0,1] sebelum visualisasi atau evaluasi. Simpan normalisasi dalam konfigurasi.
- **Kontrol kebocoran** Pilih hiperparameter dan checkpoint melalui train/validation. Evaluasi numerik final pada test. Tahap 5 MLP secara eksplisit meminta statistik laten dari test dan dijalankan sebagai analisis akhir dengan decoder terkunci. Catat bahwa sampel dari tahap ini menggunakan statistik test dan gunakan hanya untuk eksperimen MLP tersebut.
- **Lingkungan usulan** Gunakan PyTorch dengan CUDA pada RTX 5070 Ti 12 GB. Catat versi Python, torch, torchvision, perangkat, precision, dan konfigurasi tiap run. Struktur awal yang cukup adalah modul data, model, training, eksperimen MLP, evaluasi, dan pembuatan figur, beserta konfigurasi dan README. Simpan log per epoch, checkpoint lokal, metrik, dan figur sehingga tahap berikutnya bisa diulang tanpa training ulang.

## 3. Urutan implementasi dan pelatihan

### Langkah 1 — Siapkan data, baseline, dan pengukuran runtime

1. Inisialisasi repo dan cek GPU. Unduh dataset melalui torchvision, buat indeks split, lalu cek ukuran, rentang, dan ketiadaan irisan antar-split.
2. Siapkan konfigurasi, pencatatan per epoch, penyimpanan checkpoint, dan jalur resume. Simpan optimizer, epoch, dan konfigurasi bersama bobot apabila training perlu dilanjutkan.
3. Buat satu putaran uji singkat untuk DDPM. Ukur waktu satu epoch train dan reverse sampling satu batch 64 dengan jumlah langkah yang direncanakan. Perkirakan total waktu train dan sampling FID sebelum menetapkan jumlah epoch final. Cek juga jalur ekstraksi fitur FID pada batch contoh.
4. Latih MLP sejak awal dan lihat akurasi validasi. Jika masih di bawah target, cek skala input, split, loss pada logits, dan konvergensi. Sesuaikan epoch atau learning rate berdasarkan validasi. Hitung akurasi test setelah konfigurasi dipilih.

**Selesai ketika** split bisa dimuat ulang dengan hasil identik, forward/backward DDPM berjalan dengan loss finite, dan estimasi runtime sudah mencakup sampling FID.

### Langkah 2 — Latih enam arsitektur

| Model | Rancangan awal | Loss dan bukti yang disimpan |
|---|---|---|
| Vanilla AE | MLP 784 → 256 → 64 → 256 → 784, ReLU tersembunyi dan sigmoid keluaran | MSE rekonstruksi. Simpan loss, rekonstruksi test, dan kode laten 64 sebagai kontrol MLP. |
| Sparse AE | Kapasitas sama seperti Vanilla AE, penalti pada aktivasi laten 64 | MSE + λ × rerata norma L1 per sampel. Catat λ, lokasi penalti, dan proporsi aktivasi mendekati nol dengan ambang yang sama untuk kedua AE. |
| VAE | Encoder MLP dengan kepala μ dan log-variance, laten 64, reparameterisasi dan decoder | Rekonstruksi + KL terhadap N(0,I). Dokumentasikan skala reduksi kedua komponen dan simpan kurvanya. Gunakan μ untuk rekonstruksi dan titik awal interpolasi. |
| Vanilla GAN | Generator MLP dari noise Gaussian 100 ke 784 piksel, discriminator MLP dengan satu logit | Simpan loss G dan D. BCE dengan logits dan objektif generator non-saturating dapat digunakan. Catat fungsi objektif aktual saat membahas teori minimax. |
| DCGAN | Noise Gaussian 100, generator dengan upsampling konvolusional, discriminator konvolusional, periksa keluaran 28×28 | Simpan loss G dan D serta grid dari fixed noise. Catat penggunaan stride, normalisasi, dan aktivasi per blok. |
| DDPM | Denoiser konvolusional ringan dengan input timestep dan prediksi noise, ruang derau 1×28×28 | MSE antara noise asli dan prediksi. Simpan noise schedule, jumlah langkah T, kurva loss, serta beberapa snapshot reverse chain dari seed yang sama. |

Jalankan uji mini setiap model sebelum training penuh. Periksa dimensi, loss finite, update parameter, dan satu keluaran. Mulai training DDPM lebih awal setelah konfigurasi layak, lalu gunakan waktu tunggunya untuk menyiapkan evaluasi dan kerangka laporan. Antrekan proses GPU bila berbagi perangkat. Untuk model yang lebih cepat, selesaikan Vanilla AE sebelum evaluasi frozen decoder, lalu Sparse AE, VAE, dan kedua GAN. Simpan sampel dari beberapa epoch GAN agar mode collapse dapat dilihat bersama kurva G/D.

**Selesai ketika** enam checkpoint final dapat dimuat, setiap run punya log per epoch, dan sampling serta rekonstruksi berjalan dari checkpoint.

### Langkah 3 — Jalankan lima eksperimen MLP

| Tahap | Pekerjaan | Bukti yang masuk laporan |
|---|---|---|
| 1. MLP classifier | Latih 784 → 256 → 64 → 10 dengan cross-entropy. Simpan W1, W2, W3 dan ekstrak aktivasi laten setelah ReLU pada layer 64. | Akurasi test ≥88%, konfigurasi, dan dimensi bobot. |
| 2. Transpos dari kelas | Mulai dari one-hot kelas, terapkan W3ᵀ → ReLU → W2ᵀ → ReLU → W1ᵀ → sigmoid sesuai panduan. | Grid proyeksi kelas berlabel. Catat bentuk, ketajaman, dan pola yang terlihat. |
| 3. Transpos dari laten | Ambil z dari citra test dan terapkan W2ᵀ → ReLU → W1ᵀ → sigmoid. Gunakan sampel test yang sama untuk gambar pembanding. | Grid citra asli dan proyeksi dari z. |
| 4. Frozen latent decoder | Bekukan seluruh classifier. Latih decoder terpisah 64 → 256 → 784 dengan MSE pada train, pilih checkpoint dari validation. Bandingkan dengan Vanilla AE berlatent 64 pada test yang sama. | MSE, PSNR, SSIM, dan grid rekonstruksi. Bahas detail intra-kelas yang bertahan atau hilang sesuai hasil. |
| 5. Sampling laten MLP | Hitung μ dan σ per dimensi pada fitur test. Sampel dari Gaussian diagonal tersebut dan dekode dengan decoder tahap 4 yang terkunci. | Grid sampel dan observasi tentang bentuk, keragaman, serta batasan pendekatan Gaussian diagonal. |

Untuk teori, jelaskan perbedaan P(y∣x) dan pemodelan distribusi citra. Pada lapisan kompresif, rank W dibatasi oleh jumlah keluaran sehingga WᵀW tidak dapat menjadi identitas pada seluruh ruang input. Transposisi mengikuti formula eksperimen, sementara pemotongan nilai negatif oleh ReLU memberi sumber kehilangan informasi tambahan. Hubungkan absennya regularisasi prior dengan hasil tahap 5. Jelaskan kemampuan rekonstruksi parsial jika terlihat dan batasi klaim sampling pada prosedur yang diuji.

## 4. Kontrak evaluasi

| Pengukuran | Protokol | Model |
|---|---|---|
| MSE, PSNR, SSIM | Pakai seluruh test set, skala piksel [0,1], dan data range 1 untuk PSNR/SSIM. Tetapkan cara agregasi per citra secara konsisten, termasuk penanganan MSE nol. | Vanilla AE, Sparse AE, VAE. Ulangi protokol untuk frozen decoder. |
| FID | Ukur citra sintetis terhadap citra test dengan pipeline fitur identik. Ulang kanal grayscale menjadi tiga untuk extractor RGB dan pakai skala/preprocessing yang sama. Targetkan 7.000 sampel sintetis per model untuk 7.000 citra test. Jika runtime memaksa subset, gunakan satu subset test tetap dan jumlah sintetis yang sama bagi empat model, lalu laporkan N dan batasan estimasinya. | VAE, Vanilla GAN, DCGAN, DDPM. AE deterministik N/A. |
| Latensi batch 64 | Mode evaluasi dan tanpa gradien. Warm-up, sinkronkan CUDA di batas pengukuran, gunakan perangkat/precision/ulangan yang sama, lalu laporkan satu statistik konsisten, misalnya median. Catat ruang lingkup timing serta T untuk DDPM. | Rekonstruksi untuk kedua AE deterministik. Sampling dari noise yang sudah siap untuk VAE, Vanilla GAN, DCGAN, dan DDPM. DDPM mencakup reverse chain penuh. |
| Waktu train per epoch | Catat detik/epoch dengan batas timing seragam dan tulis apakah mencakup validasi atau I/O. Laporkan jumlah epoch serta perangkat. | Enam model. |
| Stabilitas dan mode coverage | Kaitkan bentuk kurva dalam masing-masing model dengan sampel dari beberapa epoch. Amati variasi bentuk dan kemiripan antarsampel. Jika memakai MLP untuk menghitung distribusi label prediksi sebagai pelengkap, catat keterbatasannya pada citra sintetis. | Semua model untuk stabilitas. Empat generator untuk keragaman. |

Konversi Tabel Bagian IV menjadi satu baris per model dengan kolom **dimensi laten, detik/epoch, ms/batch 64, MSE, PSNR, SSIM, FID, kestabilan**. Kolom rekonstruksi Vanilla GAN, DCGAN, dan DDPM diisi N/A. Tambahkan tabel kecil khusus frozen decoder vs Vanilla AE dan akurasi test MLP. Gunakan checkpoint, konfigurasi, dan seed tercatat untuk setiap angka.

## 5. Daftar lima figur wajib

| Figur | Susunan dan kriteria |
|---|---|
| 1. Rekonstruksi | Minimal 8 citra test yang sama pada empat baris. Urutan asli, Vanilla AE, Sparse AE, VAE. |
| 2. Random sampling | Empat panel 4×4, masing-masing minimal 16 sampel. Urutan VAE, Vanilla GAN, DCGAN dari prior Gaussian, lalu DDPM dari x_T Gaussian. Gunakan indeks sampel yang ditetapkan sebelumnya. |
| 3. Eksperimen MLP | Empat panel. Citra asli, transpos dari kelas dan laten, frozen decoder berdampingan dengan Vanilla AE, serta sampling Gaussian diagonal laten MLP. |
| 4. Interpolasi | Lima baris untuk Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN. Enam kolom α = 0, 0.2, 0.4, 0.6, 0.8, 1. Pakai citra test dari dua kelas kontras yang sama untuk AE/VAE. Untuk kedua GAN, pilih dua titik prior dengan keluaran yang mewakili kelas kontras tersebut dan catat seed serta cara pemilihannya. |
| 5. Kurva dan reverse trajectory | Kurva loss enam model, dua garis G/D per GAN, loss noise DDPM, dan urutan x_T → x_0 dari satu reverse chain dengan timestep berlabel. |

Ekspor figur dalam format yang tetap terbaca pada ukuran cetak laporan. Gunakan caption yang menjelaskan konfigurasi atau sampel dan hubungkan pengamatan dengan Q terkait.

## 6. Struktur laporan 10–12 halaman dan bukti untuk Q1–Q6

| Bagian | Alokasi usulan | Isi |
|---|---:|---|
| I. Identitas | 1 hlm | Nama, NIM, kelas, judul, tautan GitHub publik yang aktif. |
| II. Teori dan arsitektur | 2 hlm | Tujuh diagram, loss lengkap, serta konfigurasi lapisan, laten, aktivasi, optimizer, learning rate dan parameter khusus model. |
| III. Pembuktian MLP | 2 hlm | Derivasi ringkas dan hasil lima tahap, termasuk tabel perbandingan decoder. Rujuk Figur 3 dari bagian visual. |
| IV. Komparasi numerik | 1 hlm | Tabel enam model dan keterangan protokol, device dan batasan. |
| V. Figur komparatif | 2–3 hlm | Lima figur bernomor dengan label, caption dan pengamatan utama. |
| VI. Pembahasan | 2–3 hlm | Jawaban Q1–Q6, Generative Trilemma, rekomendasi industri, dan referensi. |

| Pertanyaan | Bukti dan pokok pembahasan |
|---|---|
| Q1 | Gunakan MSE, PSNR, SSIM untuk fidelitas tiga AE/VAE dan FID empat generator untuk kualitas generatif. Kaitkan hasil Sparse AE dengan λ serta tingkat sparsity yang terukur. |
| Q2 | Kaitkan Figur 1–2 dengan keburaman dan ketajaman. Bahas ekspektasi kondisional untuk MSE, Jensen–Shannon pada formulasi minimax GAN ideal, serta cara penalti L1 mengubah aktivasi Sparse AE. |
| Q3 | Gunakan Figur 2 dan 5, FID, dan waktu sampling. Bandingkan dinamika Vanilla GAN/DCGAN, potensi vanishing gradient atau mode collapse, dan prediksi noise DDPM. Bahas kualitas, kecepatan, dan cakupan mode masing-masing. |
| Q4 | Dari Figur 4, bandingkan kontinuitas dan artefak pada enam titik interpolasi. Hubungkan pola yang muncul dengan latent voids dan regularisasi VAE. |
| Q5 | Gunakan Figur 3 dan tabel MLP. Hubungkan hasil dengan objektif P(y∣x), ketidakbalikan kompresi dan ReLU, serta ketiadaan prior laten. |
| Q6 | **Inspeksi manufaktur** bandingkan Vanilla AE vs Sparse AE berdasarkan sensitivitas reconstruction error. Jika sempat, uji keduanya dengan gangguan sintetis yang sama pada citra test dan laporkan sebagai proxy. **Avatar mobile** bandingkan Vanilla GAN/DCGAN/VAE terhadap anggaran 30 ms. Bedakan hasil batch 64 di laptop dari validasi single-image pada perangkat mobile. **Aset visual** bandingkan DCGAN vs DDPM dari FID, detail visual, biaya sampling, dan batas ekstrapolasi hasil 28×28 ke resolusi tinggi. |

## 7. Jadwal kerja enam jam

| Waktu sejak mulai | Fokus | Pencapaian |
|---|---|---|
| 0:00–0:35 | Setup, split, GPU, repo, rangka MLP/DDPM dan pengukuran jalur sampling | Dataset dan skema eksperimen dikunci. Estimasi awal waktu train dan FID tersedia begitu uji singkat berjalan. |
| 0:35–2:30 | Implementasi, uji mini dan jalankan model secara bertahap. Prioritaskan memulai DDPM begitu uji mini lolos. | Pipeline enam model berjalan dan checkpoint pertama tersimpan. MLP dicek dari validasi. |
| 2:30–3:40 | Selesaikan run serta tahap MLP. Siapkan sampling dan ekstraksi fitur dari checkpoint final yang siap. | Lima tahap MLP berjalan dan metrik rekonstruksi serta latensi mulai terkumpul. |
| 3:40–4:30 | Selesaikan FID dan ekspor lima figur | Tabel dan visual dari run aktual tersedia. |
| 4:30–5:40 | Tulis dan rapikan laporan | PDF mencakup Q1–Q6 serta tabel, diagram dan figur. |
| 5:40–6:00 | Audit, buka tautan repo, periksa PDF, lalu unggah | Paket pengumpulan memenuhi format. |

Jadwal ini agresif dan mengasumsikan environment GPU sudah siap serta familiar dengan PyTorch. Gunakan hasil pengukuran pertama untuk penyesuaian. Perkiraan waktu FID DDPM mencakup sekitar 110 batch untuk 7.000 citra pada batch 64, ditambah ekstraksi fitur. Jika melewati anggaran, kecilkan denoiser, sesuaikan anggaran training lewat validasi, atau gunakan subset FID konsisten sesuai protokol. Simpan checkpoint dan lanjutkan run jika crash. Pertahankan enam model, lima tahap MLP, dan lima figur. Jika hasil masih belum stabil atau MLP belum memenuhi ambang, dokumentasikan status aktual dan lanjutkan bila waktu tenggat resmi memungkinkan.

## 8. Checklist sebelum pengumpulan

- [ ] **15% repositori dan reproduktibilitas** GitHub publik, tautan bisa dibuka dari sesi tanpa login, README menjelaskan cara setup dan urutan eksekusi, requirements.txt lengkap, seed/split tercatat, dan riwayat commit bertahap.
- [ ] **25% arsitektur dan training** Enam arsitektur, formulasi loss, konfigurasi dan log riil tercantum. Checkpoint yang dipakai dapat dimuat.
- [ ] **15% MLP** Akurasi test minimal 88%, bobot dan fitur laten tersimpan, semua tahap punya bukti, dan pembanding AE memakai laten 64.
- [ ] **15% metrik** Semua metrik yang berlaku terisi beserta unit, perangkat, jumlah sampel, split, dan catatan kestabilan.
- [ ] **15% visualisasi** Lima figur wajib lengkap. Cek jumlah sampel, susunan panel, resolusi, caption dan label pada PDF akhir.
- [ ] **15% analisis** Q1–Q6 dijawab berdasarkan eksperimen. Bahas hubungan loss–visual, Generative Trilemma, dan tiga skenario industri. Bedakan hasil yang terukur dari dugaan atau keterbatasan.
- [ ] PDF final berjumlah 10–12 halaman. Isi NIM, kelas, dan nama lengkap. Pakai nama resmi `TUGAS_DL_SESI07_[NIM]_[NAMA_LENGKAP].pdf`. Buka PDF hasil ekspor, klik tautan GitHub dari halaman judul, cek tenggat aktual, lalu unggah sesuai petunjuk Binusmaya.
