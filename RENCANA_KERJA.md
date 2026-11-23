# Rencana eksekusi — PRD revisi

Permintaan terbaru: “prd nya saya ubah, kerjakan sesuai prd yg baru saja, yuk gas”.
Acuan: `PRD_Tugas_DL_Sesi07_Generative_Models (1).md`. Dokumen ini menggantikan rencana lama yang merujuk pembatasan AI pada PRD sebelumnya. Implementasi dibantu AI; angka eksperimen hanya berasal dari run aktual.

## Keputusan
- Solo implementasi dan integrasi; tidak memakai subagent tersembunyi.
- Fashion-MNIST, gabungan train resmi dan test resmi sebelum split acak 56.000/7.000/7.000; seed 42 dan indeks disimpan.
- Python 3.10 virtualenv; PyTorch 2.7.1 + CUDA 12.8 dipilih untuk GPU Blackwell sm_120. Kompatibilitas wajib diuji, tidak disimpulkan dari nvidia-smi.
- Float32; batch 128; enam model generatif, MLP dan frozen decoder.
- VAE: BCE sum piksel per citra + KL sum laten per citra, dirata-rata batch. Rekonstruksi evaluasi dari mu.
- Sparse AE: MSE mean piksel + lambda mean-batch sum-laten |z|, lambda awal 1e-4.
- DDPM: schedule linear beta 0,00025..0,05, 400 langkah (schedule 1.000 langkah 1e-4..0,02 diskalakan 1.000/T); alpha-bar terminal diperiksa mendekati nol. Tentukan anggaran epoch berdasarkan runtime aktual.
- FID default 7.000; subset tetap hanya bila runtime mengharuskan dan jumlahnya sama bagi empat generator.

## Tahap dan bukti penerimaan
- [x] Baca PRD revisi; audit awal hardware dan Python.
- [ ] Environment CUDA lolos forward/backward dan dependensi tercatat.
- [ ] Split tersimpan dan terbukti disjoint serta reproduksibel.
- [ ] Uji mini semua model: bentuk, finite loss, update bobot, checkpoint/resume.
- [ ] Benchmark DDPM satu epoch + sampling batch 64 + extractor FID.
- [ ] Training enam model, MLP, frozen decoder; log dan best/last checkpoint.
- [ ] Lima tahap MLP, termasuk bobot, fitur test dan statistik Gaussian diagonal.
- [ ] Rekonstruksi seluruh test; FID, timing seragam, sparsity AE.
- [ ] Lima figur wajib, tujuh diagram, tabel berbasis checkpoint yang dipilih validation.
- [ ] Laporan 10–12 halaman, Q1–Q6, audit PDF.
- [ ] Identitas dan repo publik dikonfirmasi pengguna; verifikasi tautan tanpa login.

## Risiko / batasan
- Instalasi wheel CUDA besar (~3,3 GB) masih perlu selesai dan diuji.
- Target >=88% MLP adalah kriteria, bukan hasil yang dijanjikan.
- GAN dipilih pada epoch terakhir sesuai budget yang ditetapkan melalui pilot; tidak memilih berdasarkan test/FID final. Evaluasi generator secara terpisah dari rekonstruksi.
- Resume deterministik pada batas epoch; log per-epoch, optimizer dan RNG disimpan. Waktu crash di tengah epoch tidak dihitung sebagai epoch selesai.
- NIM, kelas, tenggat resmi, dan URL GitHub belum tersedia. Tidak membuat identitas/tautan palsu atau melakukan publikasi tanpa konfirmasi.
- PDF final dan angka tidak dinyatakan selesai sebelum evidence benar-benar tersedia.
