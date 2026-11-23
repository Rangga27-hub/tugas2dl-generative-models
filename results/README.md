# Bukti run Fashion-MNIST — D6421 Sesi 07

**Status: eksperimen selesai; laporan DRAFT, belum siap dikumpulkan.**

## Berkas utama

- [Laporan draft 11 halaman](report/DRAFT_Laporan.pdf): tujuh diagram, formulasi loss, lima tahap MLP, lima figur dan Q1–Q6.
- [Tabel enam model](evaluation/comparison.csv).
- [Hasil MLP dan pembanding decoder](mlp_experiments/results.json).
- [Lima figur PNG/PDF](figures/), termasuk provenance dan pemilihan endpoint interpolasi.
- [Audit](audit.json), [manifest SHA-256](manifest.json), [konfigurasi](protocol/config.json), indeks split dan dependensi aktual di `protocol/`.
- `ae/`, `sparse_ae/`, `vae/`, `gan/`, `dcgan/`, `ddpm/`, `mlp/`, `decoder/`: history/config/environment aktual. Snapshot GAN epoch 1/20/40 ikut disertakan.

## Ringkasan angka aktual

| Model | Epoch | s/epoch | ms/batch64 | MSE | PSNR dB | SSIM | FID |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla AE | 20 | 1,81 | 0,33 | 0,00851 | 21,72 | 0,8277 | N/A |
| Sparse AE | 20 | 2,01 | 0,34 | 0,00993 | 20,98 | 0,8010 | N/A |
| VAE | 25 | 2,21 | 0,23 | 0,01317 | 19,66 | 0,7654 | 84,55 |
| Vanilla GAN | 40 | 2,95 | 0,31 | N/A | N/A | N/A | 147,31 |
| DCGAN | 40 | 3,61 | 0,63 | N/A | N/A | N/A | 19,78 |
| DDPM | 30 | 8,32 | 1343,88 | N/A | N/A | N/A | 28,62 |

- MLP test **89,7571%**, 7.000 citra; checkpoint validation epoch24 dari 25 epoch.
- Frozen decoder MLP: MSE **0,01816**, PSNR **18,25 dB**, SSIM **0,7028**, dibanding Vanilla AE laten64 di atas.
- Aktivasi laten `|z|<0,001`: AE **31,47%**, Sparse AE **54,16%**, lambda=0,0001.
- Rekonstruksi seluruh 7.000 test [0,1]; FID **7.000 sintetis vs 7.000 test** dengan subset/pipeline identik.
- RTX 5070 Ti Laptop, Python3.10.9, torch2.7.1+cu128, float32. Latensi median10 ulangan setelah3 warm-up; DDPM400 langkah reverse penuh. s/epoch termasuk validation, tidak termasuk checkpoint/log/sample I/O.
- **18 tes lulus**, tanpa failure/error/skipped. Audit otomatis lolos konsistensi checkpoint, split, FID N/subset, sumber figur dan jumlah halaman. PDF dirender dan diperiksa oleh asisten AI.

Interpretasi dibatasi satu seed dan budget ini: DCGAN memiliki FID terbaik pada run ini; DDPM tidak otomatis lebih baik. MLP masih menyimpan informasi yang dapat didekode sebagian, tetapi transpos bukan invers dan sampling Gaussian diagonal tidak menjadi prior teratur. Catatan visual dibantu AI disimpan di [visual_notes.json](visual_notes.json); mahasiswa perlu menyetujui dan memeriksanya sendiri sebelum pengumpulan.

## Belum selesai / perlu pengguna

1. Isi **NIM dan kelas**; konfirmasi tenggat resmi.
2. Buat/konfirmasi **GitHub publik**, pasang URL aktif di halaman judul, verifikasi tanpa login. Belum ada publikasi atau remote yang diklaim.
3. Periksa endpoint GAN dan interpretasi; flag verifikasi sengaja belum diisi `true` tanpa persetujuan pengguna.
4. Ekspor kandidat final dengan nama resmi setelah identitas/verifikasi lengkap, lalu periksa PDF dan unggah sendiri.

Checkpoint, bobot W1/W2/W3, fitur test, statistik Gaussian dan tensor sumber tersimpan **lokal** di `artifacts/fashion/`, bukan dalam paket Git ini. Jalankan pipeline pada README utama untuk mereproduksi atau salin checkpoint dari mesin asal secara terpisah. Path Windows di metadata adalah provenance run asal, bukan tautan unduh.

Manifest memuat hash byte aktual evidence; `.gitattributes` menonaktifkan normalisasi newline untuk `results/` agar hash tidak berubah ketika clone lintas platform. Fixture sintetis pytest hanya untuk tes kode/layout dan tidak dimasukkan sebagai hasil eksperimen.
