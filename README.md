# Studi Komparatif Generative Models

**Komang Rangga Hariwijaya · 2802493134 · LA01**

Deep Learning D6421 — Sesi 07

Fashion-MNIST: Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM, dan lima eksperimen MLP. Split 56.000/7.000/7.000, seed 42.

## Berkas utama

- **[main.ipynb](main.ipynb)** — versi ringkas: definisi model, MLP opsional, tabel metrik dan lima gambar tersimpan. Tanpa import `src.*`.
- **[Laporan PDF](results/report/TUGAS_DL_SESI07_2802493134_Komang%20Rangga%20Hariwijaya.pdf..pdf)** — laporan 11 halaman; nama file dipertahankan.
- **[Tabel metrik](results/evaluation/comparison.csv)** — hasil test enam model.
- **[Figur](results/figures/)** — lima visualisasi komparatif.

## Menjalankan notebook

Gunakan Python 3.10 dan kernel environment proyek.

```bash
python -m pip install -r requirements.txt
python -m jupyter lab main.ipynb
```

`main.ipynb` memakai **12 cell kode (383 baris)**. Default `RUN_MLP=False`: tidak mengunduh data atau training; tabel/gambar dibaca dari hasil tersimpan. Aktifkan MLP hanya jika checkpoint asli tersedia.

Untuk **training, resume, atau menghitung ulang metrik/figur**, buka [notebook lengkap](notebooks/full_pipeline.ipynb). Seluruh kemampuan sebelumnya tetap tersedia di sana; versi ringkas tidak mengklaim melakukan perhitungan tersebut.

## Hasil singkat

- Akurasi test MLP: **89,76%**.
- FID (7.000 sintetis vs 7.000 test): VAE **84,55**, GAN **147,31**, DCGAN **19,78**, DDPM **28,62**.
- Validasi kode: **22 tes lulus**. Hasil berlaku untuk konfigurasi dan satu seed run ini.

Kode modular tetap tersedia di `src/`; konfigurasi di `configs/`. Implementasi dan penyusunan notebook dibantu AI secara transparan. Tinjauan laporan oleh mahasiswa masih menunggu.
