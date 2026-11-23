# Studi Komparatif Generative Models

**Komang Rangga Hariwijaya · 2802493134 · LA01**

Fashion-MNIST: AE, Sparse AE, VAE, GAN, DCGAN, DDPM, dan eksperimen MLP. Split 56.000/7.000/7.000, seed42.

## Berkas

- [main.ipynb](main.ipynb) — 12 cell kode tanpa Markdown: model, MLP opsional, tabel dan gambar hasil.
- [Laporan PDF](results/TUGAS_DL_SESI07_2802493134_Komang%20Rangga%20Hariwijaya.pdf..pdf)
- [Metrik](results/metrics.csv) · [Hasil MLP](results/mlp_metrics.json)
- [Lima figur](results/)

## Menjalankan

Python 3.10, lalu:

```bash
python -m pip install -r requirements.txt
python -m jupyter lab main.ipynb
```

Default `RUN_MLP=False`: tidak mengunduh data atau training. Tabel dan gambar dibaca dari **hasil eksperimen tersimpan**, bukan dihitung ulang. MLP opsional memerlukan checkpoint tepercaya dan indeks split asli.

Akurasi test MLP **89,76%**. FID7000: VAE **84,55**, GAN **147,31**, DCGAN **19,78**, DDPM **28,62**.

Repo ini adalah versi ringkas untuk penyajian, **bukan pipeline lengkap untuk training ulang**. Source training dan bukti lengkap dicadangkan lokal; riwayat Git tetap dipertahankan. Tinjauan mahasiswa masih menunggu.
