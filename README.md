# D6421 Sesi 07 — Studi Komparatif Model Generatif

Fashion-MNIST: Vanilla AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM, serta lima eksperimen MLP. Acuan: [`PRD_Tugas_DL_Sesi07_Generative_Models (1).md`](PRD_Tugas_DL_Sesi07_Generative_Models%20%281%29.md).

**Status:** implementasi dibantu AI. Hasil hanya sah dari checkpoint/log run aktual; keberadaan kode tidak berarti training atau target akurasi sudah tercapai. Lihat `RENCANA_KERJA.md` dan `artifacts/` lokal untuk bukti. NIM, kelas, URL repo publik, serta tenggat perlu dikonfirmasi. Tidak ada hasil/pretrained model yang diklaim sebagai hasil eksperimen sendiri.

## Setup Windows (PowerShell)

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -c "import torch,torchvision; print(torch.__version__,torchvision.__version__,torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
.\.venv\Scripts\python -m pytest -q
```

PyTorch 2.7.1 CUDA 12.8 dipilih untuk RTX 5070 Ti Laptop (Blackwell sm_120); driver dan operasi CUDA tetap harus diuji. Wheel ~3,3 GB. Jangan memasang ke Python global. Untuk CPU/platform lain, pilih wheel yang sesuai dokumentasi PyTorch dan simpan konfigurasi/perangkat berbeda; jangan membandingkan timing lintas perangkat. Di Git Bash gunakan `.venv/Scripts/python`.

## Urutan eksekusi

```powershell
# Unduh data dan persist split
.\.venv\Scripts\python -m src.data
# Dua batch per model, di direktori *_smoke terpisah
.\.venv\Scripts\python -m src.run smoke
# Satu epoch DDPM penuh, reverse chain 64, uji extractor FID
.\.venv\Scripts\python -m src.run pilot
```

**Baca `artifacts/fashion/benchmark_ddpm.json` dahulu.** Estimasi FID mencakup ceil(N/64) reverse chain. Jika budget tidak layak, ubah konfigurasi lalu ulangi pilot. Jangan mengubah konfigurasi run yang sudah memiliki checkpoint; gunakan output_dir baru. Jadwal enam jam bukan jaminan.

```powershell
# Antrean GPU tunggal: MLP, DDPM, AE, Sparse AE, VAE, GAN, DCGAN, decoder
.\.venv\Scripts\python -m src.run train
# Metrik seluruh test, FID default N=7000, semua tahap MLP
.\.venv\Scripts\python -m src.run evaluate
# Lima figur PNG 250 dpi + PDF vektor, provenance dan tensor sumber
.\.venv\Scripts\python -m src.run figures
# Atau setelah pilot disetujui: train + evaluate + figures
.\.venv\Scripts\python -m src.run all-after-pilot
```

Subset FID bila runtime terlalu besar: `python -m src.run evaluate --fid-n 1000`. Gunakan N dan subset identik untuk empat generator, laporkan alasannya dan bias finite-sample (terutama N<2048). File indeks subset tersimpan. Jangan campur hasil N berbeda di tabel. Eksekusi tahap evaluasi tanpa `--fid` pada CLI `src.evaluate` sengaja menghasilkan hasil tanpa FID; gunakan kembali `--fid` untuk hasil lengkap.

### Training satu model / resume

```powershell
.\.venv\Scripts\python -m src.train --model ae
.\.venv\Scripts\python -m src.train --model ae --resume
# Memperpanjang run dengan config sama (catat alasan berdasarkan validation)
.\.venv\Scripts\python -m src.train --model mlp --resume --epochs 35
```

Resume pada batas epoch, termasuk optimizer dan RNG Python/NumPy/torch/CUDA. Run yang sudah ada tidak ditimpa tanpa `--resume`. Hyperparameter dan split harus identik. History checkpoint adalah sumber kebenaran setelah crash; CSV/JSON ditulis ulang pada epoch berikutnya. Jangan memuat checkpoint dari sumber tidak tepercaya karena berisi pickle/RNG state.

## Kontrak eksperimen

- Pool torchvision train resmi 60k lalu test resmi 10k; permutasi NumPy `default_rng(42)`, 56k/7k/7k. Ini **bukan** split benchmark resmi. Indeks dan SHA-256 disimpan; train saja diacak dengan generator `seed+epoch`.
- Semua model melihat split yang sama. Data uint8 di RAM, dikonversi menjadi float32 [0,1] per batch. GAN/DDPM menerima [-1,1], output dikembalikan ke [0,1].
- AE 784–256–64–256–784, ReLU hidden dan sigmoid output. Sparse AE kapasitas sama: MSE mean semua piksel + `lambda * mean_batch(sum_latent(abs(z)))`, lambda 1e-4. Sparsity diukur dengan `abs(z)<0.001` pada test untuk **kedua** AE.
- VAE: BCE sum 784 piksel per citra + KL sum 64 laten per citra, kemudian mean batch, beta=1. Rekonstruksi/interpolasi memakai mu. Nilai objective tidak dibandingkan langsung dengan MSE AE.
- MLP 784–256–64–10, ReLU, CE pada logits; transpose sesuai formula tanpa bias. Decoder terpisah 64–256–784 dilatih dengan seluruh classifier frozen, provenance SHA checkpoint classifier diverifikasi.
- Vanilla GAN noise 100, generator MLP dan discriminator satu logit; BCEWithLogits, G non-saturating. DCGAN: proyeksi 128x7x7 → transposed-conv stride2 64x14x14 → 1x28x28; D stride2 64/128. BatchNorm pada G dan blok kedua D; ReLU G, LeakyReLU D, tanh output.
- DDPM: U-Net ringan dengan residual/time embedding, 28→14→7→14→28; MSE noise. Default 400 langkah, linear beta 0.00025..0.05 (schedule 1000 langkah diskalakan 1000/T), alpha-bar terminal mendekati nol. Sampling ancestral posterior variance, predicted x0 clipped [-1,1]. Bukan DDIM atau accelerated sampling.
- Seleksi validation: MLP max akurasi; AE/Sparse/VAE/decoder min objective; DDPM min fixed-noise val MSE. GAN/DCGAN final epoch sesuai budget yang ditetapkan melalui pilot; test FID **tidak** dipakai memilih epoch. Catat keterbatasan ini, jangan mengklaim GAN optimal.
- Optimizer Adam; LR 1e-3 MLP/AE/VAE/decoder, 2e-4 GAN/DDPM; betas GAN (0.5,0.999), lainnya default (0.9,0.999). Tidak memakai AMP; semua float32.
- Waktu epoch mencakup train+validation dan sinkronisasi, tidak termasuk checkpoint/log/sample I/O. Latensi batch64 median 10 pengukuran setelah 3 warm-up; input/noise awal sudah siap di GPU. DDPM termasuk reverse chain penuh dan random noise tiap langkah.
- MSE/PSNR/SSIM: mean per citra seluruh 7000 test, data_range=1. MSE=0 dicap ke float32 tiny untuk PSNR dan jumlah kasus dicatat. SSIM TorchMetrics Gaussian 11x11 sigma1.5.
- FID: TorchMetrics/torch-fidelity Inception-v3 2048, repeat grayscale→RGB [0,1], `normalize=True`, resize oleh extractor. Membutuhkan unduh bobot extractor. FID ImageNet pada citra grayscale 28x28 memiliki keterbatasan domain dan estimasi sampel; tidak dibandingkan lintas pipeline.
- Statistik test laten MLP **hanya** untuk tahap 5 setelah decoder dikunci; tidak digunakan training, tuning atau FID empat generator.
- Figur interpolasi GAN: 2048 kandidat prior tetap, pilih prediksi MLP kelas 0 vs9 dengan confidence maksimum. Seed/indeks/probabilitas disimpan. Proxy classifier pada sintetis bukan ground truth: verifikasi visual wajib; bila kelas tidak muncul, proses berhenti, tidak memalsukan endpoint.

## Struktur

```text
configs/default.json       desain dan budget awal
src/data.py                dataset dan indeks split
src/models.py              enam arsitektur, MLP, decoder
src/train.py               loss, validation, training, checkpoint/resume
src/benchmark.py           pilot DDPM dan extractor FID
src/evaluate.py            rekonstruksi, latensi, FID, tabel CSV
src/mlp_experiments.py      bobot, fitur test, lima tahap MLP
src/figures.py              lima figur dan provenance
src/run.py                  antrean tahap dan guard pilot
src/report.py               pembangun laporan dari hasil aktual
tests/test_pipeline.py     smoke arsitektur, formula, metrik, split, resume
artifacts/fashion/         lokal; log, best/last.pt, evaluation, figures
```

Checkpoint besar dan dataset tidak masuk Git. Simpan log/figur/tabel terpilih secara eksplisit ke folder `results/` sebelum publikasi, serta jelaskan lokasi unduh checkpoint bila ingin membagikannya. Jangan commit virtualenv/kredensial. Commit hanya pada tahap nyata yang selesai, bukan merekayasa riwayat.

## Sebelum pengumpulan

- Target akurasi MLP >=88% harus **terukur**; jika belum tercapai gunakan validation untuk memperbaiki, jangan tune pada test.
- Periksa collapse/variasi dengan snapshot GAN dan kurva; sistem tidak menyimpulkan stabilitas hanya dari loss.
- Lengkapi NIM/kelas dan URL GitHub publik, buka tanpa login, jangan mempublikasikan data pribadi/kredensial.
- Laporan 10–12 halaman, tujuh diagram, lima figur, Q1–Q6, Generative Trilemma, unit/perangkat/seed/N, batas ekstrapolasi mobile dan resolusi tinggi.
- Jangan sebut draft sebagai laporan final. Batas waktu resmi mengikuti Binusmaya.

## Referensi utama

- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)
- [Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist)
- Kingma & Welling, *Auto-Encoding Variational Bayes*, 2014, https://arxiv.org/abs/1312.6114
- Goodfellow et al., *Generative Adversarial Nets*, 2014, https://arxiv.org/abs/1406.2661
- Radford et al., *Unsupervised Representation Learning with DCGANs*, 2016, https://arxiv.org/abs/1511.06434
- Ho et al., *Denoising Diffusion Probabilistic Models*, 2020, https://arxiv.org/abs/2006.11239
- Heusel et al., *GANs Trained by a Two Time-Scale Update Rule*, 2017, https://arxiv.org/abs/1706.08500
- [TorchMetrics FID](https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html)
