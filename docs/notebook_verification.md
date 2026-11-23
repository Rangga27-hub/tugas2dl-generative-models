# Verifikasi `main.ipynb`

## Ruang lingkup

Notebook menyematkan konfigurasi eksperimen asli dan definisi class/fungsi dari proyek. Tidak ada import modul proyek, `%run`, `exec` kode sumber, atau pembacaan `src/` saat runtime. Generator `tools/build_notebook.py` hanya alat pengembangan untuk menyusun cell; tidak diperlukan oleh notebook yang sudah dibuat.

Urutan 52 cell: setup/config, data/split, enam arsitektur beserta classifier/decoder pendukung, training/checkpoint, eksperimen MLP, evaluasi, dan lima figur terpisah. Helper sampling/FID minimum muncul sebelum pilot karena diperlukan untuk uji runtime. Metrik lengkap tetap dijelaskan dan dihitung pada bagian evaluasi.

Catatan bantuan AI pada repo dan notebook dipertahankan. Penyusunan notebook mempertahankan kode model, konfigurasi eksperimen, riwayat commit, dan angka hasil lama. Pembaruan laporan berikutnya mengisi kelas LA01 dan menghapus footer kiri bawah atas permintaan pengguna; nama PDF pilihan pengguna tetap dipertahankan.

## Pengujian yang benar-benar dijalankan

Perintah pada environment Python3.10/PyTorch2.7.1+cu128 proyek:

```text
python -m pytest tests/test_notebook.py -q
python tools/check_notebook.py --integration
```

### 1. Notebook-only, default aman

- Notebook disalin ke direktori sementara tanpa `src/`, `configs/`, `results/`, dataset, atau checkpoint repo.
- Seluruh cell dijalankan berurutan melalui kernel Jupyter bersih: **32 code cell** termasuk guard pengujian, **0 error**.
- Import `src` diblokir dengan import hook dalam salinan pengujian, sehingga dependensi tersembunyi tidak dapat lolos.
- Tidak ada training/pilot/FID/regenerasi figur otomatis. Cell yang memerlukan data/checkpoint memberi pesan dan dilewati; tidak mengklaim inferensi dengan model terlatih.
- Artinya kode notebook mandiri, bukan berarti dataset, pustaka, atau bobot terlatih berada di dalam file notebook.

### 2. Smoke dan integrasi checkpoint lokal

Salinan pengujian memakai `RUN_SMOKE=True`, `RECOMPUTE_METRICS=True`, `REGENERATE_FIGURES=True`. Notebook utama tetap menyimpan default `False`.

- **33 code cell** termasuk guard dan assertion pengujian, **0 error**.
- Delapan model (AE, Sparse AE, VAE, Vanilla GAN, DCGAN, DDPM, MLP, decoder): dua batch train dan dua batch validation per model; loss finite, parameter berubah, bentuk keluaran benar, checkpoint dapat dibaca kembali.
- Arsitektur, batch128, laten64/prior100, T400 dan noise schedule asli dipertahankan. Smoke hanya membatasi batch/epoch, bukan mengganti model final.
- AE smoke berhasil resume dari epoch1 ke epoch2 beserta optimizer/RNG.
- Extractor FID menerima update real/fake pada empat citra grayscale yang diulang RGB. **Tidak menghitung atau melaporkan FID low-N**.
- Delapan checkpoint asli berhasil dimuat dengan pemeriksaan konfigurasi/split; classifier frozen terikat hash yang benar.
- Fitur/akurasi MLP dan Gaussian diagonal dihitung dari seluruh **7.000 test**, setelah decoder terkunci.
- MSE/PSNR/SSIM tiga AE/VAE dan frozen decoder dihitung pada seluruh **7.000 test**. Latensi enam model memakai warm-up/ulangan dan reverse chain penuh yang sama.
- Lima figur dihitung ulang dari checkpoint dan disimpan terpisah; interpolasi memakai prosedur kandidat/seed asli. Pemilihan kelas proxy tetap membutuhkan tinjauan pengguna.
- Identitas citra untuk metrik diuji: MSE nol, SSIM mendekati1 dan PSNR finite sesuai kebijakan clamp nol asli.
- Tidak melakukan training penuh atau FID penuh 7.000 sampel ulang dalam pengujian notebook ini. FID lama tidak diklaim sebagai hasil baru.

### 3. Pemeriksaan statis dan preservasi

Empat tes notebook lulus: validasi format/AST tanpa import proyek, konfigurasi inline identik, default flag aman, kesetaraan AST model/training/data/helper/metrik yang disalin, urutan bagian dan lima cell figur. Satu perubahan pada helper FID hanya menambahkan tujuan output opsional; rumus/seed/preprocessing tetap sama. Pengumpulan MLP dipisah dari metrik pembanding agar urutan notebook mudah dibaca, tanpa mengubah operasi laten/sampling.

Regresi seluruh proyek juga dijalankan: **22 tes lulus** (18 tes proyek yang sudah ada + 4 tes notebook), tanpa failure/error. Hasil terbaru disimpan terpisah pada `artifacts/notebook_verification/tests.xml` dan `artifacts/setup/notebook_tests.log`, tidak menimpa bukti pengujian eksperimen lama.

Fingerprint byte sebelum/sesudah eksekusi memastikan **seluruh `src/`, `configs/`, `results/` dan `artifacts/fashion/` tidak berubah** selama pengujian. Output notebook dihitung ulang ke `artifacts/notebook_outputs/load/`, smoke checkpoint sementara dibersihkan.

## Bukti lokal

- `artifacts/notebook_verification/summary.json`
- `artifacts/notebook_verification/standalone.executed.ipynb`
- `artifacts/notebook_verification/integration.executed.ipynb`
- `artifacts/setup/notebook_check.log`

Salinan executed untuk verifikasi sengaja berada di `artifacts/` (diabaikan Git). `main.ipynb` tanpa output tersimpan agar default aman tetap jelas dan tidak mencampur output pengujian dengan eksperimen baru. Warning event loop/transport Jupyter Windows muncul selama tes tetapi tidak menghasilkan error cell; kernel pengujian berjalan lokal.

## Yang masih memerlukan tindakan eksplisit

Training ulang/resume penuh memerlukan mode dan konfirmasi, pilot yang sesuai, dan direktori run terpisah. FID penuh memerlukan flag sendiri, kemungkinan unduhan bobot Inception, dan runtime sampling DDPM. Tinjauan interpretasi tidak diisi oleh pengujian ini; kelas LA01 diisi berdasarkan konfirmasi pengguna, bukan dugaan.
