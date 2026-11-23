# Verifikasi notebook

## Main ringkas

`main.ipynb` disederhanakan dari sekitar **1.206 menjadi 383 baris kode**: 12 cell kode, dengan judul singkat. Model tetap berasal dari `src/models.py` proyek sendiri, bukan implementasi repo contoh. Dataset, split, seed, bentuk layer, dan hyperparameter tidak diubah.

Main kini berfokus pada setup, definisi data/model, inferensi MLP opsional, serta tampilan CSV dan lima gambar tersimpan. **Tidak berisi loop training atau penghitungan ulang FID/figur.** Ini disengaja agar mudah dibaca; kemampuan lengkap dipertahankan di `notebooks/full_pipeline.ipynb`.

### Tes aktual

- Run All di folder sementara yang hanya berisi main notebook: **0 error**; tanpa import `src`, dataset, checkpoint atau hasil repo.
- `RUN_MLP=True` dengan checkpoint tepercaya: fitur/akurasi seluruh **7.000 test**, kedua proyeksi transpose, frozen decoder, pembanding AE dan 16 sampel Gaussian diagonal berhasil; bentuk dan nilai finite diperiksa.
- Akurasi sama dengan hasil MLP asli. Tabel memuat enam model dan kelima PNG tersedia. CSV/gambar diberi label sebagai **hasil tersimpan**, bukan eksperimen baru.
- Tidak training, tidak menghitung FID baru, dan tidak mengubah hasil eksperimen lama.

Bukti: `artifacts/simple_notebook_verification/summary.json` dan salinan executed notebook pada folder yang sama.

## Pipeline lengkap

Versi lengkap sebelumnya, termasuk penyesuaian pengguna yang hanya menghapus pembuka/komentar, disimpan di `notebooks/full_pipeline.ipynb`. Jika dibuka dari subfolder `notebooks`, direktori kerja otomatis diarahkan ke root repo; jika disalin sendiri, tetap dapat dieksekusi tanpa sumber proyek.

Pengujian ulang aktual (termasuk pembukaan dari subfolder `notebooks/`, yang berhasil mendeteksi root repo):

- Mode default notebook-only tanpa berkas repo: **0 error**, tanpa training otomatis.
- Smoke delapan model, dua batch train/validation masing-masing, arsitektur/batch/T400 asli: update bobot, finite loss, bentuk keluaran dan checkpoint lolos.
- Resume AE epoch1→2, update extractor FID (tanpa skor low-N), MLP7000, metrik rekonstruksi7000 dan regenerasi lima figur lolos.
- FID7000 dan training penuh tidak diulang dalam tes ini.
- Fingerprint sebelum/sesudah membuktikan source, konfigurasi dan hasil lama tidak berubah selama eksekusi notebook.

Bukti: `artifacts/full_notebook_verification/summary.json` dan salinan executed notebook pada folder yang sama.

## Pemeriksaan kode

**22 tes proyek lulus.** Tes notebook memeriksa konfigurasi asli, default aman, format notebook, larangan import sumber proyek, serta kesetaraan AST class/fungsi yang disalin dengan kode proyek sendiri. Main disimpan tanpa output pengujian.

Perintah dari root repo:

```text
python -m pytest -q
python tools/check_notebook.py --integration
python tools/check_full_notebook.py --integration
```

Generator terpisah: `tools/build_notebook.py` untuk main ringkas; `tools/build_full_notebook.py` untuk pipeline lengkap. Tidak ada generator yang dijalankan saat Run All notebook.

Catatan bantuan AI tetap akurat pada README/notebook. Laporan dan nama PDF pengguna, kelas LA01, serta penghapusan footer tidak diubah oleh penyederhanaan ini. Seluruh hasil tetap berasal dari run asli, satu seed, dan konfigurasi tercatat; tinjauan mahasiswa masih diperlukan.
