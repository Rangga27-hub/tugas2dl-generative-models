# Verifikasi notebook

`main.ipynb` berisi **12 cell kode, 383 baris, tanpa cell Markdown**. Notebook full pipeline dan generator/pemeriksa khususnya dihapus atas permintaan pengguna. Training, resume, evaluasi dan pembuatan figur tetap tersedia sebagai skrip modular di `src/`.

Notebook utama menyematkan definisi model dan konfigurasi proyek sendiri, tanpa import `src.*`. `RUN_MLP=False` adalah default: tidak mengunduh data atau training. Tabel CSV dan lima gambar yang ditampilkan adalah **hasil eksperimen tersimpan**, bukan hitung ulang.

## Pengujian

- Run All notebook-only di folder terpisah tanpa berkas repo: tidak ada error.
- Inferensi MLP opsional dari checkpoint asli: 7.000 test, transpose kelas/laten, decoder beku, pembanding AE dan 16 sampel Gaussian diagonal.
- Akurasi sama dengan hasil MLP asli; enam baris metrik dan lima gambar tersimpan tersedia.
- Pemeriksaan AST memastikan class/fungsi yang disalin tetap sama dengan source proyek; konfigurasi dan default aman diuji.
- Model, konfigurasi eksperimen, hasil lama, serta PDF laporan pengguna tidak diubah oleh penghapusan cell Markdown.

Perintah verifikasi dari root:

```text
python -m pytest -q
python tools/check_notebook.py --integration
```

Bukti eksekusi ada di `artifacts/simple_notebook_verification/`. Generator `tools/build_notebook.py` menghasilkan notebook code-only dan tidak dipakai saat Run All. Catatan bantuan AI tetap tersedia pada README dan dokumentasi proyek; penghapusan Markdown hanya merapikan notebook.
