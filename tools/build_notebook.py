"""Build main.ipynb with embedded project definitions (no runtime src dependency)."""
from __future__ import annotations

import ast
import hashlib
import json
import pprint
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind, source, tag=None):
    value = {"cell_type": kind, "id": f"cell-{len(cells):03d}", "metadata": {},
             "source": textwrap.dedent(source).strip() + "\n"}
    if tag:
        value["metadata"]["tags"] = [tag]
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    cells.append(value)


def md(source):
    cell("markdown", source)


def code(source, tag=None):
    cell("code", source, tag)


def definitions(module, names):
    source = (ROOT / "src" / f"{module}.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    lines = source.splitlines()
    chunks = []
    for name in names:
        node = found[name]
        start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        chunks.append("\n".join(lines[start-1:node.end_lineno]))
    return "\n\n\n".join(chunks)


md("""
# Studi Model Generatif — D6421 Sesi 07
**Komang Rangga Hariwijaya · NIM 2802493134 · LA01**

Notebook mandiri dari implementasi proyek ini: semua fungsi/class yang dipakai didefinisikan
langsung dalam cell, tanpa import modul proyek atau membaca kode Python eksternal saat runtime.
Penyusunan notebook dan implementasi proyek dibantu AI secara transparan; tinjauan mahasiswa
masih diperlukan. Tidak ada hasil eksperimen yang dibuat-buat.

Urutan: setup → data/split → enam model → training/checkpoint → lima tahap MLP → metrik → lima figur.
**Run All default tidak melakukan training, benchmark, FID ulang, atau regenerasi figur.**
Checkpoint/dataset/hasil tersimpan bersifat opsional; tanpa checkpoint, definisi tetap berjalan
namun inferensi dilewati dengan pesan, bukan diganti bobot acak atau training otomatis.
""")
md("""
## 1. Setup dan konfigurasi
Gunakan kernel Python 3.10 pada environment proyek (VS Code: Select Kernel → `.venv`).
Dari terminal, instal `requirements.txt`, lalu jalankan `python -m jupyter lab main.ipynb`.
Jika hanya memiliki notebook ini, dependensi komputasinya dapat dipasang langsung:
```text
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install numpy==2.2.6 matplotlib==3.10.9 scipy==1.15.3 torchmetrics==1.9.0 torch-fidelity==0.4.0 ipykernel jupyterlab
```
CUDA 12.8 ditujukan untuk environment asli; sesuaikan wheel untuk CPU/platform lain dan jangan
bandingkan timing lintas perangkat. Notebook tidak menjalankan instalasi paket otomatis.
""")
code("""
from __future__ import annotations
import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.utils import make_grid, save_image
from torchmetrics.functional.image import structural_similarity_index_measure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Image as DisplayImage, Markdown, display
""", "imports")
cfg = json.loads((ROOT / "configs/default.json").read_text())
identity = json.loads((ROOT / "configs/identity.json").read_text())
code("DEFAULT_CONFIG = " + pprint.pformat(cfg, sort_dicts=False) + "\n\nIDENTITY = " + pprint.pformat(identity, sort_dicts=False), "embedded-config")
code("""
# Ubah pilihan ini, lalu Restart Kernel + Run All.
MODE = "load"                    # "load", "train", atau "resume"
CONFIRM_TRAINING = False          # wajib True agar train/resume benar-benar berjalan
RUN_PILOT = False                # satu epoch DDPM + sampling64 + smoke extractor FID
RUN_SMOKE = False                # dua batch/model; checkpoint sementara, bukan hasil final
SMOKE_FID_EXTRACTOR = True        # hanya aktif jika RUN_SMOKE=True; mungkin unduh bobot Inception
RECOMPUTE_MLP = False             # jalankan lima tahap MLP dari checkpoint terkunci
RECOMPUTE_METRICS = False         # seluruh 7000 test + latensi; tidak otomatis FID
RECOMPUTE_FID = False             # eksplisit; 7000 vs7000 untuk empat generator
REGENERATE_FIGURES = False        # eksplisit; lima figur dihitung ulang dari checkpoint
DISPLAY_SAVED_RESULTS = True
DOWNLOAD_DATA = True              # dipakai hanya saat data benar-benar diperlukan
CHECKPOINTS_TRUSTED = True        # hanya muat checkpoint milik sendiri/yang dipercaya
RETRAIN_OUTPUT_DIR = "artifacts/notebook_fashion"  # tidak menimpa run asli
NOTEBOOK_OUTPUT_DIR = "artifacts/notebook_outputs"

cfg = copy.deepcopy(DEFAULT_CONFIG)
if MODE not in {"load", "train", "resume"}:
    raise ValueError("MODE harus load/train/resume")
if MODE in {"train", "resume"}:
    if Path(RETRAIN_OUTPUT_DIR).resolve() == Path(DEFAULT_CONFIG["output_dir"]).resolve():
        raise ValueError("Training notebook harus memakai output_dir terpisah dari hasil asli")
    cfg["output_dir"] = RETRAIN_OUTPUT_DIR
OUTPUT = Path(NOTEBOOK_OUTPUT_DIR) / MODE
# Hasil terpublikasi hanya ditampilkan untuk mode load dengan konfigurasi asli.
SAVED_RESULTS = Path("results") if MODE == "load" else Path(cfg["output_dir"])
MODEL_NAMES = ("ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm")
ALL_NAMES = ("mlp", "ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm", "decoder")
print("Mode:", MODE, "| Training diizinkan:", CONFIRM_TRAINING)
print("Checkpoint:", cfg["output_dir"], "| Output notebook:", OUTPUT)
""", "controls")
code(definitions("common", ["save_json", "seed_all", "get_device", "sync", "environment", "rng_state", "restore_rng", "atomic_torch_save", "load_checkpoint"]), "common-definitions")
code("""
seed_all(cfg["seed"])
device = get_device(cfg)
print(json.dumps(environment(device), indent=2))
""")
md("""
## 2. Dataset dan split
Fashion-MNIST: gabungan train resmi 60.000 lalu test resmi 10.000. Permutasi NumPy seed42
membentuk 56.000/7.000/7.000; train saja di-shuffle, batch128. Penyimpanan uint8 dikonversi ke
float32 [0,1] per batch; normalisasi GAN/DDPM dilakukan pada jalur model.
Indeks `.npz` lama dipertahankan untuk kecocokan SHA checkpoint. Jika tersedia, indeks publik
`results/protocol/` disalin byte-identik; bila notebook berdiri sendiri, indeks dibuat dari seed yang sama.
Saat memindahkan checkpoint, sertakan **split .npz asli** agar hash juga cocok.
""")
code(definitions("data", ["DataModule", "to_device"]), "data-definitions")
code("""
checkpoint_root = Path(cfg["output_dir"])
checkpoint_files = [checkpoint_root / name / filename for name in ALL_NAMES for filename in ("best.pt", "last.pt")]
missing_checkpoints = [str(p) for p in checkpoint_files if not p.exists()]
training_requested = MODE in {"train", "resume"} and CONFIRM_TRAINING
load_available = MODE == "load" and CHECKPOINTS_TRUSTED and not missing_checkpoints
needs_data = load_available or training_requested or RUN_SMOKE or RUN_PILOT

data = None
if needs_data:
    split_name = f'{cfg["dataset"]}_split_seed{cfg["seed"]}.npz'
    destination = Path(cfg["data_dir"]) / split_name
    published_split = Path("results/protocol") / split_name
    if not destination.exists() and published_split.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(published_split, destination)
    data = DataModule(cfg, download=DOWNLOAD_DATA)
    print(data.summary())
    batch_x, batch_y, _ = to_device(next(iter(data.loader("validation"))), device)
    assert tuple(batch_x.shape[1:]) == (1, 28, 28)
    assert 0 <= batch_x.min() <= batch_x.max() <= 1
else:
    print("Dataset belum dimuat: tidak ada pekerjaan komputasi yang diaktifkan.")
if MODE == "load" and missing_checkpoints:
    print("Checkpoint tidak lengkap; tidak training otomatis. Salin best.pt + last.pt dan split asli, atau pilih train secara eksplisit.")
    print("Contoh berkas yang belum ada:", missing_checkpoints[:3])
""", "data-action")
md("""
## 3. Enam model
Arsitektur berikut disalin dari proyek, bukan model pengganti. Vanilla/Sparse AE memiliki kapasitas
identik; perbedaan hanya pada objective training. Definisi MLP dan decoder disertakan untuk tahap berikutnya.
""")
md("### 3.1 Vanilla AE dan Sparse AE\n784 → 256 ReLU → 64 ReLU → 256 ReLU → 784 sigmoid.")
code(definitions("models", ["Decoder", "Autoencoder"]), "model-definitions")
md("### 3.2 VAE\nEncoder256 dengan kepala mu/logvar64, reparameterisasi Gaussian, decoder sigmoid.")
code(definitions("models", ["VAE"]), "model-definitions")
md("### 3.3 Vanilla GAN\nPrior Gaussian100; generator dan discriminator MLP; keluaran tanh dan satu logit D.")
code(definitions("models", ["VanillaGAN"]), "model-definitions")
md("### 3.4 DCGAN\nProyeksi128×7×7 → transposed-conv64×14×14 → 1×28×28; D stride2.")
code(definitions("models", ["DCGAN"]), "model-definitions")
md("### 3.5 DDPM\nU-Net ringan 28→14→7→14→28, timestep sinusoidal; konfigurasi aktual T400, beta0,00025..0,05.")
code(definitions("models", ["TimeEmbedding", "TimeBlock", "Denoiser", "DDPM"]), "model-definitions")
md("### 3.6 MLP dan pemilih model\nClassifier784→256→64→10. Proyeksi transpose sengaja tanpa bias sesuai eksperimen.")
code(definitions("models", ["Classifier", "build_model"]), "model-definitions")
md("""
## 4. Training dan checkpoint
AE: MSE mean piksel; Sparse: MSE + lambda mean-batch sum-laten |z|.
VAE: BCE sum784 + KL sum64, mean batch. GAN: BCEWithLogits, G non-saturating.
DDPM: noise MSE; classifier: cross-entropy logits; frozen decoder: MSE.
Adam/LR/betas, pemilihan validation, seed, timing train+validation dan resume RNG sama dengan proyek.
Default hanya memuat checkpoint lokal tepercaya; pickle checkpoint jangan dimuat dari sumber sembarang.
""")
code('NAMES = ALL_NAMES\n\n' + definitions("train", ["objective", "gan_step", "run_epoch", "train_model"]), "training-definitions")
code(definitions("evaluate", ["load_model"]), "checkpoint-definition")
# Pilot uses the same operations as the CLI but receives config/data instead of argparse.
pilot = definitions("benchmark", ["main"])
start = pilot.index('    model = build_model("ddpm", cfg)')
end = pilot.index('    save_json(')
pilot_body = pilot[start:end].replace('args.max_batches', 'max_batches').replace('args.skip_fid', 'skip_fid')
code('def benchmark_ddpm(cfg, data, device, max_batches=None, skip_fid=False):\n' + pilot_body +
     '    save_json(Path(cfg["output_dir"]) / "benchmark_ddpm.json", result)\n    return result\n', "pilot-definition")
md("""
### 4.1 Pilot dan uji mini
Helper sampling/FID minimum diletakkan di sini karena diperlukan pilot dan smoke; penjelasan serta
penghitungan metrik lengkap tetap pada Bagian6. Pilot/tes tidak berjalan tanpa flag eksplisit.
Smoke memakai arsitektur, batch dan T asli, tetapi hanya dua batch train/validation per model.
Hasilnya tidak memenuhi target akurasi dan **bukan** hasil final.
""")
code(definitions("evaluate", ["sample", "make_noise", "build_fid"]), "sampling-definitions")
code("""
def notebook_smoke_test(cfg, data, device, fid_extractor=True):
    parent = Path("artifacts/notebook_smoke")
    parent.mkdir(parents=True, exist_ok=True)
    report = {"purpose": "bounded smoke only, not final experimental metrics", "models": {}}
    with tempfile.TemporaryDirectory(prefix="run_", dir=parent) as temporary:
        smoke_cfg = copy.deepcopy(cfg)
        smoke_cfg["output_dir"] = temporary
        for name in ALL_NAMES:
            seed_all(smoke_cfg["seed"])
            initial = build_model(name, smoke_cfg)
            before = {k: v.clone() for k, v in initial.named_parameters()}
            del initial
            model = train_model(name, smoke_cfg, data=data, epochs=1, max_batches=2)
            changed = any(not torch.equal(before[k], v.detach().cpu()) for k, v in model.named_parameters())
            assert changed, f"{name}: bobot tidak berubah"
            checkpoint = load_checkpoint(Path(temporary) / name / "last.pt")
            assert checkpoint["epoch"] == 1 and checkpoint["max_batches"] == 2
            assert all(math.isfinite(float(v)) for k, v in checkpoint["history"][-1].items())
            model.eval()
            x, _, _ = to_device(next(iter(data.loader("validation", batch_size=4))), device)
            with torch.inference_mode():
                if name in ("ae", "sparse_ae"):
                    output = model(x)[0]
                elif name == "vae":
                    output = model(x, deterministic=True)[0]
                elif name == "mlp":
                    output = model(x)
                    assert model.transpose_class(torch.eye(10, device=device)).shape == (10, 1, 28, 28)
                elif name == "decoder":
                    encoder = build_model("mlp", cfg).to(device).eval().requires_grad_(False)
                    encoder.load_state_dict(load_checkpoint(Path(temporary)/"mlp/best.pt")["model"])
                    output = model(encoder.encode(x))
                else:
                    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
                    output = sample(name, model, make_noise(name, 4, cfg, device, gen), gen)
                assert torch.isfinite(output).all()
                expected = (4, 10) if name == "mlp" else (4, 1, 28, 28)
                assert tuple(output.shape) == expected
            report["models"][name] = {"parameter_update": changed, "finite_loss": True,
                                       "output_shape": list(output.shape), "checkpoint_reload": True}
            del model
        train_model("ae", smoke_cfg, data=data, resume=True, epochs=2, max_batches=2)
        assert load_checkpoint(Path(temporary)/"ae/last.pt")["epoch"] == 2
        report["resume_ae_epoch2"] = True
        if fid_extractor:
            metric = build_fid(device)
            with torch.inference_mode():
                metric.update(x.repeat(1, 3, 1, 1), real=True)
                metric.update(output.repeat(1, 3, 1, 1).clamp(0, 1), real=False)
            report["fid_extractor"] = "real/fake update passed; no low-N FID reported"
    return report

smoke_report = None
if RUN_SMOKE:
    smoke_report = notebook_smoke_test(cfg, data, device, SMOKE_FID_EXTRACTOR)
    display(smoke_report)
if RUN_PILOT:
    pilot_report = benchmark_ddpm(cfg, data, device)
    display(pilot_report)
""", "smoke-action")
code("""
models, metadata = {}, {}
if training_requested:
    if not CHECKPOINTS_TRUSTED:
        raise ValueError("Training decoder/resume juga memuat checkpoint lokal; konfirmasi kepercayaan dahulu")
    pilot_path = Path(cfg["output_dir"]) / "benchmark_ddpm.json"
    if not pilot_path.exists():
        raise RuntimeError("Aktifkan RUN_PILOT dan periksa hasil pilot sebelum full training")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    if (pilot["config"] != cfg or pilot["split_sha256"] != data.split_sha256
        or not pilot["full_epoch_measured"] or not pilot.get("fid_extractor_smoke", {}).get("passed")):
        raise ValueError("Pilot harus full epoch, config/split identik, dan extractor FID lolos")
    if MODE == "train" and any((Path(cfg["output_dir"])/n/"last.pt").exists() for n in ALL_NAMES):
        raise FileExistsError("Output training sudah berisi checkpoint; pilih resume atau folder run baru")
    for name in ("mlp", "ddpm", "ae", "sparse_ae", "vae", "gan", "dcgan", "decoder"):
        last = Path(cfg["output_dir"]) / name / "last.pt"
        train_model(name, cfg, data=data, resume=MODE == "resume" and last.exists())
    load_available = True
elif MODE in {"train", "resume"}:
    print("Training/resume dilewati: CONFIRM_TRAINING masih False.")

if load_available:
    for name in ALL_NAMES:
        models[name], metadata[name] = load_model(name, cfg, device, data)
    print("Checkpoint tervalidasi:", {n: m["selected_epoch"] for n, m in metadata.items()})
else:
    print("Belum ada model terlatih di memori; tidak ada fallback ke bobot acak.")
""", "training-action")
md("""
## 5. Lima eksperimen MLP
1. Classifier dan akurasi test, W1/W2/W3 serta aktivasi ReLU64.
2. One-hot kelas → W3ᵀ → ReLU → W2ᵀ → ReLU → W1ᵀ → sigmoid.
3. Fitur test → W2ᵀ → ReLU → W1ᵀ → sigmoid.
4. Decoder64→256→784 memakai classifier beku, pembanding Vanilla AE laten64.
5. Gaussian diagonal dengan mean/std **populasi** fitur test; hanya analisis akhir setelah decoder terkunci.

Pengumpulan fitur dan sampling di bawah mengikuti kode proyek. MSE/PSNR/SSIM pembanding dihitung
pada Bagian6 dengan rumus yang sama; angka tersimpan lama tidak diklaim sebagai hitung ulang.
""")
# Reuse the actual five-stage code, parameterized to avoid reloading data/models or overwriting original artifacts.
mlp_source = definitions("mlp_experiments", ["run"])
start = mlp_source.index('    codes, labels, ids = [], [], []')
end = mlp_source.index('    result = {')
mlp_body = mlp_source[start:end]
code('@torch.inference_mode()\ndef collect_mlp_artifacts(cfg, data, device, models, metadata, output_root):\n'
     '    seed_all(cfg["seed"])\n    mlp, decoder, ae = (models[n] for n in ("mlp", "decoder", "ae"))\n'
     '    output = Path(output_root) / "mlp_experiments"\n    output.mkdir(parents=True, exist_ok=True)\n' + mlp_body +
     '    result = {"test_accuracy": correct/total, "target_accuracy": 0.88, "target_met": correct/total >= 0.88,\n'
     '              "n_test": total, "weight_shapes": {k: list(artifacts[k].shape) for k in ("W1", "W2", "W3")},\n'
     '              "classifier": metadata["mlp"], "decoder": metadata["decoder"], "vanilla_ae": metadata["ae"],\n'
     '              "split_sha256": data.split_sha256, "sampling_seed": cfg["seed"]+14000,\n'
     '              "figure_test_indices": ids[:10].tolist(), "transpose_bias": "not used as specified",\n'
     '              "sampling_distribution": "diagonal Gaussian of test ReLU features; population std; locked decoder"}\n'
     '    save_json(output / "results.json", result)\n    return artifacts, result\n', "mlp-definition")
code("""
mlp, mlp_results = None, None
if models and (RECOMPUTE_MLP or REGENERATE_FIGURES or RECOMPUTE_METRICS):
    mlp, mlp_results = collect_mlp_artifacts(cfg, data, device, models, metadata, OUTPUT)
    print("Akurasi test MLP:", mlp_results["test_accuracy"], "| Target tercapai:", mlp_results["target_met"])
elif DISPLAY_SAVED_RESULTS:
    saved_mlp = SAVED_RESULTS / "mlp_experiments/results.json"
    if saved_mlp.exists():
        display(Markdown(f"**Hasil MLP tersimpan, bukan hitung ulang:** `{saved_mlp}`"))
        display(json.loads(saved_mlp.read_text(encoding="utf-8")))
    else:
        print("Eksperimen MLP belum dijalankan; aktifkan RECOMPUTE_MLP setelah checkpoint lengkap.")
""", "mlp-action")
md("""
## 6. Evaluasi metrik
MSE/PSNR/SSIM: mean per citra seluruh7000 test [0,1], SSIM Gaussian11 sigma1,5,
PSNR MSE-nol dicap float32 tiny. FID: Inception-v3 2048 torch-fidelity, grayscale diulang RGB,
normalize=True; **7000 sintetis vs7000 test** untuk empat generator, subset identik.
Latensi: batch64, eval/no-grad, tiga warm-up, median10 ulangan, sinkronisasi CUDA; DDPM termasuk
reverse chain400 langkah. FID/latensi berat hanya aktif lewat flag eksplisit.
Timing/FID yang dihitung ulang disimpan terpisah, tidak menimpa angka publikasi asli.
""")
code('GENERATORS = ("vae", "gan", "dcgan", "ddpm")\nRECONSTRUCTORS = ("ae", "sparse_ae", "vae", "decoder")\n\n' +
     definitions("evaluate", ["reconstruct", "reconstruction_scores", "reconstruction_metrics", "latency", "write_table"]), "metric-definitions")
fid = definitions("evaluate", ["fid_metrics"])
fid = fid.replace('device, n):', 'device, n, output_root=None):').replace('output = Path(cfg["output_dir"]) / "evaluation"', 'output = Path(output_root or cfg["output_dir"]) / "evaluation"')
code(fid, "fid-definition")
code("""
evaluation = {}
if models and (RECOMPUTE_METRICS or RECOMPUTE_FID):
    seed_all(cfg["seed"])
    x, _, _ = to_device(next(iter(data.loader("test"))), device)
    for name in MODEL_NAMES:
        value = {"checkpoint": metadata[name], "environment": environment(device), "split_sha256": data.split_sha256}
        if RECOMPUTE_METRICS:
            if name in RECONSTRUCTORS:
                value["reconstruction"] = reconstruction_metrics(name, models[name], data, device, cfg)
            value["latency"] = latency(name, models[name], cfg, device, x)
        if RECOMPUTE_FID and name in GENERATORS:
            value["fid"] = fid_metrics(name, models[name], data, cfg, device, cfg["fid_samples"], output_root=OUTPUT)
        evaluation[name] = value
        save_json(OUTPUT / "evaluation" / f"{name}.json", value)
    if RECOMPUTE_METRICS:
        if mlp_results is not None:
            mlp_results["vanilla_ae_metrics"] = evaluation["ae"]["reconstruction"]
            mlp_results["frozen_decoder_metrics"] = reconstruction_metrics("decoder", models["decoder"], data, device, cfg, models["mlp"])
            save_json(OUTPUT / "mlp_experiments/results.json", mlp_results)
            display(mlp_results)
        table_cfg = copy.deepcopy(cfg)
        table_cfg["output_dir"] = str(OUTPUT)
        write_table(table_cfg)
    if RECOMPUTE_FID:
        protocols = {(v["fid"]["n_real"], v["fid"]["n_synthetic"], v["fid"]["subset_sha256"])
                     for v in evaluation.values() if "fid" in v}
        assert len(protocols) == 1
    display(evaluation)
elif not models and (RECOMPUTE_METRICS or RECOMPUTE_FID):
    print("Metrik dilewati: perlu checkpoint tervalidasi, tidak menggunakan bobot acak.")

if DISPLAY_SAVED_RESULTS and not (RECOMPUTE_METRICS or RECOMPUTE_FID):
    table = SAVED_RESULTS / "evaluation/comparison.csv"
    if table.exists():
        print("Tabel tersimpan; bukan evaluasi baru:", table)
        with table.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        columns = ("model", "latent_dim", "seconds_epoch", "ms_batch64", "MSE", "PSNR", "SSIM", "FID", "FID_N")
        header = "| " + " | ".join(columns) + " |"
        divider = "| " + " | ".join("---" for _ in columns) + " |"
        lines = ["| " + " | ".join(str(row[k]) for k in columns) + " |" for row in rows]
        display(Markdown("\\n".join([header, divider, *lines])))
    else:
        print("Belum ada tabel metrik tersimpan. Aktifkan evaluasi setelah checkpoint tersedia.")
""", "metric-action")
md("""
## 7. Lima visualisasi wajib
Default menampilkan PNG hasil proyek jika tersedia; bukan sampling baru. Untuk menghasilkan ulang,
aktifkan REGENERATE_FIGURES setelah checkpoint lengkap. Fungsi plotting, seed, indeks sampel,
endpoint dan trajectory disalin dari kode proyek. Sumber perhitungan disimpan ke output notebook.
""")
code('LABELS = {"ae": "Vanilla AE", "sparse_ae": "Sparse AE", "vae": "VAE", "gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM"}\n\n' +
     definitions("figures", ["image", "grid", "export"]), "plot-definitions")
code("""
VISUALS_READY = bool(models) and REGENERATE_FIGURES and mlp is not None
root, out = Path(cfg["output_dir"]), OUTPUT / "figures"
names = MODEL_NAMES
classifier = models.get("mlp")
if VISUALS_READY:
    seed_all(cfg["seed"])
    out.mkdir(parents=True, exist_ok=True)

def show_figure(filename):
    path = out / f"{filename}.png" if VISUALS_READY else SAVED_RESULTS / "figures" / f"{filename}.png"
    if path.exists() and (VISUALS_READY or DISPLAY_SAVED_RESULTS):
        print("Dihitung ulang:" if VISUALS_READY else "Figur tersimpan, bukan hitung ulang:", path)
        display(DisplayImage(filename=str(path)))
    else:
        print("Figur belum tersedia:", filename, "— lengkapi checkpoint dan aktifkan REGENERATE_FIGURES.")
""", "plot-action-setup")
fig_source = definitions("figures", ["main"])
markers = ['    selected = data.indices["test"][:8]', '    samples = {}',
           '    fig, axes = plt.subplots(2, 2, figsize=(10, 8))', '    # Same true test endpoints',
           '    fig = plt.figure(figsize=(12, 10))', '    print(f"Figures exported']
sections = [fig_source[fig_source.index(markers[i]):fig_source.index(markers[i+1])] for i in range(5)]
figure_names = ["figure_1_reconstruction", "figure_2_sampling", "figure_3_mlp", "figure_4_interpolation", "figure_5_curves_trajectory"]
headings = ["Rekonstruksi", "Random sampling", "Eksperimen MLP", "Interpolasi", "Kurva dan reverse trajectory"]
for i, (section, filename, title) in enumerate(zip(sections, figure_names, headings), 1):
    md(f"### 7.{i} {title}")
    # Same calculations, but inference_mode replaces the original function decorator.
    body = textwrap.dedent(section).strip()
    code("if VISUALS_READY:\n    with torch.inference_mode():\n" + textwrap.indent(body, "        ") +
         f'\nshow_figure("{filename}")\n', f"figure-{i}")
md("""
## Catatan akhir
- Checkpoint/split/seed sama tidak menjamin bitwise lintas perangkat atau versi pustaka.
- Jangan menyebut smoke sebagai training penuh atau angka tersimpan sebagai penghitungan baru.
- Statistik laten test hanya untuk tahap5; bukan tuning, pemilihan checkpoint, atau FID generator.
- FID ImageNet pada grayscale28×28, satu seed, dan budget training berbeda membatasi generalisasi.
- Catatan bantuan AI dan sumber modular tetap dipertahankan; notebook merupakan penyusunan ulang,
  bukan klaim bahwa implementasi sebelumnya tanpa bantuan. Kelas LA01; tinjauan mahasiswa masih menunggu.
- Notebook tidak mencakup pembangun PDF/publikasi GitHub: tujuh bagian eksperimen di atas berdiri sendiri.
""")
notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
             "language_info": {"name": "python", "version": "3.10.9"},
             "project_source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted((ROOT / "src").glob("*.py"))},
             "description": "Self-contained inline project code; default load-only, no implicit full training"},
             "nbformat": 4, "nbformat_minor": 5}
(ROOT / "main.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote main.ipynb: {len(cells)} cells")
