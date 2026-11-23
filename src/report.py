"""Build an explicitly labeled draft or audited final report from actual results only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib.font_manager as fm
from pypdf import PdfReader
from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                               Spacer, Table, TableStyle)

from .common import load_config, save_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def schematic(title, nodes):
    width = 490
    d = Drawing(width, 60)
    d.add(String(0, 47, title, fontName="Body", fontSize=9))
    gap = 12
    box_width = (width - gap * (len(nodes)-1)) / len(nodes)
    for i, text in enumerate(nodes):
        x = i * (box_width + gap)
        d.add(Rect(x, 8, box_width, 28, rx=3, ry=3, fillColor=colors.HexColor("#edf3fb"), strokeColor=colors.HexColor("#5276a1")))
        d.add(String(x + box_width / 2, 19, text, textAnchor="middle", fontName="Body", fontSize=7))
        if i < len(nodes)-1:
            end = x + box_width + gap - 2
            d.add(Line(x + box_width, 22, end, 22, strokeColor=colors.HexColor("#5276a1")))
            d.add(Polygon([end, 22, end-3, 24, end-3, 20], fillColor=colors.HexColor("#5276a1")))
    return d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--nim")
    parser.add_argument("--kelas")
    parser.add_argument("--repo")
    parser.add_argument("--notes", help="JSON of human-verified visual observations")
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["output_dir"])
    evaluation = {n: read_json(root / "evaluation" / f"{n}.json") for n in ("ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm")}
    mlp = read_json(root / "mlp_experiments" / "results.json")
    data = read_json(root / "data.json")
    provenance = read_json(root / "figures" / "provenance.json")
    interpolation = read_json(root / "figures" / "interpolation.json")
    notes = read_json(args.notes) if args.notes else {}
    required_notes = ("reconstruction", "sampling", "mlp", "interpolation", "training_stability", "ae", "sparse_ae", "vae", "gan", "dcgan", "ddpm")
    if not args.draft:
        if not all((args.nim, args.kelas, args.repo, args.notes)):
            parser.error("Final report needs --nim, --kelas, --repo and --notes; otherwise use --draft")
        if not args.repo.startswith("https://github.com/"):
            parser.error("Use a public GitHub repository URL")
        if any(not str(notes.get(key, "")).strip() for key in required_notes):
            parser.error("Final report requires complete, inspected visual/stability notes")
        if not notes.get("github_public_verified") or not notes.get("gan_endpoints_visually_verified"):
            parser.error("Verify public URL and GAN endpoints before final export")
        if not mlp["target_met"]:
            parser.error("MLP accuracy criterion not met; final export blocked, use draft")
    n_values = set()
    for name, result in evaluation.items():
        if result["split_sha256"] != data["split_sha256"]:
            raise ValueError("Inconsistent split provenance")
        if result["checkpoint"]["sha256"] != provenance["checkpoints"][name]["sha256"]:
            raise ValueError("Figures and metrics use different checkpoints")
        if name in ("vae", "gan", "dcgan", "ddpm"):
            fid = result["fid"]  # Missing actual FID blocks export; never substitute invented values.
            n_values.add((fid["n_real"], fid["n_synthetic"], fid["subset_sha256"]))
        if name in ("ae", "sparse_ae", "vae") and result["reconstruction"]["n_test"] != 7000:
            raise ValueError("Reconstruction must use whole test set")
        if not args.draft and result["checkpoint"]["completed_epochs"] < cfg["epochs"][name]:
            raise ValueError(f"Incomplete training budget: {name}")
    if len(n_values) != 1:
        raise ValueError("FID N/subset must match across four generators")
    n_fid = next(iter(n_values))[0]
    font = fm.findfont("DejaVu Sans")
    bold = fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold"))
    pdfmetrics.registerFont(TTFont("Body", font))
    pdfmetrics.registerFont(TTFont("BodyBold", bold))
    pdfmetrics.registerFontFamily("Body", normal="Body", bold="BodyBold", italic="Body", boldItalic="BodyBold")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TextID", fontName="Body", fontSize=9, leading=13, spaceAfter=8))
    styles.add(ParagraphStyle(name="HeadID", fontName="BodyBold", fontSize=14, leading=18, spaceAfter=12))
    styles.add(ParagraphStyle(name="SmallID", fontName="Body", fontSize=7, leading=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="TitleID", fontName="BodyBold", fontSize=21, leading=28, alignment=TA_CENTER, spaceAfter=20))
    story = []
    def p(text, style="TextID"):
        story.append(Paragraph(text, styles[style]))
    def h(text):
        p(text, "HeadID")
    def page():
        story.append(PageBreak())
    def observation(key):
        value = notes.get(key)
        p(escape(value) if value else "DRAFT: pengamatan visual bagian ini belum diverifikasi. Tidak menarik kesimpulan tentang ketajaman, artefak, atau collapse hanya dari loss.")
    def picture(name, max_height=300):
        path = root / "figures" / f"{name}.png"
        w, height = ImageReader(str(path)).getSize()
        factor = min(490 / w, max_height / height)
        story.append(Image(str(path), width=w*factor, height=height*factor))
        story.append(Spacer(1, 6))
    def table(rows, widths):
        cells = [[Paragraph(escape(str(cell)), styles["SmallID"]) for cell in row] for row in rows]
        t = Table(cells, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), colors.HexColor("#edf3fb")),
                               ("GRID", (0,0), (-1,-1), 0.4, colors.lightgrey),
                               ("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 4),
                               ("RIGHTPADDING", (0,0), (-1,-1), 4)]))
        story.append(t)
        story.append(Spacer(1, 10))
    def metric(name, key):
        return evaluation[name]["reconstruction"][key]
    def f(value, digits=4):
        return f"{value:.{digits}f}"
    labels = {"ae": "Vanilla AE", "sparse_ae": "Sparse AE", "vae": "VAE", "gan": "Vanilla GAN", "dcgan": "DCGAN", "ddpm": "DDPM"}

    # 1 — title
    story.append(Spacer(1, 80))
    p("Studi Komparatif Model Generatif<br/>dan Batasan MLP Classifier", "TitleID")
    p("Deep Learning D6421 · Sesi 07", "TitleID")
    story.append(Spacer(1, 30))
    p("Komang Rangga Hariwijaya")
    p("NIM: " + escape(args.nim or "BELUM DIISI") + "<br/>Kelas: " + escape(args.kelas or "BELUM DIISI"))
    if args.repo:
        p('Repositori: <link href="' + escape(args.repo, {'"': '&quot;'}) + '">' + escape(args.repo) + '</link>')
    else:
        p("Repositori publik: BELUM DIISI / belum diverifikasi")
    p("Benchmark: " + escape(cfg["dataset"]) + "; split bersama 56.000 / 7.000 / 7.000; seed 42.")
    p("Implementasi dibantu AI. Angka dalam laporan berasal dari run aktual dengan checkpoint dan split tercatat. Interpretasi visual wajib diperiksa terhadap figur sumber.")
    if args.draft:
        p("DRAFT — BUKAN BERKAS SIAP KUMPUL. Identitas, publikasi repositori, dan/atau observasi visual dapat belum lengkap.", "HeadID")
    page()

    # 2 — four architectures
    h("II. Teori dan arsitektur: encoder dan MLP")
    for title, nodes in [
        ("1. Vanilla AE", ["784", "256 ReLU", "64 ReLU", "256 ReLU", "784 sigmoid"]),
        ("2. Sparse AE (kapasitas identik)", ["784", "256 ReLU", "64 + L1", "256 ReLU", "784 sigmoid"]),
        ("3. VAE", ["784", "256 ReLU", "mu, logvar:64", "z reparameter", "decoder sigmoid"]),
        ("4. MLP classifier", ["784", "256 ReLU", "64 ReLU", "10 logits", "cross-entropy"]),
    ]:
        story.append(schematic(title, nodes))
    p("AE meminimalkan L = mean[(x - D(E(x)))²]. Sparse AE menambahkan lambda × mean_batch(sum_j |z_j|), dengan lambda=" + str(cfg["sparse_lambda"]) + ". Penalti diterapkan setelah ReLU laten, bukan pada bobot. Karena skala z dan decoder dapat saling mengompensasi, sparsity harus benar-benar diukur; jangan disimpulkan dari nama model.")
    p("VAE: z = mu + exp(0,5 logvar) × epsilon; epsilon ~ N(0,I). L = mean_batch[BCE_sum784 + KL_sum64], KL = -0,5 sum_j(1 + logvar_j - mu_j² - exp(logvar_j)). Ini adalah negative ELBO untuk likelihood Bernoulli pada intensitas berskala [0,1]. Evaluasi rekonstruksi memakai mu, bukan random z.")
    p("MLP meminimalkan cross-entropy pada logits. Tujuan diskriminatif memodelkan P(y|x), bukan P(x). Decoder frozen terpisah memiliki urutan 64 → 256 ReLU → 784 sigmoid dan MSE; tidak memperbarui classifier.")
    p("Adam: LR 0,001 untuk AE/Sparse/VAE/MLP/decoder; batch " + str(cfg["batch_size"]) + ". Beta Adam default (0,9; 0,999). Semua lapisan linear memakai bias; eksperimen transpose secara sengaja mengabaikan bias sesuai formula.")
    page()

    # 3 — remaining three architectures
    h("II. Arsitektur adversarial dan difusi")
    story.append(schematic("5. Vanilla GAN", ["noise100", "G:256/512", "784 tanh", "D:512/256", "1 logit"]))
    story.append(schematic("6. DCGAN", ["noise100", "128x7x7", "64x14x14", "1x28x28", "D:64/128,1"]))
    story.append(schematic("7. DDPM denoiser (skip connections antar-resolusi)", ["x_t,t:28x28", "C=32:28", "64:14", "128:7", "64:14,32:28"]))
    p("GAN minimax ideal: min_G max_D E_x log D(x) + E_z log(1-D(G(z))). Implementasi memakai BCEWithLogits: L_D=BCE(D(x),1)+BCE(D(G(z).detach()),0); L_G=BCE(D(G(z)),1). Objektif G non-saturating menghindari bentuk saturating langsung, tetapi tidak menjamin bebas mode collapse. G Vanilla memakai LeakyReLU; D memakai LeakyReLU. Keluaran G tanh.")
    p("DCGAN memakai ConvTranspose kernel4 stride2 padding1 (7→14→28), BatchNorm/ReLU G; D Conv kernel4 stride2 padding1, LeakyReLU, BatchNorm blok kedua. LR GAN/DCGAN 0,0002 dan Adam betas (0,5;0,999). Satu update D dan G per batch; snapshot fixed-noise disimpan setiap epoch.")
    p(f"DDPM: q(x_t|x_0)=N(sqrt(alpha_bar_t)x_0, (1-alpha_bar_t)I); L=E||epsilon-epsilon_theta(x_t,t)||². T={cfg['ddpm_steps']}, beta linear {cfg['beta_start']}..{cfg['beta_end']}. Denoiser memakai residual Conv3x3, GroupNorm8, SiLU, timestep sinusoidal+MLP, avg-pooling dan nearest upsampling dengan concat skip. LR {cfg['ddpm_lr']}. Sampling ancestral memakai posterior variance dan prediksi x0 yang di-clip [-1,1].")
    p("Input GAN/DDPM diubah dari [0,1] menjadi [-1,1], lalu keluaran kembali ke [0,1]. Semua generator unconditional. Pemilihan epoch GAN/DCGAN menggunakan budget yang ditetapkan sebelum evaluasi test; bukan minimum FID test. Kerugian per model berbeda skala sehingga angka loss mentah bukan ranking kualitas.")
    page()

    # 4 — methodology and derivation
    h("III. Protokol dan pembuktian MLP")
    p("Data train resmi 60.000 dan test resmi 10.000 torchvision digabung dalam urutan tersebut. NumPy default_rng(42).permutation(70000) membentuk train 56.000, validation 7.000, test 7.000. Split bukan benchmark resmi; semua model memakai indeks bersama. Train saja di-shuffle dengan generator seed+epoch. SHA-256 indeks: " + escape(data["split_sha256"]), "SmallID")
    p("Tahap 1 melatih 784→256→64→10 dan menyimpan W1 (256×784), W2 (64×256), W3 (10×64), serta z=ReLU(W2 ReLU(W1x+b1)+b2). Validation memilih checkpoint; test dipakai untuk pelaporan akhir.")
    p("Tahap 2: sigmoid(W1ᵀ ReLU(W2ᵀ ReLU(W3ᵀ onehot(y)))). Tahap 3: sigmoid(W1ᵀ ReLU(W2ᵀ z_test)). Transpose bukan invers. Untuk W berukuran m×n dengan m&lt;n, rank(WᵀW)≤m&lt;n sehingga WᵀW tidak dapat menjadi I_n di seluruh ruang input. ReLU juga memetakan semua nilai negatif menjadi nol, menambah ketidakbalikan. Bias tidak dipakai pada proyeksi balik sesuai formula eksperimen.")
    p("Tahap 4: classifier dibekukan seluruhnya; decoder 64→256→784 dilatih hanya pada train dengan MSE. Checkpoint dipilih dari validation lalu dibandingkan dengan Vanilla AE laten64 pada test yang sama. SHA classifier disimpan dalam checkpoint decoder untuk mencegah pembanding dari encoder berbeda.")
    p("Tahap 5: setelah decoder dikunci, hitung mu_j dan sigma_j populasi dari aktivasi test. Sampel z_j=mu_j+sigma_j epsilon_j secara independen lalu decode. Ini analisis akhir yang memakai statistik test secara eksplisit, bukan generator terlatih dengan prior Gaussian. Tidak digunakan tuning atau FID empat generator. Asumsi diagonal mengabaikan korelasi dan multimodalitas kelas.")
    p("Seleksi checkpoint: max validation accuracy MLP; min validation objective AE/Sparse/VAE/decoder; min fixed-noise validation MSE DDPM. Random draw validation dikunci agar perbandingan antar-epoch konsisten. Best dan last, optimizer, konfigurasi, RNG dan log per epoch disimpan untuk resume.")
    page()

    # 5 — MLP results
    h("III. Hasil lima tahap MLP")
    p(f"Akurasi test MLP: <b>{100*mlp['test_accuracy']:.2f}%</b> dari {mlp['n_test']} citra. Target ≥88%: {'tercapai' if mlp['target_met'] else 'BELUM tercapai'}. Checkpoint classifier dipilih pada epoch {mlp['classifier']['selected_epoch']}.")
    rows = [["Model laten64", "MSE ↓", "PSNR dB ↑", "SSIM ↑"]]
    for label, values in (("Vanilla AE", mlp["vanilla_ae_metrics"]), ("Frozen decoder MLP", mlp["frozen_decoder_metrics"])):
        rows.append([label, f(values["MSE"]), f(values["PSNR"],2), f(values["SSIM"])])
    table(rows, [190, 95, 100, 105])
    picture("figure_3_mlp", 290)
    p("Figur 3. A: 10 test asli; B: proyeksi one-hot kelas 0–9 dan transpose laten test; C: frozen decoder dan Vanilla AE untuk citra yang sama; D: 16 sampel Gaussian diagonal memakai statistik fitur test. Bobot, indeks, fitur dan seed tersedia pada artifacts.pt.", "SmallID")
    observation("mlp")
    page()

    # 6 — quantitative table
    h("IV. Komparasi numerik")
    rows = [["Model / laten", "s/epoch", "ms/64", "MSE", "PSNR", "SSIM", "FID"]]
    for name, result in evaluation.items():
        rec = result.get("reconstruction", {})
        latent = "784 noise" if name == "ddpm" else str(cfg["noise_dim"] if name in ("gan", "dcgan") else cfg["latent_dim"])
        rows.append([labels[name]+" / "+latent, f(result["checkpoint"]["mean_seconds_epoch"],2),
                     f(result["latency"]["median_ms_batch64"],2),
                     *[f(rec[k], 2 if k=="PSNR" else 4) if k in rec else "N/A" for k in ("MSE","PSNR","SSIM")],
                     f(result["fid"]["FID"],2) if "fid" in result else "N/A"])
    table(rows, [111, 60, 60, 67, 64, 64, 64])
    p("Detik/epoch mencakup train+validation, tidak termasuk checkpoint, log dan ekspor sampel. Latensi median batch64 setelah 3 warm-up, 10 ulangan, no-grad/eval, input/noise awal sudah di perangkat dan CUDA disinkronkan di batas timer. DDPM mencakup seluruh reverse chain. Ukuran noise DDPM adalah 1×28×28, bukan kode laten terkompresi.", "SmallID")
    env = evaluation["ae"]["environment"]
    p("Perangkat: " + escape(str(env["gpu"] or env["device"])) + "; Python " + env["python"] + "; torch " + env["torch"] + "; torchvision " + env["torchvision"] + "; CUDA runtime " + str(env["cuda_runtime"]) + "; float32.", "SmallID")
    p(f"Rekonstruksi: seluruh 7.000 test [0,1], mean metrik per citra, PSNR range1 dan clamp MSE nol ke float32 tiny. SSIM Gaussian11 sigma1,5. FID: {n_fid} sintetis vs {n_fid} test, subset identik, grayscale→RGB, TorchMetrics/torch-fidelity Inception-v3 2048, normalize=True. FID finite-sample pada domain Fashion-MNIST memiliki bias; tidak dibandingkan dengan publikasi memakai pipeline lain.", "SmallID")
    rows = [["Model", "Epoch selesai / dipilih", "Kestabilan (kurva + sampel)"]]
    for name, result in evaluation.items():
        info = result["checkpoint"]
        rows.append([labels[name], f"{info['completed_epochs']} / {info['selected_epoch']}", notes.get(name) or "Belum diverifikasi visual (draft)"])
    table(rows, [100, 105, 285])
    page()

    # 7 — visual 1/2
    h("V. Rekonstruksi dan random sampling")
    picture("figure_1_reconstruction", 270)
    p("Figur 1. Delapan citra test yang sama, urutan asli, Vanilla AE, Sparse AE, VAE (mu). Skala tampilan [0,1]; indeks tersedia dalam provenance.json.", "SmallID")
    observation("reconstruction")
    picture("figure_2_sampling", 190)
    p("Figur 2. Empat panel 4×4: VAE, Vanilla GAN, DCGAN dan DDPM. Gaussian prior sesuai dimensi masing-masing; DDPM dimulai dari x_T Gaussian. Indeks sampel 0–15 dan seed telah ditetapkan, bukan dipilih berdasarkan hasil test.", "SmallID")
    observation("sampling")
    page()

    # 8 — visual 4
    h("V. Interpolasi ruang laten")
    picture("figure_4_interpolation", 410)
    p("Figur 4. Lima baris AE, Sparse AE, VAE, Vanilla GAN, DCGAN; enam kolom alpha=0;0,2;0,4;0,6;0,8;1. z(alpha)=(1-alpha)z_A+alpha z_B. Endpoint AE/VAE memakai dua test kelas 0 dan9 yang sama; VAE memakai mu.", "SmallID")
    p("Endpoint GAN dipilih dari 2.048 prior berseed tetap menggunakan confidence MLP untuk kelas0 vs9, bukan label sintetis ground truth. Seleksi ini hanya untuk visual interpolasi, bukan FID. Indeks kandidat dan confidence tersimpan; representasi kelas perlu verifikasi visual.", "SmallID")
    observation("interpolation")
    p("Interpolasi linear yang kontinu pada input tidak menjamin keluaran realistis di sepanjang jalur. AE tidak memaksakan kepadatan prior; VAE mendorong posterior mendekati prior tetapi tidak menghapus semua latent void. Enam titik adalah bukti visual terbatas, bukan uji formal seluruh manifold.")
    page()

    # 9 — visual 5
    h("V. Dinamika loss dan reverse trajectory")
    picture("figure_5_curves_trajectory", 475)
    p("Figur 5. Objective tiap model; G dan D diplot terpisah untuk kedua GAN. VAE menyimpan komponen BCE dan KL. Baris bawah mengikuti satu reverse chain DDPM dari seed tetap, timestep diberi label. Warna dan range [0,1] konsisten; x_T ditampilkan setelah konversi dan clipping untuk visualisasi.", "SmallID")
    observation("training_stability")
    p("Loss rendah tidak otomatis berarti keragaman baik: GAN dapat mengalami collapse walaupun objective tampak stabil. DDPM noise MSE menguji tugas denoising, bukan skor perseptual langsung. Gunakan snapshot antar-epoch dan sampel/FID bersama, jangan bandingkan skala loss lintas-model.")
    page()

    # 10 — Q1–Q3
    h("VI. Pembahasan Q1–Q3")
    best_mse = min(("ae", "sparse_ae", "vae"), key=lambda n: metric(n,"MSE"))
    best_fid = min(("vae", "gan", "dcgan", "ddpm"), key=lambda n: evaluation[n]["fid"]["FID"])
    p(f"<b>Q1 — Fidelitas dan kualitas generatif.</b> MSE terendah pada run ini: {labels[best_mse]} ({metric(best_mse,'MSE'):.5f}); FID terendah: {labels[best_fid]} ({evaluation[best_fid]['fid']['FID']:.2f}, N={n_fid}). Ranking ini dibatasi checkpoint, budget training, satu seed dan feature extractor yang sama. MSE mengukur kesalahan berpasangan; FID mengukur distribusi fitur dan tidak boleh ditafsirkan sebagai error rekonstruksi GAN/DDPM.")
    p(f"Fraksi aktivasi laten |z|&lt;{cfg['sparsity_threshold']}: Vanilla AE={100*evaluation['ae']['reconstruction']['latent_near_zero_fraction']:.2f}%, Sparse AE={100*evaluation['sparse_ae']['reconstruction']['latent_near_zero_fraction']:.2f}% pada lambda={cfg['sparse_lambda']}. Laporkan efek trade-off rekonstruksi dan sparsity berdasarkan angka ini, bukan mengasumsikan penalti selalu meningkatkan kualitas.")
    p("<b>Q2 — Blur, ketajaman, dan objective.</b> Untuk squared error, estimator optimal bersyarat adalah E[x|z]; bila beberapa detail berbeda dipetakan ke kode yang sama, perataan dapat mengaburkan detail. VAE menambahkan kendala KL dan likelihood faktor Bernoulli sehingga trade-off kapasitas/regularisasi juga berperan. Pada GAN minimax ideal dan discriminator optimal, objective berkaitan dengan Jensen–Shannon; implementasi G non-saturating tidak identik secara numerik dengan objective minimax. Penalti L1 Sparse AE menekan aktivasi, bukan langsung menjamin tepian lebih tajam. Klaim visual harus merujuk Figur1–2 yang diperiksa.")
    gan_ms = evaluation["dcgan"]["latency"]["median_ms_batch64"]
    ddpm_ms = evaluation["ddpm"]["latency"]["median_ms_batch64"]
    p(f"<b>Q3 — Adversarial vs difusi.</b> DCGAN memerlukan {gan_ms:.2f} ms/batch64, DDPM {ddpm_ms:.2f} ms/batch64 pada perangkat yang sama; DDPM menjalankan {cfg['ddpm_steps']} denoising steps. CNN memberi bias spasial pada DCGAN, tetapi keunggulan atas GAN MLP harus didukung FID/visual aktual. Non-saturating G mengurangi saturasi objective asli, namun D terlalu kuat dan mode collapse masih mungkin. Noise prediction DDPM memiliki target supervised per timestep; keberhasilan denoising tidak membuktikan cakupan semua mode. Tidak ada estimasi ketidakpastian lintas seed, sehingga selisih kecil FID tidak boleh digeneralisasi.")
    p("<b>Generative Trilemma.</b> Kualitas, cakupan mode, dan kecepatan sampling tidak selalu optimal bersama. GAN satu-pass cepat tetapi rawan ketidakstabilan; VAE prior teratur dan sampling ringan dengan potensi kompromi detail; DDPM memakai sampling iteratif mahal. Di sini coverage dinilai terbatas dari sampel, bukan klaim recall distribusi lengkap.")
    page()

    # 11 — Q4–Q6 & refs
    h("VI. Q4–Q6, batasan dan referensi")
    p("<b>Q4.</b> Figur4 menguji enam titik pada satu pasangan endpoint. Bandingkan kontinuitas bentuk dan artefak berdasarkan observasi terverifikasi, bukan hanya kontinuitas matematis decoder. AE/Sparse tidak memiliki prior eksplisit; VAE KL mendorong keterisian sekitar prior, tanpa menjamin semua jalur realistis. Endpoint GAN dari prediksi MLP dapat bias terhadap classifier dan belum tentu mewakili kelas sebenarnya.")
    ae_mse, decoder_mse = mlp["vanilla_ae_metrics"]["MSE"], mlp["frozen_decoder_metrics"]["MSE"]
    p(f"<b>Q5.</b> MSE frozen decoder={decoder_mse:.5f}, Vanilla AE={ae_mse:.5f}; selisih={decoder_mse-ae_mse:+.5f}. Perbandingan laten64 menguji informasi fitur diskriminatif yang masih dapat didekode, bukan membuktikan semua detail hilang. Rank kompresif, ReLU dan objektif P(y|x) menjelaskan mengapa transpos bukan invers. Gaussian diagonal test tidak dilatih menjadi prior; kualitas tahap5 hanya berlaku pada prosedur tersebut.")
    p("<b>Q6 — Inspeksi manufaktur.</b> AE dan Sparse AE dapat memakai reconstruction error sebagai sinyal anomali, tetapi metrik test citra normal saja tidak mengukur sensitivitas cacat. Tidak ada dataset cacat manufaktur di eksperimen ini; rekomendasi masih hipotesis dan perlu validasi defect/normal serta threshold dari validation. Penalti sparsity dapat membantu representasi atau mengorbankan detail, bergantung lambda.")
    mobile = ", ".join(f"{labels[n]}={evaluation[n]['latency']['median_ms_batch64']:.2f}" for n in ("vae","gan","dcgan"))
    p("<b>Avatar mobile.</b> Latensi laptop batch64 (ms): " + mobile + ". Anggaran 30ms mobile harus diuji single-image end-to-end pada perangkat target; timing laptop atau pembagian waktu batch dengan64 bukan bukti memenuhi SLA mobile. Pertimbangkan memori, energi, transfer dan konversi model.")
    p(f"<b>Aset visual.</b> DCGAN vs DDPM: FID {evaluation['dcgan']['fid']['FID']:.2f} vs {evaluation['ddpm']['fid']['FID']:.2f}. Pilih sesuai kualitas visual terverifikasi dan biaya sampling pada T tercatat; benchmark grayscale28×28 tidak memvalidasi kemampuan avatar/aset RGB resolusi tinggi. Uji ulang domain, lisensi data dan resolusi target diperlukan.")
    p("<b>Batasan.</b> Satu seed, budget berbeda antararsitektur, seleksi GAN final epoch, FID ImageNet pada domain kecil, serta kemungkinan bias classifier sintetis membatasi generalisasi. Catatan visual yang belum lengkap membuat berkas tetap draft.", "SmallID")
    p("<b>Referensi.</b> Kingma &amp; Welling (2014), Auto-Encoding Variational Bayes, arXiv:1312.6114. Goodfellow et al. (2014), GANs, arXiv:1406.2661. Radford et al. (2016), DCGAN, arXiv:1511.06434. Ho et al. (2020), DDPM, arXiv:2006.11239. Heusel et al. (2017), TTUR/FID, arXiv:1706.08500. Xiao et al. (2017), Fashion-MNIST, arXiv:1708.07747. Dokumentasi PyTorch, torchvision, TorchMetrics dan torch-fidelity; tautan di README.", "SmallID")

    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    filename = "DRAFT_Laporan.pdf" if args.draft else f"TUGAS_DL_SESI07_{args.nim}_KOMANG_RANGGA_HARIWIJAYA.pdf"
    destination = out / filename
    def footer(canvas, doc):
        canvas.setFont("Body", 8)
        canvas.setFillColor(colors.grey)
        canvas.drawString(50, 25, "D6421 Sesi 07" + (" | DRAFT" if args.draft else ""))
        canvas.drawRightString(545, 25, str(doc.page))
    doc = SimpleDocTemplate(str(destination), pagesize=(595.28,841.89),
                            leftMargin=50, rightMargin=50, topMargin=42, bottomMargin=42,
                            title="Studi Komparatif Model Generatif", author="Komang Rangga Hariwijaya")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    count = len(PdfReader(destination).pages)
    save_json(out / "audit.json", {"pdf": str(destination), "pages": count, "draft": args.draft,
                                   "page_count_ok": 10 <= count <= 12, "fid_n": n_fid,
                                   "nim_present": bool(args.nim), "kelas_present": bool(args.kelas),
                                   "repo_supplied": bool(args.repo), "visual_notes_supplied": bool(args.notes)})
    if not 10 <= count <= 12:
        raise RuntimeError(f"PDF has {count} pages, outside 10–12; inspect layout before submission")
    print(f"Created {'DRAFT' if args.draft else 'final candidate'} {destination}: {count} pages; inspect rendered PDF and hyperlinks manually")


if __name__ == "__main__":
    main()
