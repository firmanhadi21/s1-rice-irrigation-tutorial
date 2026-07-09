# Promotion kit — X / Twitter

Ready-to-use posts, visuals, alt-text, and tactics for promoting the
**Sentinel-1 Rice & Irrigation Monitoring** tutorial on X (and re-usable on LinkedIn).

- **Tutorial (EN + ID):** https://firmanhadi21.github.io/s1-rice-irrigation-tutorial/
- **Code:** https://github.com/firmanhadi21/s1-rice-irrigation-tutorial

## Why this resonates (the hooks)

- **Free & laptop-friendly** — Sentinel-1 is open data; models run on CPU, no GPU.
- **Reproducible** — a trained model + sample AOI ship with the repo; one clone and you run.
- **Open-source** — an alternative to proprietary S1/S2 fusion (e.g. CropSAR) via FuseTS/MOGPR.
- **Multi-sensor** — paddy map & planting index come from Sentinel-1; irrigation performance fuses Sentinel-1 + Sentinel-2/Landsat optical + rainfall (CHIRPS) + ET.
- **Real-world stakes** — food security + irrigation performance for Indonesia.
- **Bilingual** — reaches the global EO community *and* Indonesian institutions.

---

## Visual assets (attach images — image posts get 3–5× engagement)

Best order to use them:

| # | File (`tutorial/images/`) | Use for |
|---|---|---|
| 1 | `fig_consensus_2024_2025_stable.png` | Opening "wow": stable-paddy map of Java (~3.0M ha) |
| 2 | `fig_irrigation_klambu_maps.png` | Irrigation SI/CU/RI panels, DI Klambu (0.83 / 0.94 / 0.98) |
| 3 | `fig_cropping_intensity_mt2024_25.png` | Planting-index map (1× / 2× / 3×) |
| 4 | `fig_rice_vh_profile.png` | Explains *why* radar detects rice (VH signature) |
| 5 | `fig_pipeline_operasional.png` | Closing tweet: end-to-end pipeline diagram |

### Alt-text (paste into X's image alt field — accessibility + reach)

- **Consensus:** "Map of Java, Indonesia showing stable rice-paddy areas (~3.0 million hectares) detected from Sentinel-1 radar across 2024–2025."
- **Irrigation:** "Three choropleth maps of DI Klambu's 653 tertiary blocks — Satisfaction Index 0.83, Christiansen Uniformity 0.94, Reliability 0.98 — with problem blocks highlighted at the tail-end of the network."
- **Cropping intensity:** "Map of Java showing planting intensity per pixel: single (1×), double (2×) and triple (3×) cropping, concentrated along water-secure corridors."
- **VH profile:** "Line chart of Sentinel-1 VH backscatter across a rice growth cycle: a deep flooding trough rising to a canopy peak, then falling toward harvest."
- **Pipeline:** "Flow diagram from Sentinel-1 acquisition through feature extraction and classification to irrigation performance indicators."

---

## Launch thread — English

**1/** 🌾🛰️ Free, laptop-friendly tutorial over Java, Indonesia: paddy maps & planting index from Sentinel-1 radar — then fuse it with Sentinel-2 & Landsat optical + rainfall (CHIRPS) & ET to score irrigation performance. No GPU, reproducible, open-source. 🧵
`[attach fig_consensus_2024_2025_stable.png]`

**2/** Why radar? Rice has a distinctive VH backscatter signature — a flooding trough → canopy peak every season. Sentinel-1 sees it every 12 days, through clouds. That single signal drives everything.
`[attach fig_rice_vh_profile.png]`

**3/** Product 1 — a multi-year *stable-paddy* consensus (~3.0M ha for Java), built with an SMOTE-balanced classifier on 29 VH phenology features. A cycle-based test then maps *active* paddy.

**4/** Product 2 — planting index (1×/2×/3× cropping) straight from the S1 time series, no fixed calendar.

**5/** Product 3 — irrigation performance. Fuse Sentinel-1 (crop coefficient / Kc) with Sentinel-2 & Landsat optical and rainfall (CHIRPS) + reference ET in a per-block water balance → SI 0.83 / CU 0.94 / RI 0.98 for DI Klambu, flagging the 10 problem blocks out of 653.
`[attach fig_irrigation_klambu_maps.png]`

**6/** Everything runs on your laptop from a small sample AOI + a trained model that ships with the repo. Open-source alternative to proprietary S1/S2 fusion (via FuseTS/MOGPR).

**7/** 📖 Full hands-on tutorial (EN + Bahasa Indonesia):
https://firmanhadi21.github.io/s1-rice-irrigation-tutorial/
⭐ Code: https://github.com/firmanhadi21/s1-rice-irrigation-tutorial
RT if useful — feedback welcome!
`[attach fig_pipeline_operasional.png]`

---

## Launch thread — Bahasa Indonesia

**1/** 🌾🛰️ Tutorial gratis & bisa jalan di laptop untuk pangan & irigasi di Jawa: peta sawah & indeks pertanaman dari radar Sentinel-1 — lalu difusikan dengan optik Sentinel-2 & Landsat + curah hujan (CHIRPS) & ET untuk menilai kinerja irigasi. Tanpa GPU, dapat direproduksi, open-source. 🧵
`[lampirkan fig_consensus_2024_2025_stable.png]`

**2/** Padi punya tanda hamburan-balik VH yang khas: lembah genangan → puncak kanopi tiap musim. Sentinel-1 merekamnya tiap 12 hari, tembus awan. Sinyal itu jadi dasar semua produk.
`[lampirkan fig_rice_vh_profile.png]`

**3/** Kinerja irigasi DI Klambu (653 petak tersier) dari fusi S1 + optik S2/Landsat + curah hujan + ET: SI 0.83, CU 0.94, RI 0.98 — sistem otomatis menandai 10 petak bermasalah di ujung hilir. Alih-alih cek 653 petak, BBWS cukup fokus ke 10.
`[lampirkan fig_irrigation_klambu_maps.png]`

**4/** 📖 Tutorial lengkap (dwibahasa) + kode terbuka:
https://firmanhadi21.github.io/s1-rice-irrigation-tutorial/
Cocok untuk mahasiswa, peneliti, & instansi (Kementan/PUPR/BBWS). Silakan sebarkan 🙏

---

## Standalone posts (reuse over the following weeks)

- "You don't need a supercomputer to map rice. This runs on a laptop, CPU-only, from free Sentinel-1 radar. Full tutorial 👇 [link] `[fig_rice_vh_profile.png]`"
- "Irrigation performance for 653 tertiary blocks by fusing Sentinel-1 + Sentinel-2/Landsat + rainfall + ET: SI 0.83 / CU 0.94 / RI 0.98 — and it points straight to the 10 problem blocks. [link] `[fig_irrigation_klambu_maps.png]`"
- "Proprietary S1/S2 fusion (CropSAR) is great — but a national agency on a public budget needs open tools. Here's the same idea with open-source FuseTS/MOGPR. [link]"
- "Peta sawah stabil Jawa ~3,0 juta ha, dari radar Sentinel-1 gratis. Metode + kode terbuka, dwibahasa. [link] `[fig_consensus_2024_2025_stable.png]`"

---

## Accounts to tag / mention (relevant, not spammy)

- **Global EO:** `@CopernicusEU` `@ESA_EO` `@sentinel_hub` `@EO_OpenScience` `@googleearth` (GEE)
- **Tooling / research:** `@openEOPlatform` (FuseTS) · `@VITObelgium` (CropSAR comparison)
- **Indonesia:** Kementerian PU/PUPR · Kementan · `@BRIN_Indonesia` · BIG/InaGeoportal · akademisi geodesi UNDIP

## Hashtags (pick 3–4 per post)

`#SAR #Sentinel1 #RemoteSensing #GEE #RiceMapping #Irrigation #OpenScience #Indonesia #Pangan #GISchat #EarthObservation`

---

## Tactics that lift reach

- **Timing:** ~08–10 WIB or 14–16 UTC (EU/US EO community active). Tue–Thu best.
- **No link in tweet 1** — X suppresses posts with links. Put the link in a reply / final tweet (as above).
- **Pin** the launch thread to the profile.
- **Alt-text** every image (accessibility + discoverability).
- **10–15s video/GIF** scrolling a map or a time series usually beats a static image.
- **Repurpose** the thread to LinkedIn (institutional audience); tag co-authors.
- **Follow-up a week later:** one deep-dive post per product to extend the content's life.

## Launch checklist

- [ ] Site + repo links live and public
- [ ] 5 figures exported, each with alt-text ready
- [ ] Optional: 16:9 montage card for the opening tweet
- [ ] Launch thread posted (EN), then quote-tweeted/threaded (ID)
- [ ] Thread pinned to profile
- [ ] Cross-posted to LinkedIn, co-authors tagged
- [ ] Follow-up deep-dive posts scheduled (T+7, T+14)
