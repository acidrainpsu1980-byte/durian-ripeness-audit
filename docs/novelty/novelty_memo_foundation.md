# Novelty memo (round 5): pretrained audio and vision foundation encoders for durian ripeness under batch-disjoint evaluation

Prepared 2 November 2026 for the COMPAG resubmission, round 5. Companion file: `novelty_refs_foundation.csv` (87 DOIs, all resolved: 53 via Crossref, 34 via DataCite for arXiv/Zenodo/preprint DOIs). This memo extends `novelty_memo_baseline.md` (round 4: fruit-specific intake baseline) and does not repeat its content. Two durian items (Orkweha 2026; Jing 2026 MCAFNet) appear in both lists.

**Evidence level.** Tags in the text: **[F]** full text read; **[A]** abstract (OpenAlex); **[T]** title only; **[C]** model code, config or model card inspected directly (Hugging Face API, GitHub source, Zenodo record). Only four papers were read in full, all arXiv: Zhou 2026 (BEATs ASD pooling), Quelennec 2023 (temporal support), Rauch 2025 (probing), Lian 2026 (DINOv2 food mass). **Every presence/absence call about the fruit and agricultural literature rests on abstracts and is provisional.** The full text of the closest work (Sánchez-Gendriz 2026, coconut, frozen CLAP) could not be retrieved because the publisher blocked automated access with HTTP 403. MCAFNet (Jing 2026, durian audio-visual) has no retrievable abstract.

**Search scope and reductions.** About 75 OpenAlex queries, run with the API key as relevance search and as title+abstract filter search. Hits screened: 7–12 per query. Semantic Scholar was rate-limited (HTTP 429 and proxy disconnects) and contributed nothing. There was no citation snowballing. Acronym collisions limited the searches: "AST" matched aspartate aminotransferase, "BEATs" matched musical beat, "CLAP" matched clips. Pretrained-audio searches therefore relied on phrase queries and may have missed papers that name a model only in the full text.

---

## 0. Verdict

1. **Pretrained general-purpose audio embeddings have already been used on fruit tapping sounds.** Sánchez-Gendriz 2026 [A] used frozen CLAP on coconut tapping with three classes (immature, mature, overmature) and fruit-disjoint splits; CLAP beat Mel-spectral controls by 9–10 pp per event. Phawiakkharakun 2026 [A] tested YAMNet and VGGish on pineapple, where an MFCC CNN won. ImageNet CNNs applied to knock spectrograms exist for durian (Bui 2024 [A]) and coconut (Sattar 2024 [A]). Audio-visual ripeness fusion exists for avocado (An 2026 [A]), melon (Giri 2026 [A]) and, by its title, durian (Jing 2026 MCAFNet [T]). **"First use of foundation encoders for fruit acoustics" cannot be claimed.**
2. **Not found (provisional, abstract level):**
   - Any evaluation of pretrained audio or vision foundation encoders for fruit ripeness under harvest-batch, season or orchard-disjoint evaluation.
   - Any frozen-probe versus fine-tune comparison in fruit acoustics.
   - Any audio foundation encoder applied to durian knocks.
   - Any use of within-fruit change of foundation-model embeddings (intake to endpoint).

   The closest precedent for change in embeddings is outside agriculture: Slavkova 2026 [A], longitudinal drift of mammogram foundation-model embeddings.
3. **Expectations from adjacent fields are mixed.** Frozen foundation-model probes often trail trained models in-domain but transfer better across domains: underwater audio, 11 of 12 cross-corpus directions (Yuan 2026 preprint [A]); leave-one-container-out pouring sounds with AST (Tran 2026 [A]). Counter-examples show failure or batch dominance:
   - DINOv2 fine-tuning did not transfer across cameras (Lian 2026 [F]).
   - Prithvi collapsed under phenology shift (Kolluru 2026 [A]).
   - Brain foundation-model embeddings were dominated by site batch (Tao 2026 [A]).
   - Ship-noise embeddings were dominated by recording identity (Hummel 2026 [A]).

   Pre-register the round so that a negative result is reportable.
4. **Recommended fixed set (Section 3):**
   - Audio: BEATs iter3+ AS2M (SSL), PANNs CNN14 32 kHz, LAION CLAP htsat-unfused, AST AudioSet. wav2vec2-base and HuBERT-base are extracted but kept secondary, as a pretraining-domain contrast.
   - Vision: DINOv2 ViT-B/14 and SigLIP base/16-224. CLIP ViT-B/16 is secondary.
   - Fine-tuning: **pre-specify BEATs iter3+ AS2M (SSL checkpoint)** as the single audio encoder for light fine-tuning, using LP-FT.
5. **Two input issues must be fixed before extraction (Section 3.3):**
   - (i) A 150 ms knock window is shorter than the minimum input of BEATs (about 175 ms) and of CNN14 (about 0.31 s), derived from code [C]. It must be padded under a pre-specified rule.
   - (ii) Feeding encoders from the 22.05 kHz derivative discards content above 11 kHz that CLAP (48 kHz) and CNN14 (to 14 kHz) can use. Re-extract each knock window from the original-rate recordings and resample once, directly to each encoder's rate.

---

## 1. Pretrained audio encoders on knocking, tapping and other short impulsive sounds

### 1.1 Fruit and food

| Work | Encoder(s) | Task, n | Split | Cross-batch/season? | Frozen vs FT? | Evidence |
|---|---|---|---|---|---|---|
| Sánchez-Gendriz 2026, AgriEngineering 10.3390/agriengineering8090395 | **Frozen CLAP** (checkpoint not named in abstract) vs Mel-spectral; ML/DL heads | Coconut maturity: immature/mature/overmature, 3 ridge positions, public dataset | **Fruit-disjoint**; event → fruit majority vote | Not reported | Frozen only (abstract) | [A] |
| Phawiakkharakun 2026, AgriEngineering 10.3390/agriengineering8080324 | YAMNet, VGGish embeddings vs MFCC | Pineapple juiciness, 300 fruits, mobile phone | Not stated | Not reported | Embeddings frozen; **MFCC-CNN best** (acc 0.9415) | [A] |
| Bui 2024, JAC-ECC 10.1109/jac-ecc64419.2024.11061240 | ImageNet MobileNetV2 on STFT images | Durian 'Dona' ripeness, knocking | Validation split | Not reported | Fine-tuned (implied) | [A] |
| Sattar 2024, IOCAG 10.3390/iocag2023-16880 | ImageNet ResNet50/InceptionV3/MobileNetV2 | Coconut, 122 knocks | Not stated | Not reported | Not stated | [A] |
| Li 2026, JFCA 10.1016/j.jfca.2026.109106 | CNN-Transformer (trained) vs MobileNet/ResNet/Transformer | Pear firmness, acoustic+vibration | Not stated | Not reported | n/a | [A] |

**Reading.**
- **The coconut CLAP study is the direct precedent.** It has the same 3-class immature/mature/overmature structure as the Zenodo durian dataset, repeated taps per fruit, and fruit-level aggregation. Its gains are within a single dataset under fruit-disjoint splits; the abstract mentions no batch, season or orchard factor.
- **Its fruit-level number is not comparable to ours.** The 93.06% fruit-level balanced accuracy comes from voting over taps and ridge positions; event-level accuracy was 64.72%.
- **The pineapple study is the counter-precedent.** Generic AudioSet-era embeddings (YAMNet, VGGish) did not beat a task-specific MFCC CNN in-domain.
- **None of the five reports a batch-disjoint test or a frozen-versus-fine-tuned comparison.** This holds at abstract level.

### 1.2 Other short impulsive or industrial sounds: domain shift and frozen versus fine-tuning

- **Leave-one-group-out with a pretrained transformer.** Tran 2026 [A] (pouring sounds, IEEE Access) uses leave-one-container-out over 7 containers. Single-container training reached only 59.27% on unseen containers, with AST ahead of VGGish and ResNet-50. AST with multi-task contrastive/adversarial training reached 95.02% under LOCO, versus 85.80% for plain AST. **This is the nearest design analogue to leave-one-batch-out.** The gain came from adding invariance objectives, not from the pretrained encoder alone.
- **Cross-corpus transfer of frozen encoders.** Yuan 2026 [A, preprint] benchmarked six frozen encoders (wav2vec2-base/large, HuBERT-base, WavLM-base+, AST, CLAP-HTSAT) on four underwater corpora.
  - In-domain, frozen linear probes trail a trained ResNet-18 on macro-F1.
  - Cross-corpus, the best frozen probe beats the trained ResNet-18's zero-shot transfer on 11/12 anchored directions. This is a per-direction oracle selection, as the authors note.
  - Environmental-audio pretraining (AST, CLAP) beats speech pretraining in 6/8 cells, and wav2vec2-large never beats base.
  - **This is the strongest support for the round-5 hypothesis, and also a warning: the result depends on choosing the encoder after seeing the outcome.**
- **Recording identity dominates frozen embeddings.** Hummel 2026 [A] (ship noise): the geometry of frozen audio embeddings is "largely dominated by recording-specific characteristics", which a linear probe suppresses. Expect batch and device to dominate unsupervised structure in our embeddings too, so do not interpret UMAP clusters as ripeness.
- **Machine anomalous-sound detection (DCASE Task 2) is the most mature source/target-shift literature for frozen encoders.**
  - Zhou 2026 [F, preprint]: frozen BEATs iter3+ (AS2M), 768-d. Switching the scoring backend moved target-domain AUC by **13.8 points on average versus 3.2 for switching pooling** (mean, GeM, max). Pooling therefore matters less than the head, but both must be fixed in advance. No fine-tuning arm.
  - Mkrtchian 2026 [A, preprint]: source and target AUC are negatively correlated across systems, and development-set rankings do not predict evaluation-set rankings. This argues against choosing an encoder on in-batch cross-validation.
  - Zheng 2024 [A] (SLT): LoRA versus full fine-tuning of pretrained audio models for ASD.
  - Jia 2026 [A] and Zhang 2026 [A]: domain-generalisation add-ons.
- **Partial or full fine-tuning on short, non-AudioSet signals.**
  - Svendsen 2026 [A] (acoustic emission, hydraulic seals): AST with selective layer freezing.
  - Niu 2026 [A] (respiratory events, patient-disjoint, two databases): BEATs with an exact padding-support mask for variable-length padded events. Full fine-tuning beat rank-8 LoRA by 3.94 pp (95% CI 2.54–5.40). **This is the closest methodological precedent for fine-tuning BEATs on short, padded events with subject-disjoint splits.**
- **Short inputs and pooling.**
  - Quelennec 2023 [F]: patchout transformers (PaSST, BEATs) stay effective with shorter temporal support, but the shortest tested was 1 s. **There is no published evidence on clips around 150 ms.**
  - Rauch 2025 [F]: CLS or global pooling is a bottleneck for frozen audio SSL encoders (Audio-MAE, BEATs, EAT and others). Patch-token prototype probes add 14.41 pp mAP over linear and reshuffle encoder rankings. **The encoder ranking depends on the probe, so pre-specify the probe.**
- **Speech SSL on non-speech.** Wu 2022 [A] (HEAR): speech SSL models do well on many non-speech tasks but fail on fine-grained pitch and onset tasks. Knock resonance is pitch-like, so a weaker wav2vec2/HuBERT result is expected.

### 1.3 Cross-batch, frozen-versus-fine-tune and temporal-change findings (answers to item 1)

- **Cross-batch, season or orchard tests with audio foundation encoders on fruit or food:** none found [A]. Analogues outside agriculture: LOCO containers (Tran 2026), cross-corpus underwater (Yuan 2026), DCASE source/target (Zhou 2026, Mkrtchian 2026).
- **Frozen probe versus fine-tune:** not compared in any fruit or food acoustics paper found. Elsewhere: LoRA versus full fine-tuning (Zheng 2024; Niu 2026, where full fine-tuning won); layer freezing (Svendsen 2026); frozen probes versus a fine-tuned CNN (Yuan 2026). The general result is Kumar 2022 [metadata]: fine-tuning can distort pretrained features and underperform out of distribution, and LP-FT mitigates this. WiSE-FT (Wortsman 2022) is the weight-interpolation alternative.
- **Temporal or within-object change of embeddings:** none in fruit, food or acoustics found.
  - Slavkova 2026 [A, preprint]: frozen mammography foundation-model embeddings drift faster along a data-derived direction in women later diagnosed than in matched controls. The effect appeared for clinically pretrained models but not for BiomedCLIP. Whether change is captured depends on the pretraining domain.
  - De Silva 2026 [A, preprint]: Perch v2 audio-embedding dissimilarity tracks soundscape change.
  - Zheng 2024 AnyChange [A] and Kim 2025 [A]: zero-shot bitemporal change detection in SAM latent space. Kim 2025 reports that supervised scene-change detection collapses from 77.6% to 4.6% under a different temporal condition.

---

## 2. Vision foundation encoders for RGB, thermal and multispectral fruit images; fusion with audio; domain shift

### 2.1 Vision foundation models in agriculture and food (RGB)
- **Frozen DINOv2 and CLIP backbones are standard and strong in plant pathology.** In El Karch 2026 [A] (in-the-wild plant disease), DINOv2 ViT-L/14 alone exceeds a text-augmented state of the art (77.56% vs 76.18%). Concatenating frozen DINOv2+DINOv3+CLIP reaches 80.23%, and learned routing adds nothing. **This supports pairing one SSL and one language-supervised backbone with simple concatenation or late fusion.**
- **Fruit-quality studies with language-image models exist but could not be checked.** Duan 2024 COMPAG [T] and Huang 2025 COMPAG [T] are title-only.
- **Small, domain-shifted agricultural datasets need regularised heads.** Khabarlak 2026 [A, preprint] (DINOv2, cucumber stages) is an example. Agricultural pretraining gives modest gains over ImageNet: Agri-FM+ (Al Nahian 2025 [A]) +1.27%; A100 (Das 2026 [A, preprint]) claims frozen gains over ImageNet-12K.

### 2.2 Domain shift (season, batch, orchard, camera) for these encoders
- **Camera and dataset shift.** In Lian 2026 [F], fine-tuned DINOv2-base had an in-domain MAE of 24.8 g, against 60.3 g for a zero-shot VLM, but no advantage or worse on three external datasets. DINOv2-large did not reduce the external error. The gram scale changed with framing.
  - **Implication:** the RGB arm should use a fixed crop/framing rule and stratify by camera, because the two RGB cameras in our dataset are a camera shift nested within batches.
- **Lab-to-field shift.** In Xiang 2026 [A], PlantVillage-trained ResNet-50 lost 67.7 pp on PlantDoc while confidence stayed near 80%. Adaptive BN and moment matching helped little, and DANN hurt. In a frozen-backbone comparison, DINOv2-S/14 had the best out-of-distribution detection AUROC (0.764).
- **Season and phenology shift.**
  - Kolluru 2026 [A, preprint]: Prithvi-EO-2.0 accuracy fell from 0.65 to 0.40 across continents and collapsed under phenology misalignment, with confidence still high.
  - Tang 2026 [A] (apple hyperspectral, three seasons): seasonal specificity, with pooled-season training best.
  - Chang 2025 [A]: sensor-specific pretraining (SSL4EO-S12) beats ImageNet for multispectral crop mapping.
- **Zero-shot CLIP/SigLIP under agrifood shift.** Morales-Garzón 2026 [A, preprint] covers zero-shot only (prompt ensembles help under shift); this does not transfer to linear probes.
- **Batch dominance.** Tao 2026 [A, preprint]: site batch often dominates diagnosis information in foundation-model embeddings. Here, batch equals maturity, so batch variance and label variance are confounded in the embedding (see Section 4.3).

### 2.3 How thermal matrices and multispectral bands are fed to RGB-pretrained encoders
- **Thermal.**
  - Fruit thermal transfer-learning papers fine-tune ImageNet CNNs: banana ripeness (K 2023 [A], 97 fruits) and pineapple varieties (Hashim 2025 [A]). Their abstracts do not say how temperature was mapped to RGB (camera palette versus grey replication).
  - A thermal fruit-bruising study with a cross-dataset evaluation exists (Nogay 2026 [T]); content not verified.
  - RGB-pretrained foundation models are not natively thermal. AnyThermal (Maheshwari 2026 [A]) distils DINOv2 into a thermal encoder; SAM-CLIP needs few-shot adaptation for thermal plant images (Lin 2026 [A]).
  - **No paper was found that feeds absolute-temperature matrices (°C) to DINOv2, CLIP or SigLIP for fruit.**
- **Multispectral.** Four routes appear:
  - **(a) Per-band encoding with grey replication**, then fusion of the band embeddings: ViT-B16 per filter (De Silva 2023 [A]); DINOv2 with learnable spectral positional encoding and token fusion (Ramachandra 2025/26 [A]).
  - **(b) False-colour composites of selected bands**: LDA-selected composites from 8-channel VIS-NIR (Rouš 2023 [A]); rendered band composites and indices as multiple views for VLMs (Liu 2026 [A, preprint]).
  - **(c) Channel adapters**, where a CNN stem is adapted to more channels (Papadeas 2026 [A]). ChannelViT (Bao 2023), DOFA (Xiong 2024) and SatMAE (Cong 2022) all need training on many images.
  - **(d) Sensor-specific pretrained models** (Chang 2025).
  - **Our choice.** With about 110 training fruits per fold, only (a) is feasible without training new layers or selecting bands on data. (b) needs a band choice, and choosing bands on the data is a selection step that must be fixed in advance.

### 2.4 Fusion with audio
- **Fruit audio-visual fusion so far:** CNN-level fusion of image and acoustic (+weight) for avocado (An 2026 [A]: 0.949 vs image-only 0.838) and melon (Giri 2026 [A]), plus cross-attention for on-branch durian (Jing 2026 MCAFNet [T]). None uses pretrained foundation encoders at abstract level, and none reports a batch-disjoint test.
- **Device transfer.** In agricultural robotics, audio-visual contact classification transferred zero-shot to a new probe embodiment (Spears 2025 [A]).
- **Fusion taxonomy.** General references are Baltrušaitis 2019 (early/late/hybrid) and Nagrani 2021 (attention bottlenecks). Food reviews (Chen 2026 [A]) recommend multisite and multiseason validation.
- **Choice for this sample size.** With roughly 164 fruits, transformer or attention fusion cannot be trained reliably on frozen embeddings. Late fusion of per-modality probabilities (consistent with the current T+S+R mean rule) and per-modality-PCA concatenation are the defensible options.

---

## 3. Recommended fixed encoder set for the pre-registered round

### 3.1 Table

Input facts were taken from Hugging Face `config.json`/`preprocessor_config.json`, the BEATs and PANNs source code, and the Zenodo record, on 2 Nov 2026 [C]. "Minimum input" for BEATs and CNN14 is **derived from the architecture**, not tested; the GPU track must confirm it with a 150 ms dummy input.

| Role | Encoder | Exact checkpoint | Licence | Input | Minimum input / short-clip handling | Emb. dim | Clip-level embedding (pre-specify) |
|---|---|---|---|---|---|---|---|
| **Audio 1 (primary; fine-tune target)** | BEATs iter3+ (SSL, AudioSet-2M) | `BEATs_iter3_plus_AS2M.pt` (self-supervised, *not* the "Fine-tuned … (cpt1/cpt2)" files), GitHub microsoft/unilm/beats README release links | MIT (unilm repo; no separate weight licence stated) | 16 kHz mono; Kaldi fbank 128 bins, 25 ms window / 10 ms shift; normalised (x−15.41663)/(2·6.55582) inside `preprocess` | 16×16 patch conv needs ≥16 fbank frames ≈ **2,800 samples (175 ms)**. A 150 ms window (2,400 samples) gives 13 frames, so it must be zero-padded with `padding_mask` | 768 | Mean of last-layer tokens where `padding_mask` is False (output of `extract_features`) |
| **Audio 2** | PANNs CNN14 (AudioSet-supervised CNN) | `Cnn14_mAP=0.431.pth`, Zenodo 10.5281/zenodo.3987831; code qiuqiangkong/audioset_tagging_cnn | Weights **CC-BY-4.0** (Zenodo); code MIT | 32 kHz; log-mel 64 bins, window 1024, hop 320, 50–14,000 Hz | Five 2×2 avg-pools need ≥32 STFT frames ≈ **9,920 samples (0.31 s)**; 150 ms gives 16 frames, so zero-pad | 2048 | Built-in `embedding` output: mean over frequency, then max+mean over time, fc1 + ReLU. Zero-padding dilutes the mean term, so fix the padded length |
| **Audio 3** | LAION CLAP (HTSAT-tiny audio tower, language-supervised) | HF `laion/clap-htsat-unfused` | Apache-2.0 | 48 kHz; log-mel 64 bins, hop 480, window 1024, 50–14,000 Hz; 10 s | Feature extractor default `padding="repeatpad"` tiles a 150 ms knock about 67× to 10 s; no hard minimum. Pre-specify repeatpad (default) | 512 (projected, L2-normalised); 768 pre-projection | `get_audio_features` (512-d, L2-normalised) |
| **Audio 4** | AST (AudioSet-supervised ViT) | HF `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause | 16 kHz; Kaldi fbank 128 bins; 1,024 frames (10.24 s); normalised mean −4.2677, std 4.5690 | Extractor zero-pads to 1,024 frames, so a 13-frame knock fills about 1.3% of the input. No attention mask; fixed positional embeddings | 768 | `pooler_output` = mean of CLS and distillation tokens |
| Audio secondary (pretraining-domain contrast) | wav2vec 2.0 base (speech SSL, LibriSpeech) | HF `facebook/wav2vec2-base` | Apache-2.0 | 16 kHz raw waveform | Receptive field 400 samples (25 ms), stride 320 (20 ms); 150 ms gives 7 frames; no padding needed | 768 | Mean over frames of one pre-specified layer. Recommend the **mean of all 12 transformer layers' frame-means**, to avoid choosing a layer on data |
| Audio secondary | HuBERT base | HF `facebook/hubert-base-ls960` | Apache-2.0 | Same as wav2vec2-base | Same | 768 | Same rule as wav2vec2 |
| **Vision 1** | DINOv2 ViT-B/14 (SSL) | HF `facebook/dinov2-base` | Apache-2.0 | 224×224, patch 14 (16×16 tokens; positional embeddings learned at 518 and interpolated); ImageNet mean/std | Override the HF default (resize 256 + centre-crop 224) with **pad-to-square, then resize to 224**, so the fruit is not cropped | 768 | `pooler_output` = final-layer-normed CLS |
| **Vision 2** | SigLIP base/16 (language-supervised, sigmoid loss) | HF `google/siglip-base-patch16-224` | Apache-2.0 | 224×224 fixed resize; mean = std = 0.5 | Pad-to-square, then resize to 224 | 768 | Vision tower `pooler_output` (MAP attention-pooling head) |
| Vision secondary | CLIP ViT-B/16 | HF `openai/clip-vit-base-patch16` | Code MIT (GitHub). HF card has no licence tag and states "**Any** deployed use case … is currently out of scope" | 224×224; CLIP mean/std | Same | 512 projected / 768 pooled | `get_image_features` (512-d) |

**Why this set.**
- **The four primary audio encoders differ on objective and architecture, with the pretraining corpus mostly held fixed:**
  - BEATs: self-supervised transformer, AudioSet.
  - AST: supervised transformer, AudioSet.
  - CNN14: supervised CNN, AudioSet.
  - CLAP: language-supervised (audio-text pairs from LAION-Audio-630K plus AudioSet with keyword-to-caption), HTS-AT backbone.
- **CLAP is the only encoder with a direct fruit-tapping precedent** (Sánchez-Gendriz 2026).
- **Environmental-audio pretraining beat speech pretraining in the closest cross-domain benchmark** (Yuan 2026), and speech SSL is weak on pitch-like tasks (Wu 2022). wav2vec2 and HuBERT are therefore secondary contrasts, not primary candidates.
- **Vision: DINOv2 (self-supervised) and SigLIP (language-supervised) give two training objectives**, the combination that worked frozen in El Karch 2026. Both are Apache-2.0. SigLIP replaces CLIP as primary because CLIP's model card declares deployed use out of scope.
- **Not recommended:** DINOv3 (gated, custom licence) and SigLIP 2 (newer; no agricultural evidence found).

### 3.2 Gaps in the candidate list

- **EAT** (HF `worstchan/EAT-base_epoch30_finetune_AS2M`, MIT; 16 kHz, 128-bin fbank; model-card usage snippet pads/truncates to `target_length = 1024` frames (10 s), while `config.json` lists `max_length: 768`, so the GPU track must confirm the admissible length; 768-d CLS). This is a strong SSL alternative to BEATs, but adds little diversity because it has the same corpus and a similar objective. Optional; not needed.
- **No encoder uses the full bandwidth of the recordings.** BEATs, AST, wav2vec2 and HuBERT are 16 kHz models with Nyquist at 8 kHz, so they discard the 8–11 kHz band that the current 22.05 kHz CNN sees. Only CNN14 (to 14 kHz) and CLAP (to 14 kHz at 48 kHz) can use more. This is unavoidable and should be stated as a limitation.
  - **Upsampling the 22.05 kHz derivative to 32 or 48 kHz leaves CNN14 and CLAP mel bands above 11 kHz empty.** Re-extract each knock window from the original iPhone m4a / Android wav files at the same onset times, and resample once to each encoder's native rate.
- **Thermal-native encoder.** AnyThermal (Maheshwari 2026) exists, but its release and licence were not checked. Not recommended for a fixed round.

### 3.3 Pre-specified input rules (to freeze in ANALYSIS_PLAN_round5.md)

1. **Audio window.**
   - Primary: the same 150 ms knock windows (same onsets) as the current CNN, re-cut from the original-rate audio.
   - Padding to each encoder's convention: BEATs zero-pad to 1.0 s with `padding_mask`; CNN14 zero-pad to 1.0 s; AST extractor zero-pad to 10.24 s; CLAP repeatpad to 10 s; wav2vec2/HuBERT none.
   - One exploratory sensitivity: a 0.5 s real-context window.
   - Average the knock embeddings within each capture, then within each fruit and visit, exactly as for the CNN embedding.
2. **Thermal (396×528, °F).**
   - Convert to °C and clip to a fixed physical window declared before extraction (for example 20–40 °C). Report the clipped-pixel fraction, label-free.
   - Scale to 0–255, replicate to 3 channels, pad to square, resize to 224.
   - Do not use per-image min–max as primary, because it removes absolute temperature. Use it only as a sensitivity analysis.
   - No colormap; a palette is an arbitrary nonlinearity.
3. **Multispectral (8 bands).**
   - Encode each band separately: fixed per-band intensity range, grey replicated to 3 channels.
   - Concatenate the 8 band embeddings in fixed wavelength order (8×768 = 6,144-d), then reduce inside each fold with the same PCA/regularisation as the other modalities.
   - No band selection, no false colour, no channel adapters.
4. **RGB.**
   - Embed each image with pad-to-square + resize.
   - Average per fruit within camera, and keep camera as a stratum. If the two cameras are pooled, report per-camera results.
5. **Heads and pooling.**
   - One head for all encoders: the existing regularised logistic head with fold-internal standardisation, PCA dimension and C fixed in advance.
   - Pooling as listed in the table.
   - This follows Zhou 2026 and Rauch 2025: the head and pooling choice can reorder encoders.
6. **Change embedding.** Δz = z_endpoint − z_intake, computed with the same frozen encoder, same device, same knock-site rule. Because the encoder is frozen and label-free, the round-4 requirement for fold-nested encoder training does not apply to the frozen arms. This methodological gain over round 4 is worth stating.

### 3.4 The single audio encoder to pre-specify for light fine-tuning: **BEATs iter3+ AS2M (SSL)**

**Reasons.**
1. **It handles the shortest input with masking.** It needs about 175 ms (derived) and supports `padding_mask`, so a 150 ms knock needs minimal padding. AST and CLAP force 10 s inputs (≈99% padding or ≈67 repeats). CNN14 needs ≥0.31 s and has no mask.
2. **The closest fine-tuning precedent uses BEATs.** Niu 2026 [A] fine-tuned BEATs on short, variable-length events with an exact padding-support mask, subject-disjoint splits and a pre-specified protocol. Full fine-tuning beat LoRA.
3. **It is designed for fine-tuning.** The SSL checkpoint has no AudioSet label head, so there is no supervised AudioSet label prior to unlearn, and the BEATs paper reports its best results after fine-tuning. Among the candidates, transformer encoders stay usable at shorter temporal support (Quelennec 2023 [F]; tested only down to 1 s).
4. **It costs least to fine-tune.** It is one model with one 768-d output and a short input; AST and CLAP would process ≈10 s per knock.
5. **Licence:** MIT.

**Pre-specified fine-tuning protocol (suggested).**
- Use LP-FT (Kumar 2022): fit the head on frozen features, then unfreeze only the top 2 of 12 transformer blocks plus the head.
- Fixed learning rate, epochs and weight decay; no early stopping on the held-out batch.
- Inner validation uses fruit-grouped splits within the training batches only.
- Average three seeds.
- Optionally pre-specify WiSE-FT interpolation at α = 0.5 (Wortsman 2022) as a second, labelled-exploratory arm.
- Fine-tuning must be fold-nested: no held-out-batch audio, intake or endpoint, in any gradient step.

**Alternative if BEATs fails the dummy-input test:** CNN14. It is convolutional, cheap to fine-tune and accepts variable length above 0.31 s.

---

## 4. Novelty: what is and is not new

### 4.1 Not new (do not claim)
- **Pretrained general-purpose audio embeddings for fruit tapping or maturity:** Sánchez-Gendriz 2026 (frozen CLAP, coconut, 3 maturity classes); Phawiakkharakun 2026 (YAMNet/VGGish, pineapple). ImageNet CNNs on knock spectrograms, including durian: Bui 2024; Sattar 2024.
- **Audio-visual or multimodal fusion for fruit ripeness, including durian:** An 2026; Giri 2026; Jing 2026 MCAFNet [T].
- **Frozen DINOv2/CLIP/SigLIP features and backbone fusion in agriculture:** El Karch 2026; Xiang 2026; Khabarlak 2026.
- **Evaluating foundation encoders under group-held-out domain shift** (in other fields): Tran 2026 (LOCO); Yuan 2026 (cross-corpus); DCASE ASD (Zhou 2026; Mkrtchian 2026); Kolluru 2026; Lian 2026.
- **Within-subject change of frozen foundation-model embeddings** (in other fields): Slavkova 2026. **Bitemporal foundation-model change detection:** Zheng 2024 AnyChange.
- **Repeated or intake measurements of the same fruit:** see the baseline memo.

### 4.2 Plausibly new (provisional, abstract level; to be stated as "to our knowledge")
1. **A batch-disjoint (leave-one-harvest-batch-out) test** of pretrained audio and vision foundation encoders for fruit ripeness or maturity. The coconut CLAP study is fruit-disjoint within one dataset, and no study found tests batch, season or orchard transfer with such encoders.
2. **A pre-specified frozen-probe versus light fine-tune comparison** of an audio foundation encoder on fruit knocks.
3. **Audio foundation encoders (BEATs/CNN14/CLAP/AST) applied to durian knocks.** Durian acoustic work uses task-specific CNNs, MFCC networks or ImageNet MobileNetV2.
4. **An intake-to-endpoint change vector in a frozen foundation-model embedding for a perishable product,** compared against the endpoint embedding, the batch-mean reference and the shuffled-fruit control from round 4. With a frozen, label-free encoder, the change arm avoids fold-nested encoder training. This removes the leakage concern that constrained round 4.
5. **Thermal temperature matrices and 8-band multispectral fruit images through RGB foundation encoders,** using a pre-specified per-band or absolute-temperature mapping, evaluated batch-disjoint. Individual ingredients exist (per-band DINOv2 in iris PAD; thermal transfer learning for banana and pineapple); the combination for fruit ripeness was not found.

**Claim wording.** "We test, under a pre-registered leave-one-batch-out protocol, whether general-purpose pretrained audio and vision encoders transfer across harvest batches better than a task-specific knock CNN" is defensible. "First use of foundation models for fruit acoustics" is not. Avoid "foundation models solve batch shift" unless shown; the adjacent literature includes clear failures.

### 4.3 Risks that limit any novelty claim
- **Batch equals maturity.** A frozen encoder cannot separate batch-specific acoustics (device sessions, ambient conditions, storage duration) from ripeness. Frozen embeddings tend to encode recording and batch identity (Hummel 2026; Tao 2026). Any "transfer" gain is a gain in cross-batch-and-cross-maturity transfer, as in round 4.
- **Best-of-k selection.** With 4 audio × 2 vision encoders plus fusions, the best arm will look good by chance. The underwater benchmark explicitly reports per-direction oracle selection (Yuan 2026), and ASD shows that dev rankings do not predict eval rankings (Mkrtchian 2026). Two options:
  - **Option A (recommended):** pre-specify BEATs-frozen as the single primary audio contrast against the current CNN. Report the other encoders descriptively, or with Holm across the four audio encoders as one family.
  - **Option B:** test the four audio encoders as one Holm family.

  In both options, fusion and change arms are exploratory unless pre-specified one by one.
- **Expected direction.** The Overmature-batch collapse (Ripe recall 0) may be label/batch shift rather than representation weakness. A better representation does not fix label shift. Frame round 5 as a test that can be negative.

---

## 5. Conclusions that rest on abstracts only

- **Everything about Sánchez-Gendriz 2026 (coconut CLAP) beyond its abstract:** checkpoint, CLAP padding mode, whether harvest batches existed in the public dataset, and whether any frozen-versus-fine-tuned arm exists. **Obtain the full text before submission.** Library access is required, because the publisher site returns HTTP 403 to automated requests.
- **MCAFNet (Jing 2026, COMPAG; title only):** whether it uses pretrained audio or vision encoders, and whether it tests across batches, seasons or orchards. It is the most likely durian-specific pre-emption of any "fusion" claim; read it.
- **All absence calls in Sections 0 and 4.2**, and the input-mapping details for the thermal and multispectral fruit papers (Section 2.3).
- Yuan 2026, Kolluru 2026, Slavkova 2026, Tao 2026, Mkrtchian 2026, Morales-Garzón 2026, Liu 2026, Khabarlak 2026 and Das 2026 are **preprints** (not peer reviewed). Their abstracts were read; their numbers are as stated by the authors.
- Read in full: Zhou 2026 (BEATs pooling vs backend), Quelennec 2023 (temporal support ≥1 s), Rauch 2025 (probing), Lian 2026 (DINOv2 camera/dataset transfer).
- Model-specific facts in Section 3 are from configs and code [C], not papers. Minimum lengths for BEATs and CNN14 are derived from the architecture and **must be confirmed empirically** by the GPU track (dummy 150 ms input through each encoder) before ANALYSIS_PLAN_round5.md is tagged.

## 6. Next steps for the literature check
1. Obtain the full texts of Sánchez-Gendriz 2026 and Jing 2026 (MCAFNet) through library access; update Sections 1.1, 2.4 and 4.
2. Optional: run a forward-citation search from Sánchez-Gendriz 2026 and Tran 2026 for fruit or food papers with group-held-out evaluation. Not done here.
3. Record in the analysis plan which BEATs file the GPU track has actually extracted. "BEATs iter3+ AS2M" names both the SSL checkpoint and its two AudioSet-fine-tuned versions (cpt1/cpt2), and these differ.
