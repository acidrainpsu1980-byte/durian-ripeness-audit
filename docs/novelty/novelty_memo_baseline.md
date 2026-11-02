# Novelty memo: fruit-specific (intake-referenced) baseline for durian knock classification

Prepared November 2026 for the COMPAG resubmission (round 4 analysis). Companion file: `novelty_refs_baseline.csv` (66 verified DOIs).

**Evidence level.** Most of this memo rests on abstracts. Only one paper was read in full: Işık 2026, via PMC. Twelve references have no retrievable abstract and are cited by title only; the CSV `access` column marks them. In the text, **[A]** means abstract, **[S]** a Semantic Scholar abstract or TLDR, **[T]** title only, and **[F]** full text. Every presence or absence call below is **provisional at abstract level**.

**Search scope.** About 110 OpenAlex queries (with API key) and Crossref bibliographic queries, run in five rounds across the four requested categories. A further 66 DOIs were resolved against Crossref metadata; the two DataCite DOIs (arXiv, Zenodo) were checked through OpenAlex. Full-text retrieval was attempted for five closest works and succeeded for one (Işık 2026). Elsevier full texts were not reachable: Taniwaki 2009/2010, De Ketelaere 2004/2006, Nordey 2019.

**Reductions.** OpenAlex returned about 8–15 hits per query, and only those were screened; there was no snowballing through citations. Searches for "Siamese" in agricultural titles were swamped by "Siamese = Thai" results and gave almost no deep-learning hits. Titles were therefore screened by eye and category 3 may be under-sampled.

---

## 1. Bottom line

**What is established is the idea itself: a fruit's own earlier measurement used as a reference for predicting its later ripening state.** The novelty that can be defended is narrower. It is an *evaluation* result: whether a learned, within-fruit change representation from smartphone knocks transfers across harvest batches better than (i) the endpoint knock alone and (ii) batch-level anchoring. This holds only if the result is positive and survives a shuffled-baseline control (Section 5).

- **Established, many crops.** Repeated nondestructive acoustic or firmness measurement of the *same* fruit during storage dates back to at least 1996. Examples include kiwifruit (Hopkirk 1996 [A]), tomato (De Ketelaere 2003 [A]), avocado (Galili 1998 [A]), melon (Sugiyama 1998 [A]; Kuroki 2006 [A]), pear (Hiruta 2021 [A]) and mango (Schouten 2018 [A]).
- **Established: the fruit's own harvest value as a predictor.** Suzuki 2011 [A, Japanese] predicts the start and end of the persimmon eating window from each fruit's day-0 elasticity index Y0. Lurie 2013 [S] uses I_AD at harvest to set each peach's biological shift factor and to predict its softening. Han 2023 [A] predicts days-to-ripe for individual avocados from intake hyperspectral images. Taniwaki 2010 [T] measured melon "ripening speed" acoustically and is probably close, but its content is unverified.
- **Established in other fields: paired or Siamese encoders with difference fusion.** Examples are remote-sensing change detection (Caye Daudt 2018 [A]) and longitudinal medical imaging (Li 2020 [A]; Veasey 2020 [A]; Işık 2026 [F]). Işık 2026 uses exactly [f_first; f_last; f_last − f_first], a matched final-image-only baseline, and patient-grouped, leakage-controlled pretraining.
- **Established in other fields: subject-specific baseline normalisation and delta features.** Examples are EEG baseline removal (Ahmed 2023 [A]), wearable baselines (Chan 2020 [A]; Tognotti 2026 [A]), delta-radiomics (Fave 2017 [A]; Nardone 2024 review [A]) and cross-tool wear with an initial baseline under leave-one-tool-out validation (Ting 2026, preprint [A]).
- **Not found (provisional):**
  1. A learned (CNN or Siamese) within-fruit change representation for fruit ripeness or maturity from any sensor.
  2. Any postharvest study testing whether a per-fruit intake reference improves *cross-batch or cross-season* transfer. Season shift is documented (Sneddon 2024 [A]; Nguyen 2026 [A]) but was handled by pooled or updated calibration (Mishra 2021 [A]).
  3. Any durian study that uses repeated measurements of the same fruit as a reference. Durian acoustic studies all classify single captures. Brilliantina 2026 [A] groups repeated measurements by fruit to *prevent leakage*, not to use them as a baseline.

**Implication for the claim.** Do not claim a new concept ("fruit-specific baseline", "personalised ripeness", "first Siamese network for fruit"). The contribution should be framed as a pre-registered *test* of a long-standing postharvest idea, with a learned encoder and a cross-batch evaluation, in a crop and setting where it has not been tested. A negative result is also reportable on the same terms.

---

## 2. What is established (by category)

### 2.1 Repeated measurement of the same fruit (category 1)
- **Device and kinetics work.** Same-fruit repeats are routine in firmness-device validation and kinetic modelling: Hopkirk 1996, De Ketelaere 2003/2004/2006, Diezma-Iglesias 2006 [S], Galili 1998, Hiruta 2021 (per-fruit nonlinear trend fits of f²m^(2/3)) and Schouten 2018 (per-fruit repeats across batches from several origins).
- **Why repeated measures matter.** De Ketelaere 2003 [A] showed that repeated nondestructive firmness data need mixed models because measurements of the same fruit are correlated. This is the statistical argument for a within-fruit reference.
- **Intake value as the reference.** Suzuki 2011 is the clearest acoustic case found: the day-0 elasticity index of each persimmon enters the prediction equation for its eating window.
- **No classifier in the pipeline sense.** None of these (at abstract level) trains a classifier on the change from a fruit's own intake measurement, compares it with the endpoint alone, or tests transfer across batches.

### 2.2 Within-fruit or initial-value approaches (category 2)
- **Per-fruit offsets in kinetic models.** The biological-shift-factor and biological-age framework gives each fruit its own offset (Tijskens 2007 [A]; Hertog 2004 [S]). Initial firmness enters as a per-fruit random parameter in Gwanpua 2013 [S].
- **Harvest index plus shift factor.** Lurie 2013 [S] combines both.
- **Intake-only prediction.** Han 2023 [A] (avocado HSI to days-to-ripe) works. Nordey 2019 [A] (mango, several harvests, years and orchards) found *weak* relationships between harvest traits and post-ripening quality. This bounds what the "baseline alone" arm can be expected to do.
- **Change features in postharvest.** Ratios to initial firmness and delta indices appear in kinetics papers. No paper was found (abstract level) that uses them as classifier inputs and compares them with endpoint-only features.

### 2.3 Paired, Siamese and change-detection networks (category 3)
- **Mature outside postharvest.** Caye Daudt 2018 and Yang 2021 (remote sensing), Hu 2026 FHNet (crop-field bi-temporal change detection, the only agricultural temporal Siamese network found), and Li 2020, Veasey 2020 and Işık 2026 (longitudinal medical imaging).
- **Agricultural and food Siamese networks found are metric or few-shot learners, not temporal change models.** Examples are honey HSI (Zhang 2021) and plant disease (Goncharov 2020).
- **Baseline-referenced classification elsewhere.** EEG baseline subtraction is described as the "most common" preprocessing (Ahmed 2023). Delta-radiomics has its own review (Nardone 2024). Tognotti 2026 shows that baseline normalisation is a leakage hazard.

### 2.4 Durian (category 4)
- **All single-capture.** Kharamat 2020, Phapatanaburi 2025 (KnockNet), Nguyen 2026 (two years, pooled training), Brilliantina 2026 (fruit-grouped CV), Orkweha 2026 COMPAG [T] and Jing 2026 COMPAG [T].
- **"Phase-difference MFCC" is a different idea.** In Orkweha 2026 it is a signal-phase feature, not a between-visit difference (inferred from the title; to be confirmed).
- **Destructive storage studies.** Wiangsamut 2023, Sriyook 1994 and Leyble 2024 document ripening duration, dehiscence and weight loss over storage. These are the physical confounds of a two-visit design.
- **No durian work using a fruit's earlier measurement as a reference was found.**

---

## 3. The five closest works and how the plan differs

| # | Work | Evidence | What it does | How the planned analysis differs |
|---|---|---|---|---|
| 1 | Suzuki 2011, Hort. Res. (Japan) 10.2503/hrj.10.421 | [A] (Japanese) | Persimmon: eating-window start and end predicted from each fruit's harvest-day elasticity index Y0 (errors 0.5–1.3 d and 0.7–3.5 d). | Uses a scalar physical index and a regression equation within one setting, with no transfer test. The plan uses a learned knock embedding, a classification endpoint (Ripe vs Overripe), and leave-one-batch-out transfer against batch-level anchoring. |
| 2 | Işık 2026, Tomography 10.3390/tomography12090129 | [F] | Siamese DenseNet with [f_first; f_last; f_last − f_first]; matched baseline with the final image in both branches; matched tasks holding the final image fixed; patient-grouped CV; fold-nested pretraining. | Different domain (chest X-ray). The architecture and evaluation template are nearly identical, so cite it as the design source rather than claim the architecture. The plan adds cross-batch transfer, which Işık 2026 does not test. |
| 3 | Lurie 2013, PBT 10.1016/j.postharvbio.2012.08.007 | [S] | Peach: I_AD at harvest sets each fruit's biological shift factor and predicts its firmness decay during shelf life. | A kinetic model with an optical index. The plan is a discriminative classifier on acoustic change, with batch-held-out evaluation. |
| 4 | Han 2023, Precis. Agric. 10.1007/s11119-023-10022-y | [A] | Avocado: intake hyperspectral imaging of 476 unripe fruit predicts days-to-ripe (PLSR and DCNN). | This is the "baseline alone" arm (intake only); the plan adds the endpoint and within-fruit change. Han 2023 is the precedent that the "baseline-alone" comparator is a recognised method, not a straw man. |
| 5 | Taniwaki 2010, PBT 10.1016/j.postharvbio.2009.11.007 (and Taniwaki 2009 pears) | [T] | Title: acoustic "ripening speed" and optimum ripeness of melons; probably per-fruit EI decline. | **Unverified.** If it uses per-fruit EI change to classify optimum ripeness, it is the closest acoustic precedent for "change from own baseline" and the hand-crafted comparators (EI or f ratio) must be credited to it. **Obtain the full text before submission.** |

Runners-up: Schouten 2018 (per-fruit repeated acoustic stiffness across several origin batches; kinetic), Ting 2026 preprint (initial-baseline normalisation plus leave-one-unit-out validation, with a strong no-signal baseline; same logic, other domain), Li 2020 (Siamese longitudinal severity), Tognotti 2026 (baseline-normalisation leakage).

---

## 4. Claim wording

### Defensible (if results support it)
- "Using each fruit's intake knock as a reference has a long history in postharvest acoustics (e.g., Suzuki 2011; Lurie 2013 for optical indices). We test, under a pre-registered leave-one-harvest-batch-out protocol, whether a learned within-fruit change representation transfers across batches better than the endpoint knock alone and than batch-level anchoring."
- "To our knowledge (OpenAlex and Crossref, searched November 2026), no durian study has used repeated measurements of the same fruit as a reference, and we found no postharvest study that evaluated a per-fruit intake reference under batch-held-out validation."
- "The paired encoder follows longitudinal Siamese designs from medical imaging (Li 2020; Işık 2026)."
- If the result is negative: "A fruit-specific intake reference did not improve cross-batch transfer over [best comparator]; the endpoint knock already carries the ripeness information that transfers."

### Avoid
- "Novel fruit-specific baseline", "we propose personalised (or subject-specific) ripeness classification", "first Siamese or paired network for fruit" (the agricultural Siamese search was confounded and may have missed work), "first to measure the same fruit repeatedly".
- "Eliminates batch effects" or "removes between-fruit variability". The baseline removes a fruit offset, not batch-by-time interactions.
- "Change detection" as the framing. The task is endpoint classification with a reference, not detection of whether a change occurred.
- Any claim that difference fusion beats concatenation, unless the head is nonlinear and the result is shown (see 5.1).
- "Depositor-labelled ripeness." Keep the existing reconstructed-label wording.

---

## 5. Comparators reviewers will expect (beyond the planned list)

The planned list is: current knock alone; baseline alone; concatenation; hand-crafted differences (dominant-frequency ratio, decay-time difference, stiffness ratio); batch-level anchored standardisation. Add the following.

### 5.1 The concatenation arm may be redundant by construction
[z_t, z_t − z_0] is an invertible linear transform of [z_t, z_0]. With a linear (logistic) head both span the same function class. They differ only through regularisation (weight decay penalises the two parameterisations differently) or a nonlinear head.

Two options:
- **Pre-register them as equivalent** and report the difference arm only as a parameterisation check.
- **Add a pure-change arm** (head sees z_t − z_0 only), which actually removes the fruit-level offset and is the cleaner test of the "baseline" idea.

### 5.2 Additional comparators
1. **Shuffled-baseline control (essential).** Pair each endpoint knock with intake knocks from a *different* fruit in the same batch, ideally the same size class, and repeat over several permutations. If the shuffled arm matches the own-fruit arm, the gain is batch-level, not fruit-specific. This is the only comparator that licenses the word "fruit-specific".
2. **Batch-mean intake reference in the same embedding.** z_t − mean_batch(z_0). This sits between the planned "batch-anchored standardisation" and the shuffled control. Use the same encoder so that only the reference changes.
3. **Matched endpoint-only network.** Following Işık 2026, use the same paired architecture with z_t in both slots (or z_0 set to a constant), so that capacity and training are matched. This guards against the paired model winning only because of a different head or training schedule.
4. **ANCOVA-style residualisation.** Regress z_t (or the hand-crafted endpoint features) on z_0 within training folds and classify the residuals. The trials literature shows that ANCOVA dominates change scores and that percentage change is least efficient (Vickers 2001; Clifton 2019). Reviewers with a statistics background will ask why ratio features were chosen.
5. **Elapsed storage time and capture-day baselines.** The capture-day rule (0.656 endpoint macro-F1) and a days-since-intake-only model. A within-fruit difference encodes elapsed time almost by construction (Kataoka 2018 [A] regresses storage time from acoustics). If Ripe and Overripe endpoints differ systematically in storage duration, z_t − z_0 can win by learning time. Report the paired arm's gain *over the time-only model*, or stratify and match by storage duration.
6. **Hand-crafted physics with mass.** Stiffness ratio (f²m^(2/3))_t / (f²m^(2/3))_0 needs mass at both visits. If endpoint mass is not recorded, say so and use f² ratios only. Also report the stiffness *difference* with intake as a covariate (item 4).
7. **Intake-to-intake noise floor.** Within-fruit test–retest variability of z_0 across the intake knocks against the intake-to-endpoint change. If the noise is comparable to the change, the reference adds variance (cf. regression to the mean, Barnett 2004). Average several intake knocks into one reference embedding and report how many were used.
8. **Fruit-clustered inference.** Fruit-level bootstrap or permutation for the paired-minus-comparator contrast, per held-out batch. With few batches (three, if batch equals maturity level), report per-fold results rather than only the mean.

---

## 6. Known pitfalls

1. **Label leakage through the protocol.**
   - Every intake capture is Unripe by design. If any analysis scores intake captures, the label is given away by the capture role. Score endpoint captures only (as already pre-registered for round 3).
   - If the endpoint class is tied to a fixed storage duration or capture date, the change vector encodes time; see 5.2 item 5.
2. **Transductive encoder leakage.** The knock encoder must be fitted on training batches only. This includes self-supervised or contrastive pretraining and any normalisation statistics. Use the fold-nested regime of Işık 2026. Do not use the held-out batch's intake knocks to fit anything except the fixed reference operation.
3. **Baseline-window leakage.** Per-fruit or per-batch normalisation statistics must exclude endpoint captures from evaluation (Tognotti 2026 [A]: 3–13 pp inflation when the baseline window overlaps evaluation data).
4. **Device mismatch between visits.** Phone model, microphone response, automatic gain control and sampling rate can differ between intake and endpoint. Cross-device spectral colouring can be as large as differences between condition states (Ahmed 2026 [A]).
   - Use same-device pairs only, or include device as a stratum.
   - The project has dual-phone sessions (163 identical sessions per project notes); pair within phone.
   - A difference removes device effects only if the same device is used at both visits.
5. **Operator, knocking force, knocking site and support surface.** These change between visits. Record or randomise them. Robustness to knocking force is claimed by Nguyen 2026 [A], not shown here.
6. **Temperature.** Fruit and ambient temperature differ between field intake and storage. The acoustic elasticity index depends on temperature (Fumuro 2014 [A]; Estrada-Flores 2003 [T]); Kuroki 2006 [A] found within-day melon EI changes not due to ambient temperature. Thermal images exist in this dataset, so surface temperature can be added as a covariate or used to stratify.
7. **Water loss, mass change and dehiscence.**
   - Acoustic firmness responds to dehydration separately from ripening (Mishra 2022 [A]; Molina-Delgado 2009 [S]).
   - Durian loses weight and the husk splits during ripening (Sriyook 1994 [A]; Leyble 2024 [A]).
   - Resonance frequency falls with mass loss and with dehiscence. The change vector mixes softening, water loss and husk cracking. It cannot be called "ripening change" without the mass-normalised or physics arm.
8. **Regression to the mean and the correlation between change and baseline.** Change scores correlate negatively with baseline (Clifton 2019). Fruit with unusually high intake values will show larger apparent drops. Intake-referenced features can therefore look informative because of measurement error. Use ANCOVA-style adjustment and the noise floor in 5.2 item 7.
9. **Selection from incomplete pairs.** Audio coverage is 175/189 fruits and fewer have complete intake and endpoint pairs. The paired analysis is restricted to fruits with both visits, so report the comparator arms on the *same* fruit subset.
10. **Confounding between batch and maturity.** Harvest batch equals maturity level in this dataset. "Cross-batch transfer" is therefore also cross-maturity transfer. State this, and do not attribute gains to batch-effect removal specifically.
11. **Few folds.** Leave-one-batch-out with about three batches gives three estimates. Pre-register the decision rule per fold (for example, wins in at least two of three folds and the pooled clustered CI excludes 0) to avoid post-hoc averaging.

---

## 7. Gaps and next steps for the literature check
- **Get the full text of Taniwaki 2009 (pears) and Taniwaki 2010 (melons)** through library access. These are the most likely to pre-empt the hand-crafted "ratio to intake" comparator and would need crediting. Also get De Ketelaere 2004/2006 and Diezma-Iglesias 2006 to confirm whether any classifier used change from the fruit's own baseline.
- **Read the Suzuki 2011 full text (J-STAGE, open access)** to confirm the form of the Y0 prediction equation. The current reading is from the Japanese abstract only.
- **Optional:** a forward-citation search from Işık 2026, Li 2020 and Suzuki 2011 for any fruit or food application. This was not done here.
