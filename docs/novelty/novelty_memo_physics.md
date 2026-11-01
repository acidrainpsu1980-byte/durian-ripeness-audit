# Physics-guided acoustic features for durian ripeness: novelty memo

**Bottom line.** The physical feature set is not new, either for fruit in general or for durian. Resonance frequency, the mass-normalised stiffness coefficient f²·m^(2/3), elasticity indices and damping or loss coefficients have been standard acoustic-firmness descriptors since the 1970s. Durian has its own precedents: a resonance-and-mass "frequency index" for Monthong maturity (Terdwongworakul et al. 1998, AGRIS record, no DOI), laser-Doppler elastic constants ([Kongrattanaprasert 2001](https://doi.org/10.1109/iscas.2001.921342)), an exponential fall in natural frequency during postharvest ripening ([Yuwana 2021](https://doi.org/10.1063/5.0037558)), size-aware "acoustic properties" versus ripeness ([An 2024](https://doi.org/10.1016/j.postharvbio.2024.113103)), and peak-frequency/spectral-centroid features tested over two harvest years ([Nguyen 2026](https://doi.org/10.1016/j.rineng.2026.110709)). No retrieved work does the specific evaluation proposed here: a leave-one-batch-out test of whether mass-normalised resonance and damping parameters from smartphone knocks transfer across independently acquired postharvest batches better than a spectrogram CNN, alone, as a hybrid, and with intake-anchored calibration. Under COMPAG's scope rule, any novelty has to rest on that evaluation and the transfer result, not on the features.

**Scope and limits (read first).**
- Sources: about 70 queries across OpenAlex, Crossref and web search; Semantic Scholar was mostly rate-limited (HTTP 429), so its coverage is partial.
- One durian precedent (Terdwongworakul 1998) has no DOI; it was retrieved as an AGRIS record and is listed as such.
- **No full texts were obtainable.** All 61 judgements rest on abstracts (42), publisher or indexer abstract snippets (9), or title and metadata only (9), plus the AGRIS abstract. Every claim below about what a paper did or did not test is therefore provisional. That applies especially to "transfer not tested", because hold-out designs are often described only in the methods section.
- One durian free-vibration study (Neamsorn & Terdwongworakul 2004) is known only from a secondary citation in [Timkhum 2012](https://doi.org/10.1016/j.jfoodeng.2012.05.018). It is not in the table or the CSV.
- Non-English and Thai-language theses and proceedings without DOIs were not searched systematically. That is the most likely place for further durian resonance work.

## (a) What is new and what is not

### Not new: the physical parameters

Resonance-based firmness rests on Cooke's elastic-sphere theory ([Cooke 1972](https://doi.org/10.13031/2013.38074); three-media version [Cooke & Rand 1973](https://doi.org/10.1016/0021-8634%2873%2990023-1)). Microphone-based impulse methods ([Yamamoto 1980](https://doi.org/10.1111/j.1745-4603.1980.tb00312.x)) and finite-element work then established that the first flexural and second longitudinal modes relate to elastic modulus through f²·m^(2/3) ([Chen 1993](https://doi.org/10.13031/2013.28529)). Sonic stiffness coefficients track tissue stiffness more closely than puncture force ([Abbott 1995](https://doi.org/10.13031/2013.27971); [Armstrong 1990](https://doi.org/10.13031/2013.31480)). Damping is equally established: the avocado damping ratio rises monotonically during ripening ([Galili 1998](https://doi.org/10.13031/2013.17164)), and LDV studies define both S = f₂²·m^(2/3) and a half-power-bandwidth loss coefficient ([Terasaki 2001](https://doi.org/10.13031/2013.2291)). Stiffness and elasticity coefficients have also been applied to mandarin storage ([Wang 2006](https://doi.org/10.1111/j.1745-4557.2006.00081.x)). The Japanese "elasticity index" (f²·m^(2/3)) has been used to time optimum eating ripeness in pear and melon, including per-fruit time courses ([Taniwaki 2009](https://doi.org/10.1016/j.postharvbio.2008.08.004); [Taniwaki 2010](https://doi.org/10.1016/j.postharvbio.2009.11.007)). The proposed "acoustic stiffness coefficient" is therefore a textbook quantity, and the paper should present it as such.

### Not new for durian or thick-rind tropical fruit

The durian literature already contains physical acoustic parameters:

- **Mass-normalised resonance.** Terdwongworakul et al. (1998, AGRIS) tapped on-tree Monthong fruit and related f² to m^(2/3). Within one maturity stage the relation was exponential, so they proposed the index (2/3)·f²·ln m, which fell with days after blossom.
- **Lower resonance with maturity.** Neamsorn & Terdwongworakul (2004, secondary only) reported that resonant frequency falls as the fruit matures, but correlates only weakly with dry matter (r = 0.45).
- **Laser-Doppler measurements.** [Kongrattanaprasert 2001](https://doi.org/10.1109/iscas.2001.921342) measured vibration velocity, elastic constants and PSD between the prickles and related them to dry weight. The companion paper did spectral template matching ([Kongrattanaprasert 2001b](https://doi.org/10.1142/s0218488501001162)).
- **Natural frequency during ripening.** [Yuwana 2021](https://doi.org/10.1063/5.0037558) found that durian natural frequency decays exponentially during ripening, with single- and double-peak spectra. This is the closest physical result for postharvest ripening.
- **Size-aware acoustic properties.** [An 2024](https://doi.org/10.1016/j.postharvbio.2024.113103) collected acoustic signals alongside transmittance spectra across ripeness stages, analysed how durian size and structure affect both, and optimised acoustic parameters by their correlation with ripeness. The size correction is circumference-based and, as far as the snippets show, was applied to the spectra.
- **Ultrasonic properties.** [Haryanto 2001](https://doi.org/10.1016/s1474-6670%2817%2934148-4) measured ultrasonic velocity and attenuation of mature versus immature fruit.
- **Peak frequency over two harvest years.** The Can Tho group used peak frequency, spectral centroid and MFCC for on-tree durian maturity over two orchards and two harvest years. Pooled ("mixed-domain") training improved robustness to inter-annual variability ([Nguyen 2026](https://doi.org/10.1016/j.rineng.2026.110709)), and the same features were used for jackfruit ([Tran 2026](https://doi.org/10.1016/j.atech.2026.101932)).
- **Electromagnetic analogue.** Natural-frequency tracking also exists in an electromagnetic form ([Tantisopharak 2016](https://doi.org/10.1109/tap.2016.2533660)), which reviewers may conflate with the acoustic method.

A claim such as "first physics-based or interpretable acoustic model for durian" is therefore false.

### Not new: smartphones, cross-season data, physics features with ML

- **Smartphone recording.** Smartphone thump recording of a thick-rind fruit dates to [Zeng 2013](https://doi.org/10.1007/s00779-013-0706-7) for watermelon. Durian knocks were recorded on a smartphone by [Kharamat 2020](https://doi.org/10.1109/ieecon48109.2020.229571). [Li 2025](https://doi.org/10.1016/j.jfca.2025.108496) studied variety, recording position and tapping intensity as nuisance factors for mobile-phone watermelon classification.
- **Multi-season data.** Multi-season acoustic data exist for kiwifruit (three seasons and many orchards; [Sneddon 2024](https://doi.org/10.1080/01140671.2024.2314496)), pomelo (two seasons pooled; [Phuangsombut 2020](https://doi.org/10.1111/ijfs.14746)) and avocado (two origins and two ripening temperatures; [Landahl 2020](https://doi.org/10.1016/j.biosystemseng.2020.04.001)).
- **Physics features with ML.** Physically motivated dynamic features (rise time, overshoot, damping coefficient) have been combined with ML feature selection for kiwifruit ([Abdanan Mehdizadeh 2025](https://doi.org/10.1016/j.atech.2025.101606)).

### Plausibly new (provisional, abstract-level)

1. **Held-out batch transfer of physical parameters for postharvest durian ripeness stages.** No retrieved durian study holds out a whole acquisition batch, harvest or season. The two-year durian work pooled the years ([Nguyen 2026](https://doi.org/10.1016/j.rineng.2026.110709)), and the CNN line of work (HMM, N-gram, MFCC+NN, CNN, KnockNet, RFAConv-KAN, phase-aware cepstra) reports random or fruit-level splits as far as the abstracts show.
2. **A direct comparison under batch shift.** Interpretable modal parameters (f, damping or Q, f²·m^α with harvest mass), a log-mel CNN and their hybrid have not been compared under the same batch-shift protocol for any thick-rind fruit found here.
3. **Intake-anchored calibration of physical parameters.** This means referencing each batch's knocks to its own harvest-day (known Unripe) knocks. Relative or temperature-referenced stiffness exists in other forms: per-fruit time courses ([Taniwaki 2010](https://doi.org/10.1016/j.postharvbio.2009.11.007)) and temperature-correction models ([Challies 2010](https://doi.org/10.17660/actahortic.2010.877.248)). The first track also found that known-class anchoring is established in fault diagnosis and e-nose drift correction. The defensible piece is narrower: batch-level anchoring of physical acoustic parameters, using process-labelled intake scans, in postharvest grading.

## (b) Closest works

"Transfer" means a train/test split across season, batch, orchard, device or cultivar. "Pooled" means data from several conditions were combined, but no condition was held out.

| # | Work | Year | Fruit | Parameters | Transfer tested? | How the proposal differs |
|---|---|---|---|---|---|---|
| 1 | Terdwongworakul et al. (AGRIS, no DOI) | 1998 | Durian (Monthong), on-tree | f, m; index (2/3)f² ln m | No (single season, 6 trees) | Postharvest ripeness, smartphone, batch hold-out |
| 2 | [Kongrattanaprasert 2001](https://doi.org/10.1109/iscas.2001.921342) | 2001/02 | Durian | LDV velocity, elastic constants, PSD vs dry weight | Not reported | Microphone knocks, ripeness not maturity |
| 3 | [Kongrattanaprasert 2001b](https://doi.org/10.1142/s0218488501001162) | 2001 | Durian | LDV high-frequency spectrum template; ultrasonic | Not reported | Interpretable modal parameters, transfer |
| 4 | [Haryanto 2001](https://doi.org/10.1016/s1474-6670%2817%2934148-4) | 2001 | Durian (Sunan) | Ultrasonic velocity, attenuation | No | Audio-band knocks |
| 5 | [Yuwana 2021](https://doi.org/10.1063/5.0037558) | 2021 | Durian, mango, sawo | Natural frequency vs ripening time (exponential) | No | Multi-batch, classification, mass normalisation |
| 6 | [An 2024](https://doi.org/10.1016/j.postharvbio.2024.113103) | 2024 | Durian | Acoustic properties optimised vs ripeness; circumference correction (spectra) | Not reported (abstract) | Batch hold-out; damping; CNN comparison |
| 7 | [Nguyen 2026](https://doi.org/10.1016/j.rineng.2026.110709) | 2026 | Durian (Ri6), on-tree | Peak frequency, spectral centroid, MFCC (28 features) | Pooled 2 years ("mixed-domain") | Held-out batch, mass-normalised stiffness, anchoring |
| 8 | [Tran 2026](https://doi.org/10.1016/j.atech.2026.101932) | 2026 | Jackfruit, on-tree | Peak frequency, spectral centroid, MFCC | No (tapping position examined) | Durian, postharvest, transfer |
| 9 | [Phothisonothai 2010](https://doi.org/10.1109/isspa.2010.5605440) | 2010 | Durian | Knock fractal dimension, by weight class | No (weight-stratified) | Physical rather than fractal features |
| 10 | [Kharamat 2020](https://doi.org/10.1109/ieecon48109.2020.229571) | 2020 | Durian | Smartphone knocks, CNN | No | Physics vs CNN under shift |
| 11 | [Phapatanaburi 2025](https://doi.org/10.1016/j.rineng.2025.108170) | 2025 | Durian | KnockNet: MFCC + group-delay cepstra | No (fold CV) | Interpretable parameters, transfer |
| 12 | [Brilliantina 2026](https://doi.org/10.1038/s41598-026-61395-9) | 2026 | Durian | Gas, thermal, scalar acoustic dB | No (fruit-grouped CV) | Spectral modal parameters |
| 13 | [Cooke 1972](https://doi.org/10.13031/2013.38074) | 1972 | General | Elastic-sphere resonance theory | n/a | Source of f²m^(2/3) |
| 14 | [Chen 1993](https://doi.org/10.13031/2013.28529) | 1993 | Apple, pineapple | FE modes; f²m^(2/3) | n/a | Mode identity for spiny, thick-rind fruit untested |
| 15 | [Abbott 1995](https://doi.org/10.13031/2013.27971) | 1995 | Apple | Sonic stiffness coefficients | Cultivars compared, not held out | Durian, batch transfer |
| 16 | [Galili 1998](https://doi.org/10.13031/2013.17164) | 1998 | Avocado | Firmness index, damping ratio, centroid | No | Damping as ripening cue |
| 17 | [Terasaki 2001](https://doi.org/10.13031/2013.2291) | 2001 | Kiwifruit (LDV) | S = f₂²m^(2/3), loss coefficient | No | Microphone; transfer |
| 18 | [Wang 2006](https://doi.org/10.1111/j.1745-4557.2006.00081.x) | 2006 | Mandarin | f, stiffness coefficient, elasticity coefficient | No | — |
| 19 | [Taniwaki 2010](https://doi.org/10.1016/j.postharvbio.2009.11.007) | 2010 | Melon | Elasticity index per-fruit time course | No | Per-batch anchoring vs per-fruit tracking |
| 20 | [Zhang 2019](https://doi.org/10.1016/j.postharvbio.2018.11.002) | 2019 | Pear (shapes) | Four resonance and two propagation indices | Shape variation | Single-frequency index weakest for irregular fruit |
| 21 | [Fumuro 2014](https://doi.org/10.21273/hortsci.49.6.791) | 2014 | Mango | Elasticity index vs temperature, wrapping, size | Conditions compared | EI size-insensitive but temperature-sensitive |
| 22 | [Challies 2010](https://doi.org/10.17660/actahortic.2010.877.248) | 2010 | Kiwifruit | Acoustic stiffness vs fruit temperature | Correction model | Temperature as batch confounder |
| 23 | [Hertog 2004](https://doi.org/10.1016/j.postharvbio.2004.01.005) | 2004 | Tomato | Acoustic firmness vs temperature and VPD | Conditions compared | Turgor and water-loss confound |
| 24 | [Sneddon 2024](https://doi.org/10.1080/01140671.2024.2314496) | 2024 | Kiwifruit | Acoustic stiffness (AWETA) | 3 seasons, many orchards | Closest multi-season acoustic precedent |
| 25 | [Phuangsombut 2020](https://doi.org/10.1111/ijfs.14746) | 2020 | Pomelo (thick rind) | Resonant frequency + visible reflectance | Pooled 2 seasons | Held-out batch instead of pooling |
| 26 | [Landahl 2020](https://doi.org/10.1016/j.biosystemseng.2020.04.001) | 2020 | Avocado (LDV) | LDV resonance | 2 origins / 2 temperatures (design unclear) | — |
| 27 | [Zeng 2013](https://doi.org/10.1007/s00779-013-0706-7) | 2013 | Watermelon | Mobile-device sub-band energy, SVM | No | Smartphone precedent |
| 28 | [Li 2025](https://doi.org/10.1016/j.jfca.2025.108496) | 2025 | Watermelon | Mobile phone; variety, position, intensity factors | Factor analysis | Nuisance factors for smartphone knocks |
| 29 | [Abdanan Mehdizadeh 2025](https://doi.org/10.1016/j.atech.2025.101606) | 2025 | Kiwifruit | Transfer-function damping, rise time + ML | No | Closest "physics features + ML" |
| 30 | [Arwatchananukul 2024](https://doi.org/10.1016/j.atech.2024.100601) | 2024 | Pineapple | Dominant resonance + factor analysis | No | Thick-rind tropical analogue |

## (c) Baselines and normalisations to compare against

1. **Raw dominant or first-mode frequency f**, per knock and as the fruit-level median ([Yamamoto 1980](https://doi.org/10.1111/j.1745-4603.1980.tb00312.x); [Yuwana 2021](https://doi.org/10.1063/5.0037558)).
2. **f²·m^(2/3)**, using harvest mass ([Cooke 1972](https://doi.org/10.13031/2013.38074); [Chen 1993](https://doi.org/10.13031/2013.28529); [Terasaki 2001](https://doi.org/10.13031/2013.2291); elasticity index in [Taniwaki 2009](https://doi.org/10.1016/j.postharvbio.2008.08.004)).
3. **f²·m and an empirically fitted exponent f²·m^α**, reported with α. This matters because the durian relation between f² and m^(2/3) was exponential rather than proportional (Terdwongworakul 1998, AGRIS). Include the Terdwongworakul index (2/3)·f²·ln m as a durian-specific baseline.
4. **Circumference- or volume-based size normalisation** instead of mass ([An 2024](https://doi.org/10.1016/j.postharvbio.2024.113103) used circumference correction for spectra). Harvest circumference is available in the fruit log.
5. **Damping: decay time, damping ratio, Q or half-power loss coefficient** ([Galili 1998](https://doi.org/10.13031/2013.17164); [Terasaki 2001](https://doi.org/10.13031/2013.2291); [Abdanan Mehdizadeh 2025](https://doi.org/10.1016/j.atech.2025.101606)).
6. **Multi-peak indices**, i.e. second peak or dual-frequency, for non-spherical, double-peak fruit ([Zhang 2019](https://doi.org/10.1016/j.postharvbio.2018.11.002); [Yuwana 2021](https://doi.org/10.1063/5.0037558) for double peaks in durian).
7. **Shallow ML on peak frequency, spectral centroid and MFCC**, the strongest durian-specific feature baseline ([Nguyen 2026](https://doi.org/10.1016/j.rineng.2026.110709)).
8. **MFCC + NN and CNN knock classifiers** ([Khunarsa 2016](https://doi.org/10.4028/www.scientific.net/amm.855.75); [Kharamat 2020](https://doi.org/10.1109/ieecon48109.2020.229571); [Phapatanaburi 2025](https://doi.org/10.1016/j.rineng.2025.108170)), plus the paper's own log-mel CNN.
9. **Pooled multi-batch training ("mixed-domain") without anchoring**, as the transfer baseline that the literature actually uses ([Nguyen 2026](https://doi.org/10.1016/j.rineng.2026.110709); [Phuangsombut 2020](https://doi.org/10.1111/ijfs.14746)).
10. **Relative change from a reference measurement**: per-fruit EI time course ([Taniwaki 2010](https://doi.org/10.1016/j.postharvbio.2009.11.007)) or a temperature-referenced stiffness ([Challies 2010](https://doi.org/10.17660/actahortic.2010.877.248)). This should be contrasted with intake-batch anchoring.
11. **Trivial and leakage baselines (no literature reference needed)**: harvest weight alone, circumference alone, device identity, and capture day or storage time. If capture day predicts the label, anchored models must be scored on endpoint captures only, as noted in the first track.

## (d) Novelty claim

**Defensible (one sentence).** "Resonance, damping and mass-normalised stiffness are established acoustic firmness descriptors, including for durian; we test, for the first time to our knowledge, whether these interpretable parameters extracted from smartphone knock recordings transfer across independently acquired postharvest harvest batches better than a log-mel CNN, alone, in a hybrid and with intake-anchored batch calibration, under leave-one-batch-out evaluation."

Keep "to our knowledge", and support it with the search scope in the cover letter.

**Wording that would overclaim:**
- "first physics-based / physics-informed / interpretable acoustic model for durian" (contradicted by Terdwongworakul 1998, Kongrattanaprasert 2001, Yuwana 2021, An 2024)
- "we propose a novel acoustic stiffness coefficient" (Cooke 1972 and its descendants)
- "physics-informed neural network" (implies a PDE or physics-loss constraint; use "physics-guided features" or "hybrid")
- "first to show that durian resonance falls with ripening or maturity" (Neamsorn 2004 secondary; Yuwana 2021)
- "first cross-season durian acoustic study" (Nguyen 2026 covered two harvest years, though pooled)
- "batch-invariant", "device-independent" or "generalises to any orchard" (three batches and two phone families cannot support this)
- "first smartphone acoustic ripeness method" (Zeng 2013; Kharamat 2020)
- any wording that physical features "explain" ripening physiology. Acoustic firmness reflects tissue stiffness plus turgor ([Hertog 2004](https://doi.org/10.1016/j.postharvbio.2004.01.005)), not ripening per se.

## (e) Methodological pitfalls reported in the literature

**Temperature at measurement.** Acoustic stiffness changes with fruit temperature, so fruit must be equilibrated or a correction model applied ([Challies 2010](https://doi.org/10.17660/actahortic.2010.877.248)). The elasticity index also responds to storage temperature and film wrapping ([Fumuro 2014](https://doi.org/10.21273/hortsci.49.6.791)), and LDV texture signals differ between storage temperatures ([Muramatsu 2000](https://doi.org/10.21273/jashs.125.1.120)). Batches captured under different ambient conditions will carry this as a confounder. Record room or fruit temperature if the dataset has it, and otherwise state it as a limitation.

**Water loss and turgor.** Non-invasive acoustic firmness depends on vapour-pressure deficit as well as temperature, unlike puncture firmness ([Hertog 2004](https://doi.org/10.1016/j.postharvbio.2004.01.005)). For durian this has two consequences. First, storage-dependent mass loss affects both f and the true m, so S computed with harvest mass is an approximation that drifts with storage. Second, end-of-storage weight must stay excluded, as already planned.

**Shape, size and mode identity.** Fruit shape alters the acoustic response ([Jancsók 2001](https://doi.org/10.1016/s0925-5214%2801%2900098-9); [Chen 1993](https://doi.org/10.13031/2013.28529)). The conventional single-frequency index performed worst for irregular pears ([Zhang 2019](https://doi.org/10.1016/j.postharvbio.2018.11.002)), and irregular avocado shape reduced the consistency of test results ([Galili 1998](https://doi.org/10.13031/2013.17164)). Durian is large and lobed, and spectra can be single- or double-peaked ([Yuwana 2021](https://doi.org/10.1063/5.0037558)). The airborne "dominant frequency" may therefore not be the first flexural mode. Report the peak-picking rule and the agreement between the dominant and first-mode estimates. The durian mass exponent may also not follow Cooke scaling (Terdwongworakul 1998).

**Thick rind and spines.** Durian's thickness of peel, prickly skin, non-uniform shape and large size complicate vibration and ultrasonic measurement, which is why excitation was applied between the prickles ([Kongrattanaprasert 2001b](https://doi.org/10.1142/s0218488501001162)). The shell's mesocarp and thorns absorb energy, as quasi-static compression tests showed ([Ha 2020](https://doi.org/10.1016/j.jmbbm.2019.103603)). Complex structure and size also limit acoustic and spectral ripeness detection ([An 2024](https://doi.org/10.1016/j.postharvbio.2024.113103); review in [Wang 2023](https://doi.org/10.3390/agriculture13091838)). A rind-dominated response may track rind rather than pulp changes. This is also why raw frequency correlated only weakly with dry matter in Neamsorn 2004 (secondary).

**Knock force, position and support.** Tapping intensity, recording position and variety all affect mobile-phone watermelon features ([Li 2025](https://doi.org/10.1016/j.jfca.2025.108496)). Sensor geometry, fruit mass and impact conditions mattered for piezo resonance measurements ([Shmulevich 1996](https://doi.org/10.13031/2013.27595)), and support condition (growing position versus a hard base) was tested for field watermelon ([Stone 1996](https://doi.org/10.13031/2013.27743)). By contrast, tapping position had limited influence for jackfruit ([Tran 2026](https://doi.org/10.1016/j.atech.2026.101932)). Per the ResearchGate snippet of [Yuwana 2021](https://doi.org/10.1063/5.0037558), excitation method, location and holding did not shift resonant frequencies but did change amplitudes of higher peaks. Frequency parameters should therefore be more robust than amplitude- or energy-based ones. That is a testable prediction, and it is worth stating.

**Smartphone microphones.** Apps and devices differ substantially in measured sound levels ([Kardous 2014](https://doi.org/10.1121/1.4865269)). That study covers SPL only, so for spectral shape this is a caution, not a quantified bias. Device-specific frequency response, low-frequency roll-off, automatic gain control and noise suppression are plausible further confounders (no direct fruit-acoustics reference found). The iPhone/Android split in the dataset should be treated as a factor: estimate f within device and report the device × batch interaction. The observed frequency range (696 → 419 → 368 Hz, iPhone) is plausibly above phone low-frequency roll-off, but this should be checked per device rather than assumed.

**Biological variance and validation design.** Most acoustic firmness studies cover a single season ([Sneddon 2024](https://doi.org/10.1080/01140671.2024.2314496)). Pooled multi-season models ([Phuangsombut 2020](https://doi.org/10.1111/ijfs.14746); [Nguyen 2026](https://doi.org/10.1016/j.rineng.2026.110709)) do not show held-out transfer. Fruit-level grouping of repeated knocks is necessary ([Brilliantina 2026](https://doi.org/10.1038/s41598-026-61395-9) used fruit-grouped CV). Individual fruit also ripen at different speeds ([Taniwaki 2010](https://doi.org/10.1016/j.postharvbio.2009.11.007)), so day-of-capture labels carry within-class heterogeneity.

## Reference file

[novelty_refs_physics.csv](novelty_refs_physics.csv) lists all 61 works with their category (1 = acoustic firmness physics and parameters; 2 = robustness, normalisation, smartphone, ML and pitfalls; 3 = durian and thick-rind tropical fruit), verification status, access level and a one-line note.
