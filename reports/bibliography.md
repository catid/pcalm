# Checked bibliography and code inventory

Audit date: 2026-10-29 UTC. URLs were fetched live from `foureyes`; arXiv metadata and publisher pages are recorded as returned on that date. A listed item without a verified code link is marked accordingly rather than inferred.

## Primary method sources

| ID | Verified title | Authors | Canonical source/status | Code/source evidence |
|---|---|---|---|---|
| PC-ALM | **Augmented Lagrangian Predictive Coding** | Jeffrey Seely; Julian Gould | arXiv:2605.31022v1, submitted 2026-05-29, https://arxiv.org/abs/2605.31022 | No public code link was present in the arXiv record during this audit. |
| ePC | **ePC: Fast and Deep Predictive Coding in Digital Simulation** | Cédric Goemaere; Gaspard Oliviers; Rafal Bogacz; Thomas Demeester | ICML 2026, PMLR 306, pp. 35407–35442, published 2026-09-29, https://proceedings.mlr.press/v306/goemaere26a.html | Official repository https://github.com/cgoemaere/error_based_PC, HEAD/master `95c555197699f6b4d15452451da15049946b49ca` (shallow clone audit). README identifies this as official code and links arXiv:2505.20137. |
| Meta-PCN | **Stable and Scalable Deep Predictive Coding Networks with Meta-Prediction Errors** | Myoung Hoon Ha; Hyunjun Kim; Yoondo Sung; Youngha Jo; Min Kang; Sang Wan Lee | ICLR 2026, conference paper, https://proceedings.iclr.cc/paper_files/paper/2026/hash/319e084cd710c3db50a9f9dd7a02a9a8-Abstract-Conference.html | No code repository was exposed by the publisher page/arXiv search in this audit. |
| Deep PC / iPC | **Towards the Training of Deeper Predictive Coding Neural Networks** | Chang Qi; Matteo Forasassi; Thomas Lukasiewicz; Tommaso Salvatori | ICML 2026, PMLR 306, pp. 100334–100362, published 2026-09-29, https://proceedings.mlr.press/v306/qi26f.html | Official implementation https://github.com/changqi97/DeepPCNs, HEAD/main `99472b071f4bec483ddfe7ed37c6c3c10680d8bc`. README states ICML 2026 and includes PC/iPC training scripts. |
| Stable parameterization | **On the Infinite Width and Depth Limits of Predictive Coding Networks** | Francesco Innocenti; El Mehdi Achour; Rafal Bogacz | ICML 2026, PMLR 306, pp. 49761–49799, published 2026-09-29, https://proceedings.mlr.press/v306/innocenti26a.html | No code link exposed in the publisher record during this audit. |
| WF-Act-PC | **Weight Feedback Computes the Jacobian Transpose Locally in Modern Deep Networks** | Junlong Shen; Xingyu Li | arXiv:2607.13380v1, submitted 2026-07-15, https://arxiv.org/abs/2607.13380 | arXiv record links https://github.com/jlshen025/pcax; HEAD/main `95e37f7fb5ad6607a0ffd54110bb133e7105cef9`. Treat as reference code lead, not proof of Transformer reproduction. |
| Error Highways | **Error Highways: Scaling Predictive Coding to Very Deep Networks** | Amirhossein Mohammadi; Alexander G. Ororbia | arXiv:2606.22744v1, submitted 2026-06-22, https://arxiv.org/abs/2606.22744 | No code link exposed in arXiv record during this audit. |
| Closed-form precision PC | **Closed-form predictive coding via hierarchical Gaussian filters** | Aleksandrs Baskakovs; Sylvain Estebe; Kenneth Enevoldsen; Kristoffer Nielbo; Chris Mathys; Nicolas Legrand | arXiv:2605.20293v1, submitted 2026-05-19, https://arxiv.org/abs/2605.20293 | No code link exposed in arXiv record during this audit. |
| Direct FA | **Accelerated Predictive Coding Networks via Direct Kolen-Pollack Feedback Alignment** | Davide Casnici; Martin Lefebvre; Justin Dauwels; Charlotte Frenkel | arXiv:2602.15571v1 (updated 2026-03-07), https://arxiv.org/abs/2602.15571 | No code link exposed in arXiv record during this audit. |
| Stream/amortized init | **Faster Predictive Coding Networks via Better Initialization** | Luca Pinchetti; Simon Frieder; Thomas Lukasiewicz; Tommaso Salvatori | arXiv:2601.20895v1, submitted 2026-01-28, https://arxiv.org/abs/2601.20895 | No code link exposed in arXiv record during this audit. |

## Optional and follow-on anchors

| ID | Verified title | Authors | Canonical source/status |
|---|---|---|---|
| Bregman PC | **Analog-Friendly Predictive Coding without Activation Derivatives** (activation-matched Bregman PC) | Francesco Innocenti | arXiv:2609.32350v1, submitted 2026-09-26, https://arxiv.org/abs/2609.32350. The user-provided “Activation-matched Bregman PC” label is a method description; the verified title is the one shown here. |
| Temporal PC | **Persistence, Exposure, and Credit Routing in Linear Temporal Predictive Coding** | Akira Tanase; Julian Gould; Jeffrey Seely | arXiv:2610.36412v1, submitted 2026-10-26; accepted to NeurReps workshop per arXiv comment, https://arxiv.org/abs/2610.36412. This is a contemporaneous preprint/workshop item, not a claimed full conference publication. |
| SpatialPC/APD | **SpatialPC: Deep Computation with Sublinear Local Credit Propagation** | Benjamin Prada; Shion Matsumoto; Ankur Arjun Mali | arXiv:2610.36139v1, submitted 2026-10-26, https://arxiv.org/abs/2610.36139 |
| μPC | **μPC: Scaling Predictive Coding to 100+ Layer Networks** | Francesco Innocenti; El Mehdi Achour; Christopher L. Buckley | NeurIPS 38 (2025), DOI 10.52202/085713-2923, pp. 87414–87455, publisher page https://proceedings.neurips.cc/paper_files/paper/2025/hash/7ea5ad70c984262c064ee9859f7e5f56-Abstract-Conference.html |

## Repository/control audit

- **ePC** (`cgoemaere/error_based_PC`, commit above): README says the repository is official code for ePC and that `SO`/`EO` denote state/error optimization. `pc_e.py` uses autograd on trainable error tensors during error inference (`E.backward()`), then computes local weight energy with detached block inputs in `E_local`; this is a local-autodiff implementation, not strict no-autograd. It targets MNIST/FashionMNIST (and a CIFAR branch), not a causal language model.
- **DeepPCNs** (`changqi97/DeepPCNs`, commit above): README lists JAX/pcx implementation for image datasets and explicit PC/iPC scripts; no FineWeb/Transformer claim was found.
- **Noise** (`qlabs-eng/noise`, HEAD/main `ebee4e37e0df78d5cf7d9a9d606089224b46252c`): README describes a forward-only, no-autograd 37.7M-parameter Transformer on a 30M-token FineWeb-derived set and fixed BPE-4096 tokenizer. This is a software/control reproduction lead, not a published PC method. Its rules forbid autograd and chained nonlinear credit; any port must retain the commit and deviations.

## Reproducibility cautions

The arXiv identifiers and publisher pages above were reachable at audit time, but most 2026 items are contemporary preprints or 2026 proceedings. Claims in this project should cite the exact URL/version and distinguish faithful reproduction, Transformer adaptation, and new proposal. Repository HEAD hashes are point-in-time audit pins; recheck before final runs if repositories move.
