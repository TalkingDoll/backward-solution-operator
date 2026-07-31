# Experimental Details

## Purpose and terminology

This document is the reproducibility record for all six experiments. Sections
2–7 contain the governing system or dataset, preprocessing and split, fixed
configuration, evaluation protocol, reported result, and interpretation
limits needed to draft a Numerical Experiments section. Section 8 maps every
publication figure to its notebook and provides its copy-ready caption.
Section 9 summarizes the principal result and limitation of each benchmark.

Numerical values come from executed notebook outputs and the reference
artifacts used to create the committed figures. Large raw datasets,
checkpoints, and runtime result bundles are excluded from Git unless stated
otherwise. The proposed method is the **backward solution operator (BSO)**;
EDMD/Galerkin projection is its finite-dimensional construction step, not a
separate method name. Unless stated otherwise, experiments use random seed
`2026`; the OU learned-dictionary stages use seeds `2033–2038`.

## 1. Common computational framework

For a Markov process, the finite-time Kolmogorov backward solution operator is

```math
U_\tau f(x)=\mathbb{E}[f(X_\tau)\mid X_0=x].
```

The experiments learn its action on observable fields. Given samples separated
by a horizon $\tau$, the common pipeline is:

1. Construct one-step and, where used, two-step pairs `(u0, uT, u2T)`.
2. Train a neural operator to approximate `u0 -> uT`.
3. Build an observable dictionary from analytic functions, a neural dictionary, polynomially lifted PCA coordinates, EOF coordinates, or snapshot modes.
4. Project the learned evolution into that dictionary with weighted EDMD/Galerkin least squares.
5. Propagate the resulting finite-dimensional BSO and compare its spectrum or forecasts with appropriate references and baselines.

Suggested background citations for the manuscript are the original
[Fourier Neural Operator paper](https://iclr.cc/virtual/2021/poster/3281),
the [EDMD/Galerkin formulation](https://arxiv.org/abs/1408.4408), and the
[kernel Koopman formulation](https://arxiv.org/abs/1411.2260). The EEG
EDMD-DL comparator has its separate method citation in Section 7.4.

The default finite-dimensional matrix is

```math
K = (G + \alpha I)^{-1} A,
```

where `G` is the weighted dictionary Gram matrix and `A` couples the dictionary to its evolved image. `solver_edmd.py` trains a dictionary with an invariance residual. `solver_resdmd.py` keeps the same EDMD matrix but trains with refreshed eigenpair residuals.

The shared OU, Triple-well, and SST Phase-A training paths, as well as the
recorded Lorenz recipes, use float32, AdamW, cosine annealing with `T_max`
equal to the epoch budget and `eta_min=0`, gradient-norm clipping at 5, and a
seeded random validation split. EEG is an explicit exception: it trains on
all 104 transition pairs for a fixed epoch budget, without a validation split
or best-checkpoint selection. Unless a section states otherwise, Adam/AdamW
use PyTorch defaults `betas=(0.9,0.999)`, `eps=1e-8`, and `amsgrad=False`.
With
`F` denoting the neural operator and `ell_rel` the experiment's weighted or
unweighted mean relative `L2`, its training objective is

```math
\mathcal{L}_{\mathrm{op}}
=\ell_{\mathrm{rel}}(F u_0,u_\tau)
+w_{\mathrm{sg}}\ell_{\mathrm{rel}}(F(Fu_0),u_{2\tau})
+w_{\mathrm{const}}\mathrm{mean}(F\mathbf{1}-\mathbf{1})^2 .
```

This objective applies to OU, Triple-well, Lorenz, and SST; EEG uses the
squared relative-$L^2$ objective in Section 7.4. The constant term is active
only where a nonzero weight is listed. Where a
validation split is used, checkpoint selection uses the one-step validation
`ell_rel(Fu_0,u_tau)` alone, without the semigroup or constant term. Phase-B
neural-dictionary refinement uses Adam and the complete fitting dataset, with
no separate dictionary-validation split. Phase-A and Phase-B optimization,
including a refreshed projected target inside Phase B, use float32. Final
reported EDMD matrices, eigendecompositions, polynomial/EOF maps, and metrics
use float64 unless an artifact is explicitly stored in float32.

The `neuralop.models.FNO` instances use `GridEmbeddingND` with each grid coordinate mapped to `[0,1]`, GELU, lifting/projection ratio 2, channel-MLP expansion 0.5, linear FNO skips, soft-gating channel-MLP skips, and no normalization, domain padding, dropout, spectral factorization, or outer residual wrapper. These are the `neuraloperator 1.0.2` constructor defaults left unchanged by the experiment wrappers.

Where linear-combination augmentation is enabled, each mixed sample draws the configured number of training indices with replacement. Independent standard-Gaussian coefficients are normalized to unit Euclidean norm. The mixed pair is accepted only when its target norm exceeds 0.25 times the smallest component-target norm; otherwise the original minibatch item is retained.

In the shared OU/Triple-well neural-dictionary pipeline, the first function is a pinned constant and the dictionary is orthonormalized by weighted QR. Lorenz-63, Lorenz-96, and SST instead use the explicitly standardized finite-coordinate constructions described in their own sections; they do not inherit this pinned-constant/QR statement. Generator eigenvalues are recovered as `lambda = log(mu) / T_eff`, where `T_eff` is the projection horizon (`tau` for OU and `3 tau` for Triple well). OU tables match each analytic target independently to its nearest BSO value; Triple well uses one-to-one Hungarian matching.

### Reproduction environment

The current artifacts and notebooks were validated in the following local environment. These versions describe the present reproduction environment; the compact OU, Triple-well, Lorenz-63, and Lorenz-96 checkpoints do not contain enough metadata to establish the exact environment in which their neural operators were trained.

| Component | Version / device |
|---|---|
| Python | 3.11.15 |
| PyTorch | 2.5.1+cu121 |
| neuraloperator | 1.0.2 |
| NumPy / SciPy | 1.26.4 / 1.14.1 |
| pandas / xarray | 2.2.3 / 2025.1.2 |
| Matplotlib | 3.10.1 |
| MNE | 1.8.0 |
| dask / Zarr | 2025.1.0 / 2.18.7 |
| numcodecs / requests | 0.15.1 / 2.32.5 |
| GPU | NVIDIA RTX 4000 SFF Ada Generation; CUDA 12.1 runtime |

Preparing the SST source dataset additionally requires `dask`, `numcodecs`,
`pandas`, `requests`, `xarray`, Zarr 2.x, and the applicable `fsspec` backend;
anonymous access to the `gs://` source requires `gcsfs`. These downloader
dependencies are not needed merely to display the committed reference figure.

### Experiment overview

| Experiment | Learned state | Training volume | Main comparison | Primary purpose |
|---|---:|---:|---|---|
| Ornstein-Uhlenbeck (OU) | Scalar functions on 128 grid points | 384 observable triples | Analytic Hermite spectrum | Controlled spectral validation |
| Triple well | Scalar functions on a `64 x 64` grid | 64 observable triples | Finite-difference generator spectrum | Metastable spectral validation |
| Lorenz-63 | 20-site sensor curves | 4,096 FNO triples; 12,000 BSO launch curves | Frozen FNO rollout | Held-out chaotic forecasting |
| Lorenz-96 | 20-site sensor curves | 4,096 FNO triples; 11,998 BSO two-step-compatible starts | Autoregressive FNO rollout | Held-out spatiotemporal forecasting |
| SST | Eleven `32 x 64` ocean-atmosphere fields; SST is the only target | 9,859 FNO triples; 9,860 BSO one-step launches | FNO, SFNO, damped persistence, raw persistence, climatology | Global daily SST forecasting |
| EEG | `64 x 160` one-second delay fields | 104 overlapping transitions from S029R02 | FNO, EDMD-DL, kernel EDMD, PCA-VAR | Real high-dimensional resting-state rollout |

## 2. Ornstein-Uhlenbeck experiment

The objective is to validate the learned finite-time operator and its BSO
projection against an analytic generator spectrum, first with the correct
Hermite function family supplied and then with a learned dictionary.
The process, discretization, and observable bank below define the repository
benchmark directly; no external dataset is used.

### 2.1 Scope and data

The OU process is

```math
\mathrm{d}X_t=-X_t\,\mathrm{d}t+\sqrt{2D}\,\mathrm{d}W_t,\qquad D=0.5,
\qquad \mathcal{L}f=-x f'(x)+D f''(x).
```

It is represented on the truncated domain `[-5, 5]` with 128 spatial points (`dx = 10/127`). The interior generator uses centered first and second differences; its two boundary rows are zero, so endpoint observable values remain fixed during the numerical solve. Crank--Nicolson uses 200 steps over `tau = 0.15`, hence internal step `7.5e-4`; the `2 tau` target uses 400 steps. The dataset contains 384 functions and their images at `tau` and `2 tau`, each with shape `(128,)`.

Of the 384 initial functions, 117 are randomly selected enveloped Hermite modes of degree 0--17 and 267 are random mixtures of Fourier components and Gaussian bumps. The envelope is `e(x) = exp[-0.1 (x/5)^2]`. Each random function contains two to four components; each component is Fourier or Gaussian with equal probability. Fourier components use integer frequency 1--4, amplitude in `[-1,1]`, and phase in `[0,2 pi]`; Gaussian components use center in `[-3,3]`, width in `[0.3,1]`, and amplitude in `[-1.5,1.5]`. Hermite samples are individually normalized in discrete `L2(rho)`, whereas random mixtures retain their sampled amplitude. Spatial inner products use normalized trapezoidal weights multiplied by `rho(x) proportional to exp[-x^2/(2D)]`. The seed-2026 Phase-A split contains 346 optimization samples and 38 validation samples. The analytic infinite-domain generator spectrum is

```math
\lambda_n=-n,\qquad n=0,1,2,\ldots
```

and is used as the exact theoretical spectrum of the continuous,
infinite-domain OU process. It is not the exact spectrum of the truncated
finite-difference generator.

### 2.2 Neural operator

| Setting | Value |
|---|---:|
| Backend | FNO |
| Input/output channels | 1 / 1 |
| Fourier modes | 24 |
| Width | 96 |
| Layers | 4 |
| Trainable parameters | 592,225 |
| Epochs | 3,000 |
| Learning rate | `6e-4` |
| Batch size | 24 |
| Weight decay | `7.5e-5` |
| Two-step semigroup weight | 0.25 |
| Constant-function weight | 0.15 |
| Linear-combination augmentation | None |
| Validation fraction | 0.10 |
| Weighted loss | Yes |
| Seed | 2026 |

### 2.3 Experiment A: fixed Hermite dictionary

The dictionary has 20 columns: the pinned constant plus `He_n(x/sqrt(D)) e(x)` for `n=1,...,19`, each normalized in the discrete `L2(rho)` geometry. These are enveloped Hermite columns on the finite domain rather than bare infinite-domain polynomials. The formal notebook uses the ResDMD dictionary-loss backend, while the finite-dimensional matrix is the EDMD least-squares projection. Phase B has one epoch, frozen operator and dictionary, model learning rate zero, data-loss weight zero, and invariance weight one. The final one-step projection uses `T_eff=tau=0.15`, ridge `1e-5`, a pinned `mu=1` constant, no matrix symmetrization, and no spectral-radius clipping.

The learned spectrum closely matches the analytic reference:

| Target mode | Target `lambda` | Nearest learned `lambda` | Absolute error |
|---:|---:|---:|---:|
| 0 | 0 | `0.000000` | `0.000e+0` |
| 1 | -1 | `-1.000337` | `3.375e-4` |
| 2 | -2 | `-1.999708` | `2.921e-4` |
| 3 | -3 | `-3.001732 + 0.001382i` | `2.216e-3` |
| 4 | -4 | `-4.002709` | `2.709e-3` |
| 5 | -5 | `-5.001388` | `1.388e-3` |
| 6 | -6 | `-6.001089` | `1.089e-3` |

For the six targets displayed in the notebook summary (`n=0,...,5`), mean absolute error is `0.001157` and standard deviation is `0.001028`. For the first six nonconstant modes (`n=1,...,6`), the corresponding values are `0.001338` and `0.000895`. The effective rank is `20/20`, and the reported Gram condition number is `1.000e+0` after orthonormalization.

The fixed-dictionary eigenfunction panel displays only `|x| <= 3`. The BSO and Hermite curves are separately normalized by their maximum absolute value on that interval, and their signs are aligned by the discrete `L2(rho)` inner product.

### 2.4 Experiment B: learned dictionary

The second OU experiment learns slow functions sequentially without using analytic eigenvalue targets.

| Setting | Value |
|---|---:|
| Dictionary size | 20 |
| Dictionary MLP | Tanh, hidden width 224, depth 4, 16 Fourier frequencies |
| Deflation stages | 5 attempted; 2 accepted columns |
| Epochs per deflation stage | 1,000 |
| Deflation learning rate | `3e-4` |
| Deflation / refinement weight decay | 0 / 0 |
| Eigen-residual weight | 5.0 |
| Consistency weight | 5.0 |
| Non-expansiveness penalty | 10.0 |
| Smoothness weight | 0.05 in deflation and joint refinement |
| Residual / consistency tolerances | `2e-3` / `5e-3` |
| Dictionary refinement epochs | 1,000 |
| Dictionary learning rate | `2e-4` |
| Basis steps per epoch | 2 |
| Invariance weight | 2.5 |
| Operator learning rate during refinement | 0 |
| Data-loss weight | 0 |
| Symmetric joint target / final matrix | Yes / yes |
| Final EDMD ridge | `1e-5` |
| Projection horizon / spectral clipping | `tau = 0.15` / none |
| Dictionary master seed | 2033; stage seeds 2034--2038 |

The network input is `xi=x/5` together with `sin(pi k xi)` and `cos(pi k xi)` for `k=1,...,16`, giving 33 coordinate features. Depth four means three hidden affine--Tanh layers followed by the output layer. Each deflation-stage epoch makes two Adam updates. The checkpoint contains a pinned constant, two accepted deflation columns, and 17 columns supplied by the joint network.

For a candidate `phi` that has been weighted-orthogonalized against the constant and previously accepted functions and normalized to `||phi||_w=1`, define `U=U_tau` and `mu_phi=<phi,U phi>_w`. A deflation stage minimizes

```math
\mathcal{J}(\phi)=-\mu_\phi
+5\frac{\|U\phi-\mu_\phi\phi\|_w^2}{\|U\phi\|_w^2+10^{-12}}
+5\frac{\|U\phi-L\phi\|_w^2}{\|U\phi\|_w^2+10^{-12}}
+10[\max(\mu_\phi-0.999,0)]^2
+0.05\frac{1}{p-1}\sum_{i=1}^{p-1}(\phi_{i+1}-\phi_i)^2 .
```

Here the empirical data operator is

```math
L=S_\tau^\top(SWS^\top+\epsilon I)^{-1}SW,\qquad
\epsilon=10^{-8}\mathrm{tr}(SWS^\top)/n,
```

where the rows of `S` and `S_tau` contain the training observables at time zero and `tau`. Acceptance additionally requires eigen-residual below `2e-3`, data-consistency residual below `5e-3`, and `|mu_phi| <= 1.005`. The non-expansiveness loss is therefore a one-sided penalty on `mu_phi`, not an absolute-value penalty.

The final float64 EDMD projection uses the frozen one-step FNO, ridge `1e-5`, a symmetric projected matrix, and no spectral clipping. Because the operator learning rate is zero in joint refinement, that operator is numerically the Phase-A checkpoint. No analytic target enters training; analytic eigenvalues and parity are used only for evaluation and figure postprocessing. For targets `n=0,...,5`, the learned-dictionary result has mean absolute error `0.1660` and standard deviation `0.2236`. The errors are `0.000000, 0.135246, 0.040708, 0.149316, 0.020591, 0.650115`.

**Result note.** The learned dictionary resolves the displayed slow directions. The sixth listed mode has the largest reported absolute error.

The learned-eigenfunction figure is a qualitative diagnostic. Each displayed mode is propagated four times and divided by its corresponding `mu^4`, followed by one weighted projection onto the degree-10 polynomial space, parity symmetrization, restriction to `|x| <= 3`, maximum normalization, and sign alignment to the analytic mode. No raw eigenfunction-error norm or repeated-seed uncertainty estimate is reported.

`nb_ou.ipynb` is the formal two-part execution path: it runs the fixed and learned dictionaries with `solver_resdmd.py`. The default `python -m experiments.ou.exp_ou` command runs only the fixed-dictionary experiment and defaults to `solver_edmd.py`; it is therefore not the complete two-part reproduction command.

## 3. Triple-well experiment

The objective is to test whether a learned BSO recovers metastable decay rates
and spatial partitions in a two-dimensional diffusion, using a separately
computed finer-grid finite-difference generator as the spectral reference.

### 3.1 Scope and data

The triple-well system is the reflected diffusion

```math
\mathrm{d}X_t=-\nabla V(X_t)\,\mathrm{d}t+1.09\,\mathrm{d}W_t,
\qquad D=1.09^2/2=0.59405,
```

with

```math
\begin{aligned}
V(x,y)={}&3e^{-x^2-(y-1/3)^2}-3e^{-x^2-(y-5/3)^2}
-5e^{-(x-1)^2-y^2}-5e^{-(x+1)^2-y^2}\\
&+0.2x^4+0.2(y-1/3)^4.
\end{aligned}
```

This exact potential is the in-repository benchmark definition. Its source
code and retained artifacts do not record an external provenance, so it
should not be attributed to another work without separate verification.

The domain is `[-2,2] x [-1,2]`, the training grid is `64 x 64` (`dx=4/63`, `dy=3/63`), and `tau=0.30`. The sparse backward generator uses upwind drift, centered diffusion, and reflecting boundaries. `scipy.sparse.linalg.expm_multiply` produces `uT` and `u2T` directly, without a time-stepping `dt`.

Each of the 64 random observables contains two to four Fourier/Gaussian components and is multiplied by `exp(-0.1 x^2 - 0.1 y^2)`. Each component is Fourier or Gaussian with equal probability. Fourier terms use `kx,ky in {1,2}`, independent phases in `[0,2 pi]`, and amplitude in `[-1,1]`. Gaussian terms use centers `x in [-1.2,1.2]`, `y in [-0.6,1.2]`, widths in `[0.4,1]`, and amplitude in `[-1.5,1.5]`. The sampled observables are not individually amplitude-normalized. Training and final EDMD use uniform spatial weights `1/4096`.

The reported spectral reference uses the same finite-difference stencil on a separate finer `128 x 128` grid and extracts the six generator eigenvalues with largest real part using ARPACK. The stored quantitative table in `spectrum_metrics.npz` uses `tol=1e-8`, `maxiter=8000`; the notebook display recomputes the same reference with `tol=1e-9`, `maxiter=20000`. They agree to the precision shown in the table. The seed-2026 Phase-A split contains 55 optimization samples and 9 validation samples.

### 3.2 Full configuration

| Component | Setting |
|---|---|
| Neural operator | Single-channel FNO, `16 x 16` modes, width 96, 5 layers, 6,767,473 parameters |
| Operator training | 500 epochs, learning rate `1e-3`, batch 16 |
| Regularization | weight decay `8e-5`, two-step semigroup weight 0.30 |
| Constant-function penalty / linear-combination augmentation | 0 / none |
| Validation fraction | 0.15 |
| Dictionary | 49 functions: pinned constant plus 48 learned functions |
| Dictionary coordinates | `xi_x=x/2`, `xi_y=y/2`, and `sin(pi k xi_d), cos(pi k xi_d)` for each axis `d` and `k=1,...,4`; 18 inputs total |
| Dictionary MLP | Tanh MLP, hidden width 256, depth 5, 48 outputs |
| Projection solver | EDMD invariance loss; final projected matrix symmetrized |
| EDMD ridge | `1e-8` |
| Joint refinement | 250 epochs |
| Joint learning rates | operator `1.5e-5`, dictionary `5e-5` |
| Basis steps per epoch | 10 |
| Invariance / data weights | 10.0 / 1.0 |
| Dictionary-training horizon | One step, `tau = 0.30` |
| Final spectral horizon | Three frozen Phase-A steps, `3 tau = 0.90` |
| Final spectral operator / radius clipping | Frozen Phase-A FNO / none |
| Seed | 2026 |

On checkpoint loading, the frozen Phase-A operator has all-sample one-step relative `L2 = 2.5801e-2`. Phase B uses Adam without a scheduler or weight decay for 250 outer epochs and ten dictionary updates per epoch. Each operator data-loss update draws at most 24 of the 64 samples; the EDMD target is refreshed each outer epoch without symmetrization. There is no Phase-B two-step, smoothness, or spectral-radius penalty. The 48 MLP outputs are joined with the constant and orthonormalized by uniform-weight QR. The projected dictionary has effective rank `49/49` and reported Gram condition number `1.000e+0`.

The quantitative spectrum composes the frozen Phase-A operator three times, forms the float64 EDMD matrix with ridge `1e-8`, pins the constant at `mu=1`, and then applies `(K+K^T)/2` before eigendecomposition. Because the projection geometry is uniform `L2(dx)` rather than the Gibbs invariant measure, this is a symmetry constraint motivated by reversibility, not an exact discrete `L2(rho)` self-adjoint projection. Generator values use `lambda=log(mu)/0.9`, with no spectral-radius clipping. The reported spectral table therefore uses the frozen Phase-A FNO, whereas the qualitative eigenfunction display uses the jointly refined one-step operator.

### 3.3 BSO spectral result

| Mode | FD reference `lambda` | Hungarian-matched BSO `lambda` | Absolute error |
|---:|---:|---:|---:|
| 0 | `0.0000` | `0.0000` | `3.059e-13` |
| 1 | `-0.0979` | `-0.0751` | `2.280e-2` |
| 2 | `-0.7851` | `-1.0619` | `2.768e-1` |
| 3 | `-3.9269` | `-3.9832` | `5.627e-2` |
| 4 | `-5.3446` | `-5.3374` | `7.205e-3` |
| 5 | `-6.7242` | `-6.6978` | `2.639e-2` |

**Assessment.** One-to-one Hungarian matching gives a six-mode mean absolute error of `0.0649` and maximum error of `0.2768`. The constant mode is exact, the slowest nontrivial decay rate is close to the finite-difference reference, and the three faster reported rates all have absolute error below `0.057`. The second nontrivial mode remains the least accurate of the six.

The constant BSO mode is first fixed by minimum `|lambda|`; the remaining five modes are assigned by Hungarian matching on complex generator-eigenvalue distance, so no BSO value is reused. The saved eigenfunction visualization is qualitative and selects the three BSO modes with largest real generator eigenvalue, rather than the FD-matching indices. Gibbs weights `rho proportional to exp[-(V-V_min)/D]` are used only for display centering, and the first panel is replaced by a spatial constant. Each nonconstant mode is propagated four times with the jointly refined one-step operator and divided by `exp(lambda tau)^4`; the second mode is constrained odd in `x` and anchored positive near `(1,0)`, while the third is constrained even and anchored positive near `(0,5/3)`. Both are smoothed with a Gaussian of `sigma=7` grid pixels; the second additionally uses display gamma `0.75`. The three color ranges are `[-1,1]`, `[-1.2,1.2]`, and `[-0.5,1.5]`. These operations do not change the spectral table. No repeated-seed interval or independent operator test set is reported.

## 4. Lorenz-63 forecasting

The objective is to compare an autonomous finite-dimensional BSO rollout with
the same frozen FNO used autoregressively on held-out chaotic sensor curves.
The governing system follows
[Lorenz (1963)](https://journals.ametsoc.org/view/journals/atsc/20/2/1520-0469_1963_020_0130_dnf_2_0_co_2.xml).

### 4.1 Dynamics and data protocol

The Lorenz-63 dynamics are

```math
\dot x=\sigma(y-x),\qquad
\dot y=x(\rho-z)-y,\qquad
\dot z=xy-\beta z,
```

with `sigma=10`, `rho=28`, and `beta=8/3`. The seed-2026 initial state is `[1,1,1] + 0.01 N(0,I)`. A fourth-order Runge--Kutta trajectory is generated with `dt=0.005` after 4,000 burn-in steps. The stored trajectory contains 300,000 integration steps sampled every `tau=0.1` (20 RK4 steps), giving 15,000 observable times.

The leading 80% (12,000 times) is used for training and dictionary fitting; the trailing 20% (3,000 times) is held out for forecasting. Twenty smooth sensor sites lie at `s_j=j/20`, with `theta_j=2 pi s_j`. Define `X=x/8`, `Y=y/9`, and `Z=(z-25)/8`. The four deterministic families are:

<details>
<summary>Exact Lorenz-63 observable recipe</summary>

- `ring_projection`: `X cos(theta)+Y sin(theta)+0.5 Z sin(2 theta)`;
- `quadratic_moment`: `XY cos(theta)+(Z^2-1) sin(theta)+0.4 XZ cos(2 theta+0.7)`;
- `nonlinear_sensor`: `tanh[X cos(theta)+Y sin(theta)+0.3Z]+0.25Z cos(2 theta+1)+0.15XY sin(theta+0.5)`;
- `radial_bump`: Gaussian distance to site-dependent centers `(12 cos(theta),12 sin(theta),25+6 sin(2 theta))`, scaled by `(9,10,9)`.

Twelve further families are generated sequentially from the seed-2026 random generator. Each draws three to five features without replacement from `X,Y,Z,XY,XZ,YZ,X^2-1,Z^2-1` and a random `tanh(aX+bY+cZ)` with `a,b,c independently distributed as N(0,0.8^2)`. Every selected feature receives a coefficient from `U[0.4,1]` and an independently sampled site envelope

```math
e(s)=a_0+\sum_{h=1}^3 a_h\cos(2\pi h s+\varphi_h),
\quad a_0\sim\mathcal{U}[-0.3,0.3],\quad
a_h\sim\mathcal{N}(0,(0.55/h)^2),\quad
\varphi_h\sim\mathcal{U}[0,2\pi].
```

</details>

Each of the 16 families contributes 256 uniformly spaced complete `(t,t+tau,t+2tau)` starts, for 4,096 FNO triples. Each family is normalized by one scalar mean and standard deviation computed jointly over all training times and sites; `1e-6` is added to the standard deviation. Forecast errors therefore refer to these dimensionless sensor curves.

### 4.2 Operator and BSO settings

| Component | Setting |
|---|---|
| Neural operator | `neuralop.models.FNO`, 1-D GridEmbedding, one channel, 10 modes, width 128, 5 layers, non-residual |
| Trainable parameters | 724,545 |
| Recorded operator training recipe | 700 epochs, learning rate `5e-4`, batch 64 |
| Regularization | weight decay `5e-5`, two-step semigroup weight 0.25 |
| Linear-combination augmentation probability | 0.75 |
| Terms per linear combination | 4 |
| Validation fraction | 0.10 |
| Operator loss geometry | Unweighted site relative `L2` (uniform over 20 sites) |
| BSO dictionary | Standardized 20-site curve plus degree-2 through degree-4 monomials of its leading five PCA coordinates, 140 features total |
| Dictionary fitting data | Every `nonlinear_sensor` training time, 12,000 curves total |
| Projection target | Frozen FNO one-step image of each training curve |
| Transition / decoder relative ridge | `1e-3` / `1e-10`, scaled by the mean Gram diagonal |
| Maximum transition spectral radius | 0.97 |
| Seed | 2026 |

The recorded Phase-A objective is one-step relative `L2` plus `0.25` times the two-step semigroup error. The seeded internal split contains 3,687 optimization and 409 validation triples; it is a random fitting-data split, not the chronological test. AdamW, cosine annealing, and gradient clipping at 5 follow Section 1.

The BSO uses the saved Phase-A FNO without dictionary fine-tuning. The 20 field values are centered and divided by their per-site training standard deviations. PCA is fitted to the centered field, the leading five PCA scores are divided by their training standard deviations, and all degree-2, degree-3, and degree-4 monomials are appended. Each standard deviation in this BSO transform receives `1e-12`; the resulting 140 features are again centered and scaled. No explicit constant or weighted QR is used. Separate ridge maps fit the feature transition and the decoder; in each case the ridge is `eta trace(X^T X)/d`. The decoder reconstructs the centered 20-site curve and then restores the field mean. Eigenvalues would be clipped radially only if they exceeded 0.97. The all-sample FNO one-step relative `L2` over 4,096 fitting triples is `0.05243`; this is an operator-fitting diagnostic.

The compact `fno.pt` verifies the architecture and weights but does not store optimizer state, completed epochs, validation history, or the full recorded recipe above. `experiments/lorenz63/exp_lorenz63.py` reconstructs and evaluates BSO from that frozen checkpoint; it does not retrain the FNO.

### 4.3 Held-out evaluation

Forecasts start from 64 uniformly spaced origins in the held-out tail, from test indices 25 through 2,888. Each forecast contains 110 transitions and 111 saved leads (`t=0` to `t=11`). At lead `k`, the reported metric is

```math
E_k=\frac{\sqrt{\sum_{o,j}w_j(\hat y_{o,j,k}-y_{o,j,k})^2}}
{\sqrt{\sum_{o,j}w_j y_{o,j,k}^2}},\qquad w_j=1/20,
```

so it is a ratio after pooling origins and sites, not an average of 64 per-origin ratios. Forecast windows overlap and are not independent samples.

| Horizon band | FNO mean relative `L2` | BSO mean relative `L2` | BSO reduction |
|---|---:|---:|---:|
| Short, `t <= 0.5` | 0.21264 | 0.23198 | -9.1% |
| Middle, `0.5 < t <= 2` | 0.95675 | 0.94898 | 0.8% |
| Long, `2 < t <= 11` | 1.19545 | 0.99582 | 16.7% |
| All steps | 1.11007 | 0.94820 | 14.6% |

`All steps` is the arithmetic mean over all 111 saved leads, including the
initial value at `t=0`.

The fitted transition has spectral radius `0.95780`, below the configured cap,
so no eigenvalue clipping is active. Its eigenvector condition number is
`1699.13`. The finite-dimensional rollout remains autonomous over
`t = 0--11`.

## 5. Lorenz-96 forecasting

The objective is to test whether the finite-dimensional projection improves
rollout stability and relative error for a spatially extended chaotic system.
The model follows
[Lorenz's ECMWF predictability paper](https://www.ecmwf.int/en/elibrary/75462-predictability-problem-partly-solved).

### 5.1 Dynamics and data protocol

The Lorenz-96 dynamics are

```math
\dot x_i=(x_{i+1}-x_{i-2})x_{i-1}-x_i+F,
\qquad i=1,\ldots,20,
```

with periodic indices and `F=8`. The seed-2026 initial state is `F 1 + 0.05 N(0,I)`. Integration uses fourth-order Runge--Kutta with `dt=0.01`, 4,000 burn-in steps, and sampling horizon `tau=0.02`. The full run contains 30,000 integration steps and therefore 15,000 sampled observable times.

As in Lorenz-63, the first 12,000 observable times are used for training and the last 3,000 are held out. With periodic shifts denoted by subscripts, the four deterministic families are:

<details>
<summary>Exact Lorenz-96 observable recipe</summary>

- `nonlinear_sensor`: `tanh[0.35x_i+0.08(x_{i-1}+x_{i+1})]+0.04x_i x_{i+1}`;
- `local_energy`: `x_i^2+0.25(x_{i-1}^2+x_{i+1}^2)`;
- `advective_tendency`: the Lorenz-96 right-hand side;
- `flux_gradient`: `x_i x_{i+1}-x_{i-1}x_{i+2}+0.125(x_{i+1}-x_{i-1})`.

Twelve random families are generated sequentially from the seed-2026 generator. Their periodic envelope is

```math
e(s)=b+(1-b)\exp[-d_{\mathrm{per}}(s,c)^2/(2w^2)]
+a\cos(2\pi h s+\varphi),
```

with `c~U[0,1]`, `w~U[0.08,0.25]`, `b~U[0.25,0.55]`, `a~U[-0.20,0.20]`, integer `h` uniform on 1--5, and `varphi~U[0,2pi]`. Let `g=(x_{i+1}-x_{i-1})/2`, `c_2=x_{i+1}-2x_i+x_{i-1}`, `a_d=(x_{i+1}-x_{i-2})x_{i-1}`, `p=x_i x_{i+1}-x_{i-1}x_{i+2}`, and `e_l=x_i^2+0.35(x_{i-1}^2+x_{i+1}^2)`. One of the following branches is drawn with probabilities `0.45/0.35/0.20`:

- **Stencil:** coefficients for `(x_i,x_{i-1},x_{i+1},x_{i-2},x_{i+2})` are independent normals with means `(1,0.45,0.45,-0.15,-0.15)` and standard deviations `(0.22,0.18,0.18,0.10,0.10)`. The core is augmented by `N(0,0.22^2)g`, `N(0,0.16^2)c_2`, and `N(0,0.12^2)tanh(r core)` with `r~U[0.20,0.80]`, then multiplied by `e(s)`.
- **Flux/energy:** `z U[0.45,1.05] a_d + U[0.12,0.34] e_l + N(0,0.18^2)p + N(0,0.16^2)g + N(0,0.14^2)x_i`, where `z` is equiprobably `-1` or `+1`, multiplied by `e(s)`.
- **Spectral:** two to four terms are drawn; each uses an integer spatial frequency 1--6, a phase in `[0,2pi]`, and independent cosine/sine coefficients from `N(0,0.22^2)`. The resulting spatial weight multiplies `x_i`, with additions `N(0,0.18^2)e(s)x_i^2`, `N(0,0.14^2)e(s)x_i x_{i+1}`, and `N(0,0.08^2)e(s)tanh(0.25x_i)^3`.

Every branch finally adds `0.04[e(s)-mean_s e(s)]g`.

</details>

Each of the 16 families contributes 256 complete `(t,t+tau,t+2tau)` starts to FNO fitting, producing 4,096 triples. The BSO projection uses all 11,998 two-step-compatible starts `0,...,11997` from the training portion of `nonlinear_sensor`. Per-family normalization uses one scalar mean and standard deviation over all training times and sites, with `1e-6` added to the standard deviation. All BSO dictionary statistics use the training window only.

### 5.2 Full configuration

| Component | Setting |
|---|---|
| Neural operator | `neuralop.models.FNO`, 1-D GridEmbedding, one channel, 10 modes, width 128, 5 layers, non-residual |
| Trainable parameters | 724,545 |
| Recorded operator training recipe | 700 epochs, learning rate `5e-4`, batch 64 |
| Regularization | weight decay `5e-5`, two-step semigroup weight 0.25 |
| Validation fraction | 0.10 |
| Linear-combination augmentation probability | 0.75 |
| Terms per linear combination | 4 |
| Operator loss geometry | Unweighted site relative `L2` (uniform over 20 sites) |
| BSO dictionary | Standardized 20-site field plus quadratic monomials of four PCA coordinates |
| Dictionary size | 30 |
| Projection target | Frozen FNO one-step image |
| Transition ridge | `3e-2 trace(X^T X)/30` |
| Decoder ridge | `1e-10 trace(X^T X)/30` |
| Spectral stabilization radius | 0.99 |
| Seed | 2026 |

The recorded Phase-A objective, optimizer, internal split (3,687 optimization and 409 validation triples), and checkpoint rule are the same as Lorenz-63. The compact `fno.pt` verifies architecture and weights but does not retain optimizer state, completed epochs, or validation history; `experiments/lorenz96/exp_lorenz96.py` loads rather than retrains it. Loading it recomputes one-step relative `L2=1.8113e-2` over all 4,096 fitting triples; this is an operator-fitting diagnostic, not the chronological forecast score.

Its construction mirrors Lorenz-63: 20 standardized field values plus ten standardized quadratic monomials of four scaled PCA scores, with `1e-12` added to every BSO-transform standard deviation, no explicit constant or QR, separate ridge decoder/transition maps, and field-mean restoration on decode. The transition has raw spectral radius `0.98124`, below the configured `0.99` cap, so no eigenvalue clipping is active. Its eigenvector condition number is `7.47`.

### 5.3 Held-out result

The stored result artifact evaluates 64 uniformly spaced held-out origins from test indices 0 through 2,698 for 300 transitions and 301 saved leads, corresponding to `t=0` through `t=6`; the notebook reads and displays those arrays. The pooled weighted relative `L2` definition is the same as in Section 4.3, with `w_j=1/20`; the overlapping forecast windows are not independent samples.

| Forecast range | Autoregressive FNO | BSO forecast | BSO reduction |
|---|---:|---:|---:|
| Short, `t <= 0.3` | 0.64954 | 0.54786 | 15.7% |
| Middle, `0.3 < t <= 1` | 1.31018 | 0.94774 | 27.7% |
| Long, `1 < t <= 6` | 1.52026 | 0.99628 | 34.5% |
| All saved leads | 1.44955 | 0.96680 | 33.3% |

`All saved leads` is the arithmetic mean over all 301 saved leads, including
the initial value at `t=0`. Beyond the chaotic predictability time, the
autonomous rollout approaches the training-distribution mean; long leads are
not phase-accurate trajectory claims.

## 6. Sea-surface-temperature forecasting

### 6.1 Scientific scope

The objective is to assess global SST forecast skill over days 1--90 and,
separately, numerical rollout stability over 540 days.

The SST experiment forecasts daily global SST on a `32 x 64` latitude-longitude grid. SST is always the only prediction target. Ten auxiliary fields enlarge the initial state but are advanced autonomously after launch; observed future auxiliary variables are never provided to any learned rollout.

For manuscript citation, the atmospheric archive is the
[WeatherBench 2 ERA5 dataset](https://arxiv.org/abs/2308.15560), and the
ocean-state variables are provided by the
[NOAA/NCEP Global Ocean Data Assimilation System](https://psl.noaa.gov/data/gridded/data.godas.html).

The eleven channels are:

1. SST;
2. zonal wind stress;
3. meridional wind stress;
4. sea-surface height;
5. potential temperature at 55 m;
6. potential temperature at 105 m;
7. zonal current at 5 m;
8. meridional current at 5 m;
9. mixed-layer depth;
10. salinity at 5 m;
11. total downward surface heat flux.

Daily SST and wind information come from the anonymous WeatherBench2 ERA5 store `gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-64x32_equiangular_conservative.zarr`. SST is the daily mean of the 6-hourly field. Wind stress is derived from 10 m winds as `rho_air C_D |U| (u,v)`, with air density `1.225 kg m^-3` and drag coefficient `1.3e-3`, and then averaged daily.

The other ocean-state fields come from the NOAA PSL NCEP GODAS public NCSS. Source variables are `sshg`, `pottmp@55m`, `pottmp@105m`, `ucur@5m`, `vcur@5m`, `dbss_obml`, `salt@5m`, and `thflx`, downloaded over `179 W--179 E, 74 S--65 N` after horizontal stride 6 and periodically linearly interpolated to the ERA5 target grid. Potential temperature is converted from kelvin to degrees Celsius and salinity from kg/kg to g/kg; SSH, currents, MLD, and heat flux use metres, m/s, metres, and W/m² respectively, with positive heat flux downward. A GODAS monthly mean is exposed only on the first day of the following month and then forward-filled, preventing within-month future leakage. The preprocessing identifiers are `sst-data-v6-1993-2022` and `godas-ocean-state-causal-month-lag-v3`.

### 6.2 Data volume and chronological protocol

The complete record spans `1993-01-01` through `2022-12-31`, or 10,957 daily times. Each raw state contains `11 x 32 x 64 = 22,528` values before masking and compression, for 246,839,296 float values overall (about 0.920 GiB as uncompressed `float32`). The prepared compressed Zarr occupies 209,051,228 bytes. The SST verification mask contains 1,008 ocean grid cells.

| Protocol | Fit period | Evaluation period | Complete `(t, t+1, t+2)` starts |
|---|---|---|---:|
| Validation protocol | 1993-01-01 to 2016-12-31 | 2017-01-01 to 2019-12-31 | 8,764 |
| Evaluation protocol | 1993-01-01 to 2019-12-31 | 2020-01-01 to 2022-12-31 | 9,859 |

The evaluation fitting window contains 9,861 daily states and the held-out evaluation window contains 1,096 days. The ocean verification mask requires ERA5 land fraction below 0.5 and at least one finite training-period SST value. A channel-specific point is valid when it is ocean and finite on at least half of the fitting days. Invalid or missing values are filled by that channel's fitting-window mean; every channel is then globally z-scored using one mean and standard deviation over its valid fitting times and grid points, with `1e-6` added to the standard deviation. Normalization, masks, leap-aligned 367-bin calendar climatology, EOF bases, residual decay, and all learned parameters use only the applicable fitting window. The data fingerprint is `13c88b85d7836d658475`; the evaluation training fingerprint is `326c250ccbccb19daf0c`.

SST receives 0.5 of the weighted state loss; each of the ten auxiliary variables receives 0.05. Within each channel, cosine-latitude weights are normalized over its valid mask; the concatenated channel weights are then normalized to one.

The reported recipe was fixed using the 1993--2016 fitting window and the 2017--2019 chronological validation window. Neural operators and BSO were then refitted on 1993--2019 and evaluated once on 2020--2022.

### 6.3 FNO and SFNO settings

FNO and SFNO use the same nominal width, depth, spectral-mode setting, epoch budget, and optimization settings. The FNO is `neuralop.models.FNO` with two-dimensional `GridEmbeddingND`, GELU, lifting/projection ratio 2, channel-MLP expansion 0.5, linear spectral-block skips, soft-gating channel-MLP skips, and no residual wrapper, normalization, domain padding, or dropout. The local SFNO replaces the 2-D Fourier convolution by longitude rFFT plus associated-Legendre latitude bases: the latter are orthonormalized by `sin(theta)`-weighted QR, followed by complex channel mixing, a parallel `1 x 1` convolution, GELU, and an outer residual update. Their parameter counts are 5,867,819 for FNO and 10,675,979 for SFNO.

| Setting | FNO | SFNO |
|---|---:|---:|
| Input/output channels | 11 / 11 | 11 / 11 |
| Spectral modes | `(12, 24)` | `(12, 24)` |
| Width | 96 | 96 |
| Layers | 4 | 4 |
| Epochs | 180 | 180 |
| Learning rate | `5e-4` | `5e-4` |
| Batch size | 12 | 12 |
| Weight decay | `1e-5` | `1e-5` |
| Validation fraction | 0.10 | 0.10 |
| Two-step semigroup weight | 0.25 | 0.25 |
| Linear-combination augmentation probability | 0.5 | 0.5 |
| Terms per linear combination | 4 | 4 |
| Weighted state loss | Yes | Yes |
| Maximum average CUDA duty cycle | 0.8 | 0.8 |
| Seed | 2026 | 2026 |

The random fitting-window split contains 8,874 optimization pairs and 985 internal-validation pairs. Minibatch mixing follows the common rule in Section 1, with probability 0.5 and four component triples. Both checkpoints completed all 180 epochs without early stopping. Their minimum internal-validation one-step weighted relative `L2` values are `0.1209218516` for FNO and `0.1211206936` for SFNO; the validation score used for checkpointing does not include the two-step term. The internal split is not the chronological 2020--2022 evaluation. The repository SFNO is the local implementation described above, not `neuralop.models.SFNO`.

After every FNO or SFNO application, grid points outside each channel's valid mask are reset to normalized zero before the state is used autoregressively. This constrains structurally unavailable land or high-latitude values to their fitting mean.

### 6.4 BSO construction

The raw one-day FNO and its input normalization are fixed. BSO is fitted in a training-only structured anomaly representation. SST is centered by the empirical leap-aligned calendar-day climatology. For each auxiliary channel, ordinary least squares fits the 13 columns `[1, sin(k theta), cos(k theta)]`, `k=1,...,6`, where `theta` is the leap-aligned annual phase. The resulting channel anomalies are divided by their unweighted fitting-period standard deviation over valid points, floored at `1e-6`.

Each scaled anomaly field is then projected with channel-internal cosine-latitude weights onto 64 EOFs for SST and four EOFs for each auxiliary field (104 channelwise coefficients in total). The coefficient blocks are multiplied by the square root of their state-loss weight, namely `sqrt(0.5)` for SST and `sqrt(0.05)` for every auxiliary channel, before a joint PCA compresses them to rank 96.

The frozen FNO advances 9,860 one-step fitting-window launch states by one day, using an encoding batch size of 24. In the joint coordinates, the fitted one-step target is

```math
q_{\mathrm{target}}=0.6\,q_{\mathrm{observed}}+0.4\,q_{\mathrm{FNO}}.
```

The BSO uses this blended target. A linear ridge map without an intercept is fitted with equal weight for every consecutive day and regularization `0.01 trace(G)/rank`; the resulting numerical ridge coefficient is `0.6953112051`. The raw transition is then materialized through real-Schur block radial clipping. There are no delays, concatenated histories, recurrent states, higher-order autoregressions, or memory kernels.

| BSO setting | Value |
|---|---:|
| SST EOFs | 64 |
| EOFs per auxiliary channel | 4 |
| Channelwise coefficient dimension | 104 |
| Auxiliary seasonal harmonics | 6 |
| Joint snapshot rank | 96 |
| Observed-target blend weight | 0.6 |
| Ridge factor | `0.01` |
| Numerical ridge coefficient | `0.6953112051` |
| Affine intercept | No |
| Fitting-time weights | Equal |
| FNO encoding batch / maximum CUDA duty cycle | `24 / 0.8` |
| Raw spectral radius | `0.997761` |
| Rollout radius cap | 0.9975 |
| Stabilized eigenvalues | 1 |
| Stabilized spectral radius | 0.9975 |
| Raw / stabilized transition condition number | `1.70051494 / 1.70051415` |
| Relative real-Schur reconstruction error | `6.826e-15` |
| Reduced one-step target RMSE | `0.00932708` |
| Weighted unresolved structured-SST residual RMSE | `0.409011` (dimensionless anomaly units) |
| Training-only unresolved-SST residual decay | `0.905202` per day |
| Residual-decay absolute cap | 0.999 |

The component `e_0` of the launch SST field outside the joint decoder is carried as `gamma^l e_0` at lead `l`, where the scalar `gamma=0.9052023473` is estimated from consecutive fitting-period residuals and is not clipped by its 0.999 cap. Thus the decoded forecast is the structured reconstruction of `q_l` plus `gamma^l e_0`. This is an AR(1) decoder-residual closure initialized from the launch field, not an additional state-history input. The BSO trajectory otherwise evolves autonomously from one multivariate launch state. The structured-transform fingerprint is `b030692674e49b72dad3`.

### 6.5 Two-part evaluation protocol

The approved SST evaluation separates forecast skill from numerical rollout stability:

1. **Formal forecast skill (days 1--90).** Results are reported in the nonoverlapping bands 1--10, 11--30, and 31--90 days. The required metrics are physical-unit area-weighted RMSE, physical-unit area-weighted MAE, and centered anomaly ACC relative to training-only target-date climatology. Statistical baselines are interpreted by lead: raw persistence is primarily a short-lead reference, while damped persistence and calendar-day climatology remain the principal references as lead increases. All learned and statistical methods are scored on the same stored leads within days 1--90.
2. **Rollout-stability stress test (through day 540).** Only BSO, FNO, and SFNO are compared. Every rollout and validity check is computed daily from day 1 through day 540. Raw model outputs are evaluated without replacement or clipping. Every finite value is retained even after its absolute normalized state exceeds the diagnostic threshold of 1,000; only `NaN` or positive/negative infinity invalidates an origin. The figure reports physical-unit SST RMSE pooled over all 48 origins while every origin remains finite. A curve terminates only when that pooled value becomes undefined, so the finite part of an explosion is displayed directly rather than replaced by a survivor-only score. It is a numerical-stability test, not a claim of seasonal-to-annual forecast skill; climatology, raw persistence, and damped persistence are therefore not ranked in this 540-day test.

The reference run was produced in the local runtime bundle `results/sst/evaluation/sst-forecast-90d-stability-540d-seed2026/`. That bundle and `results/sst/latest_result.json` are ignored by Git; the committed figures and saved notebook summaries preserve the reported presentation. The model was fitted on 1993--2019 and evaluated on 2020--2022. Its forecast-skill and stability results are reported in Sections 6.8 and 6.9.

### 6.6 Baselines

Within the 1--90-day forecast-skill protocol, all methods use the same launch dates and lead grid.

- **FNO:** direct autoregressive rollout of the trained Fourier neural operator.
- **SFNO:** direct autoregressive rollout of the matched spherical Fourier neural operator.
- **BSO:** rank-96 structured BSO with the frozen-FNO/observed one-step blended target defined in Section 6.4.
- **Raw persistence:** repeats the launch-day physical SST at every lead.
- **Calendar-day climatology:** training-only leap-aligned month/day mean SST for the target date.
- **Damped persistence:** predicts target-date climatology plus a decayed launch anomaly,

```math
\mathrm{SST}_{\mathrm{DP}}(t+\ell)
=C(t+\ell)+\gamma^\ell[\mathrm{SST}(t)-C(t)].
```

The damped-persistence reference uses a training-only decay factor of `gamma = 0.985`. It is selected using the final 400 fitting days and 64 uniformly spaced fitting-period origins. The grid `0.90, 0.905, ..., 1.00` minimizes the sum of area-weighted RMSE at leads 7, 14, and 30 days; no held-out evaluation target is used.

These statistical baselines are excluded from the 540-day rollout-stability ranking.

### 6.7 Evaluation metrics

Formal forecast skill uses 96 uniformly spaced offsets 0--1005 from the evaluation start, corresponding to origins from `2020-01-01` through `2022-10-02`; the final 90-day target is `2022-12-31`. Every daily lead from 1--90 is scored. The separate stability test uses 48 uniformly spaced offsets 0--554, corresponding to origins from `2020-01-01` through `2021-07-08`; its last day-540 target is `2022-12-30`. Finite/nonfinite status is checked on all 540 daily states. Complete-cohort RMSE and internal diagnostics are stored on the 96-point reporting grid containing every day from 1--30, every five days from 35--180, and every ten days from 190--540.

- **RMSE:** latitude- and ocean-area-weighted physical SST error in kelvin; lower is better.
- **MAE:** latitude- and ocean-area-weighted physical absolute SST error in kelvin; lower is better.
- **Centered anomaly ACC:** centered anomaly correlation relative to training-only target-date climatology; higher is better. For each origin, the cosine/ocean-weighted spatial mean is first removed from both forecast and observed anomalies. The numerator and energies are then pooled over all origins and valid grid points before forming one correlation, rather than averaging per-origin ACC values.
- **Bias:** latitude- and ocean-area-weighted mean forecast error in kelvin; values near zero are better.
- **Anomaly RMS-amplitude ratio:** forecast anomaly RMS amplitude divided by observed anomaly RMS amplitude; values near one are better. Unlike ACC, this implementation does not remove the per-case weighted spatial anomaly mean.
- **Numerical validity:** an origin remains valid while the evaluated state is finite. The absolute normalized-value level 1,000 is recorded only as a large-state diagnostic and does not terminate the 540-day rollout. FNO and SFNO check their complete 11-channel normalized state. BSO checks decoded normalized SST; the latent coordinate and auxiliary reconstruction are not independently tested.
- **Complete-cohort RMSE:** physical SST RMSE is pooled over all 48 origins and all weighted ocean grid points. It is reported only while every origin remains finite; a survivor-only average is not substituted after `NaN` or infinity appears.

### 6.8 1--90-day forecast-skill result

The stored `forecast_metrics.npz` evaluates all 90 daily leads. The following band averages report `RMSE / MAE / centered anomaly ACC`:

| Lead band | BSO | FNO | SFNO | Damped persistence | Raw persistence | Climatology |
|---:|---:|---:|---:|---:|---:|---:|
| 1--10 days | `0.2455 / 0.1750 / 0.905` | `0.2605 / 0.1892 / 0.896` | `0.2681 / 0.1989 / 0.885` | `0.2391 / 0.1658 / 0.908` | `0.2915 / 0.2033 / 0.877` | `0.6170 / 0.4569 / N/A` |
| 11--30 days | `0.4053 / 0.2921 / 0.739` | `0.5967 / 0.4351 / 0.615` | `0.6250 / 0.4763 / 0.553` | `0.3862 / 0.2707 / 0.762` | `0.7247 / 0.5181 / 0.558` | `0.6174 / 0.4567 / N/A` |
| 31--90 days | `0.4933 / 0.3601 / 0.598` | `1.7531 / 1.0756 / 0.231` | `Incomplete` | `0.5053 / 0.3651 / 0.583` | `1.7464 / 1.2898 / 0.210` | `0.6209 / 0.4577 / N/A` |

Climatology ACC is `N/A`, not zero: subtracting the same target-date climatology leaves a zero forecast anomaly, so its anomaly correlation is undefined. SFNO has complete-cohort scores at only 21 of the 60 daily leads in the 31--90-day band. Its finite-lead diagnostic nanmean is `5.0081 / 2.2559 / 0.161`, but this is not a complete-band average and is not used for ranking.

The BSO band-mean bias and anomaly RMS-amplitude ratio are `-0.0253 K / 0.858`, `-0.0662 K / 0.691`, and `-0.1060 K / 0.526` for days 1--10, 11--30, and 31--90, respectively. They diagnose a progressively cool and under-dispersive anomaly field; they are not additional selection objectives.

### 6.9 540-day rollout-stability result

The stress-test figure contains two logarithmic RMSE panels with identical learned-rollout curves over 48 origins. The left panel shows the complete finite dynamic range; the right panel restricts the visible range to `10^-1--10^3 K` so the bounded BSO curve, growing FNO curve, and early SFNO explosion can be distinguished:

| Method | Last plotted lead | Complete-cohort RMSE | Numerical result |
|---|---:|---:|---|
| BSO | Day 540 | `0.6551 K` | Finite through the full stress-test horizon |
| FNO | Day 540 | `816.327 K` | Finite through day 540 |
| SFNO | Day 170 | `1.2046e35 K` | First `NaN` or infinity at day 171 |

The large-state diagnostic is first crossed by SFNO at day 53 and by FNO at day 475, but neither event truncates the new curves. The stress test includes no climatology, raw-persistence, or damped-persistence scores. The BSO endpoint `0.6551 K` is a bounded-rollout display, not evidence of useful 540-day SST forecast skill.

### 6.10 SST artifact and execution reproducibility

`nb_sst.ipynb` is the presentation entry point. When the ignored local
evaluation bundle is available, it reads `forecast_metrics.npz` and
`stability_metrics.npz` and regenerates both SST figures from those arrays.
From a clean checkout, it instead displays the committed reference figures.
The detailed numerical summaries remain documented in Sections 6.8--6.9 and
require the ignored metric archives to recompute; the fallback must not be
described as a fresh model fit or metric recomputation.

The implementation is under `experiments/sst/`: `download_sst_data.py`
prepares the dataset, `exp_sst.py` exposes the command-line interface,
`sst_model.py` defines the structured BSO, and `sst_evaluation.py` implements
the locked protocol. The prepared Zarr, selection lock, checkpoints, frozen
recipe, and metric archives are intentionally ignored by Git. The values in
Sections 6.8–6.9 and the committed figures are the fixed reference result;
full retraining is hardware-intensive and is not bitwise guaranteed across
library builds or GPU architectures.

## 7. EEG resting-state rollout

This section specifies the fixed EEG reference experiment implemented by
`nb_eeg.ipynb` and
`experiments/eeg/exp_eeg_resting_reproduction.py`.

### 7.1 Scope

The experiment asks whether an FNO-based BSO projection improves an
autonomous multistep forecast of a high-dimensional EEG delay field. It
compares BSO with direct FNO, EDMD with dictionary learning (EDMD-DL),
Gaussian-kernel EDMD, and PCA-VAR.

The method is called **BSO** in the manuscript and figure. The internal
reproduction key `FNO-BSO` records that this instance uses an FNO backbone; it
does not denote a separate method.

R02 is one continuous eyes-closed baseline recording with no task cue or recorded
control input. It is therefore treated as approximately autonomous. This is
an approximation: resting EEG remains nonstationary and is affected by
unobserved physiological inputs.

### 7.2 Data and preprocessing

- Source: [EEG Motor Movement/Imagery Dataset, version 1.0.0](https://physionet.org/content/eegmmidb/1.0.0/).
- Suggested citations: the dataset DOI
  [10.13026/C28G6P](https://doi.org/10.13026/C28G6P) and the
  [BCI2000 acquisition-system paper](https://pubmed.ncbi.nlm.nih.gov/15188875/).
- Record: subject S029, run R02.
- EDF SHA256: `7eff9783f20f4d3521ae877f26d7b2c87c609a920e16d12e5479072c2cb65671`.
- Signal: 64 canonical EEG channels at 160 Hz; the first 60 seconds are used without resampling.
- Preprocessing: common-average reference followed by a causal fourth-order Butterworth 8--30 Hz bandpass on the continuous record.
- Normalization: per-channel mean and standard deviation fitted only on 1--15 seconds, after the one-second causal-filter warmup.
- State: a $64\times160$ delay field containing one second of EEG.
- Map interval: 16 samples, or $\tau=0.1$ seconds.

The training envelope is 0--15 seconds. Transition origins run from 2.000 to
14.875 seconds at a 20-sample ($0.125$-second) stride, giving 104 pairs.
Their complete history/target support is 1.000--14.975 seconds. Because the
one-second histories overlap strongly, the 104 pairs are not 104 independent
trials.

The displayed forecast is one continuous boundary rollout from 15 to 19
seconds: 40 applications of the $0.1$-second map. It starts from the true
history ending at 15 seconds and receives no later observation.

### 7.3 Metric

At rollout step $k$, the reported cumulative relative $L^2$ error is

```math
E_k =
\frac{
  \left(\sum_{j=1}^{k}\|\widehat{y}_j-y_j\|_2^2\right)^{1/2}
}{
  \left(\sum_{j=1}^{k}\|y_j\|_2^2\right)^{1/2}
}.
```

Before evaluation, predictions and targets are returned from normalized
coordinates to postprocessed EEG amplitude units. The norm flattens all 64
channels and the first $16k$ future samples. Consequently, $E_{40}$
summarizes the entire 15--19 second forecast; it is not the error at only the
last sample. With identical entries, this quantity is also cumulative RMSE
divided by target RMS.

For comparator $m$, the reported BSO reduction is

```math
100\,\frac{E_m-E_{\mathrm{BSO}}}{E_m}.
```

There is one rollout, so the curve has no variance or confidence band.

### 7.4 Methods

The one-dimensional FNO treats time as the regular Fourier axis and the 64
electrodes as channels. Electrodes are not placed on an artificial
two-dimensional grid.

BSO first learns the full delay-field map

```math
x_{t+\tau}\approx F_\tau(x_t).
```

After $F_\tau$ is frozen, a training-only PCA map produces a
four-dimensional state $z$. The final dictionary is

```math
\Psi(z)=[1,z_1,\ldots,z_4,h_1(z),\ldots,h_{20}(z)],
\qquad
K=(G+\alpha I)^{-1}A .
```

The constant observable is exact and non-trainable as a dictionary column,
but the ridge fit does not hard-constrain its transition column or eigenvalue
to one. The learned functions are trained against the frozen FNO image, and
the resulting Koopman matrix is
$25\times25$. Rollout reads the four fixed coordinate entries of
$\Psi K^n$ and applies that method's fixed training-only PCA inverse.

All 104 transition pairs have the same weight, `1/104`. Phase A minimizes the
mean squared relative field error

```math
\mathcal{L}_{\mathrm{op}}
=
\sum_i w_i
\frac{\|F_\tau(x_i)-y_i\|_F^2}
{\max(\|y_i\|_F^2,10^{-8})} .
```

This is a squared relative-$L^2$ objective, despite the shorter diagnostic
label `weighted_relative_field_mse`. Phase B uses a Tanh MLP
`4 -> 16 -> 16 -> 20` for the learned functions. At the start of every
dictionary epoch, $K$ is refitted on all 104 pairs by weighted ridge EDMD.
With that $K$ fixed for the epoch, the MLP minimizes

```math
\mathcal{L}_{\mathrm{inv}}
=
\sum_i w_i\,\frac{1}{25}
\|\Psi(z_i)K-\Psi(z_i^+)\|_2^2 ,
```

where $z_i^+$ is the PCA coordinate of the frozen-FNO image.

EDMD-DL follows the alternating dictionary-learning formulation of
[Li et al. (2017)](https://arxiv.org/abs/1707.00225): it alternates a
ridge least-squares update of $K$ with gradient updates of learned
observables using observed transition pairs. The implemented comparator is
EDMD-DL on a 20-dimensional PCA delay state, not on the full
10,240-dimensional delay vector. Its learned functions use a Tanh MLP
`20 -> 16 -> 16 -> 4`; the once-per-epoch ridge update and invariance-MSE
dictionary update have the same form as above, but $z_i^+$ comes from the
observed next history rather than the frozen FNO.

Kernel EDMD uses

```math
k_\sigma(z,c)=\exp\!\left(-\frac{\|z-c\|_2^2}{2\sigma^2}\right),
```

with every training state as a center and
$\sigma=0.2$ times the median nonzero pairwise distance between training
states.

The finite-state dimensions are:

| Method | Fixed state | Dictionary / transition dimension |
|---|---:|---:|
| BSO | PCA $q=4$ | $25=1+4+20$ |
| EDMD-DL | PCA $q=20$ | $25=1+20+4$ |
| Kernel EDMD | PCA $q=8$ | $113=1+8+104$ |
| PCA-VAR | PCA $q=10$ | $11=1+10$ |
| Direct FNO | Full $64\times160$ field | No finite $K$ in prediction |

The neural parameter counts fitted for the reported methods are:

| Method | Trainable neural parameters fitted for the reported method |
|---|---:|
| BSO | FNO: 183,520 tensor entries (335,584 real scalars); dictionary: 692 |
| Direct FNO | 109,792 tensor entries (188,128 real scalars) |
| EDMD-DL | Dictionary: 676 |
| Kernel EDMD / PCA-VAR | None |

Complex FNO spectral weights account for the difference between tensor-entry
and real-scalar counts. The frozen BSO FNO generates Phase-B fitting targets
but is not called during BSO rollout; inference uses the dictionary at launch,
the finite matrix, and inverse PCA. The reported BSO Gram matrix has rank
`21/25`, while the EDMD-DL Gram matrix has rank `25/25`; both propagated
Koopman matrices remain `25 x 25`.

Each finite-dimensional method fits its own PCA map using training histories
only. Each reconstructs through the fixed inverse of that map; none adds a
trainable Koopman-state decoder or autoencoder. The FNO itself retains its
standard learned lifting and output projection.

No method uses labels, controls, a control matrix $B$, external regressors,
reconstruction loss, multistep loss, whitening gates, or spectral
clipping/damping.

### 7.5 Fixed configuration

All neural methods use seed 2026 and batch size 32. FNO training uses AdamW,
a cosine learning-rate schedule, weighted relative field MSE, and gradient
clip 5.

| Component | Active configuration |
|---|---|
| BSO field operator | Temporal FNO: width 48, 64 Fourier modes, 2 layers, 100 epochs, learning rate `7e-4`, weight decay `2e-5`. |
| BSO projection | PCA $q=4$; dictionary size 25 with hidden widths `[16,16]`; 300 Adam epochs; learning rate `7e-4`; weight decay 0; transition ridge `3e-6`. |
| Direct FNO | Temporal FNO: width 48, 32 Fourier modes, 2 layers, 160 epochs, learning rate `7e-4`, weight decay `2e-5`; prediction is the raw autoregressive FNO output. |
| EDMD-DL | PCA $q=20$; dictionary size 25 with hidden widths `[16,16]`; 120 Adam epochs; learning rate `1e-3`; weight decay 0; transition ridge `0.1`. |
| Kernel EDMD | PCA $q=8$; Gaussian bandwidth `0.2` times the training-state median distance; ridge `1e-10`; all 104 training states are kernel landmarks. |
| PCA-VAR | Affine VAR(1) on PCA $q=10$; ridge `1e-6`. |

The shared FNO training wrapper also creates one-epoch interim projections
($q=8$, size 10 for the BSO field-operator fit; $q=2$, size 4 for
direct FNO). They are discarded before the reported BSO projection and
do not enter direct-FNO prediction. The exact method hyperparameters are
returned by `locked_hyperparameters()` in the reproduction module; no
external configuration file is required.

BSO and direct FNO intentionally use different Fourier modes and training
epochs. The comparison therefore ranks the fixed complete methods; it is not
a matched-backbone ablation that isolates only the projection.

### 7.6 Reference result

| Rank | Method | Four-second cumulative relative $L^2$ | BSO error reduction vs method |
|---:|---|---:|---:|
| 1 | BSO | 0.749440902 | -- |
| 2 | FNO | 0.873672634 | 14.219% |
| 3 | EDMD-DL | 0.940987390 | 20.356% |
| 4 | Kernel EDMD | 0.944762509 | 20.674% |
| 5 | PCA-VAR | 0.963201524 | 22.193% |

At step 40, the reduction relative to the best non-BSO comparator is

```math
100\,
\frac{0.873672634-0.749440902}{0.873672634}
=14.219\%.
```

The 15--19 second result is the fixed reported endpoint, not an independently
sealed test.

The reproduction JSON uses the internal key `FNO-BSO`; the figure and
manuscript use `BSO`.

### 7.7 Reproduction

From a clean checkout, install the environment described in `README.md` and
run all cells in `nb_eeg.ipynb`. If the EDF is absent, the notebook downloads
only S029R02 through MNE, verifies its SHA256, fits the fixed setup, prints the
endpoint table, and regenerates:

- `figures/eeg/eeg_rollout_comparison.pdf`
- `figures/eeg/eeg_rollout_comparison.png`
- `results/eeg/resting-S029-R02-rollout/reproduction.json`

The equivalent command is:

```text
python -m experiments.eeg.exp_eeg_resting_reproduction --download
```

Use `--device cpu` to force CPU execution. Runtime data, checkpoints, and
numerical results are ignored by Git; the reference figure is versioned.

### 7.8 Interpretation limits

- This is one subject, one resting run, and one rollout. It does not establish population-level EEG performance or cross-subject generalization.
- The 104 training pairs overlap strongly, so their count is not an effective independent sample size.
- Resting EEG is only approximately autonomous and stationary.
- The displayed 14.219% advantage is a single-window result.
- The record, forecast window, and fixed configuration were chosen during
  method development. The reported advantage is therefore descriptive and
  is not an unbiased estimate of performance on a newly sealed EEG window.
- No uncertainty interval is available from a single rollout.

## 8. Publication figures and copy-ready captions

The repository contains eleven scientific figures. Each PDF/PNG pair is one
figure; use the vector PDF for manuscript insertion. Every plotting cell
prints the corresponding caption below the figure.

| ID | Experiment and content | Files | Notebook order |
|---|---|---|---|
| OU-1 | Fixed-dictionary generator eigenvalues | [PDF](figures/ou/ou_exp1_eigenvalues.pdf) / [PNG](figures/ou/ou_exp1_eigenvalues.png) | `nb_ou.ipynb`, figure 1 |
| OU-2 | Fixed-dictionary eigenfunctions | [PDF](figures/ou/ou_exp1_eigenfunctions.pdf) / [PNG](figures/ou/ou_exp1_eigenfunctions.png) | `nb_ou.ipynb`, figure 2 |
| OU-3 | Learned-dictionary generator eigenvalues | [PDF](figures/ou/ou_exp2_eigenvalues.pdf) / [PNG](figures/ou/ou_exp2_eigenvalues.png) | `nb_ou.ipynb`, figure 3 |
| OU-4 | Learned-dictionary eigenfunctions | [PDF](figures/ou/ou_exp2_eigenfunctions.pdf) / [PNG](figures/ou/ou_exp2_eigenfunctions.png) | `nb_ou.ipynb`, figure 4 |
| TW-1 | Triple-well finite-time spectrum | [PDF](figures/triple_well/tw_eigenvalues.pdf) / [PNG](figures/triple_well/tw_eigenvalues.png) | `nb_triple_well.ipynb`, figure 1 |
| TW-2 | Triple-well metastable eigenfunctions | [PDF](figures/triple_well/tw_eigenfunctions.pdf) / [PNG](figures/triple_well/tw_eigenfunctions.png) | `nb_triple_well.ipynb`, figure 2 |
| L63-1 | Lorenz-63 forecast error | [PDF](figures/lorenz63/l63_prediction.pdf) / [PNG](figures/lorenz63/l63_prediction.png) | `nb_lorenz63.ipynb`, only figure |
| L96-1 | Lorenz-96 forecast error | [PDF](figures/lorenz96/l96_prediction.pdf) / [PNG](figures/lorenz96/l96_prediction.png) | `nb_lorenz96.ipynb`, only figure |
| SST-1 | 1--90-day SST forecast skill | [PDF](figures/sst/sst_forecast_skill.pdf) / [PNG](figures/sst/sst_forecast_skill.png) | `nb_sst.ipynb`, figure 1 |
| SST-2 | 540-day SST rollout stability | [PDF](figures/sst/sst_rollout_stability.pdf) / [PNG](figures/sst/sst_rollout_stability.png) | `nb_sst.ipynb`, figure 2 |
| EEG-1 | Four-second resting-state rollout | [PDF](figures/eeg/eeg_rollout_comparison.pdf) / [PNG](figures/eeg/eeg_rollout_comparison.png) | `nb_eeg.ipynb`, only figure |

### 8.1 Copy-ready captions

#### OU-1: fixed-dictionary generator eigenvalues

> OU generator eigenvalues from the learned one-step solution operator projected onto a fixed enveloped-Hermite dictionary. Red circles denote the analytic spectrum $\lambda_n=-n$, and blue crosses denote the nearest BSO estimates; the first six nonconstant modes have mean absolute error 0.001338.

#### OU-2: fixed-dictionary eigenfunctions

> Leading OU BSO eigenfunctions from the fixed enveloped-Hermite dictionary compared with analytic Hermite modes on $|x|\leq 3$. Curves are independently normalized and sign-aligned, so the panels compare spatial shape rather than amplitude.

#### OU-3: learned-dictionary generator eigenvalues

> OU generator eigenvalues obtained with the learned 20-function dictionary. Red circles show $\lambda_n=-n$, and green crosses show the independently nearest BSO estimate for each target; a BSO value may therefore be reused. The first six targets, including the stationary mode, have mean absolute error 0.1660.

#### OU-4: learned-dictionary eigenfunctions

> Postprocessed OU BSO eigenfunctions from the learned dictionary compared with analytic Hermite modes on $|x|\leq 3$. The panels assess parity, nodal structure, and large-scale shape after polynomial projection, parity symmetrization, normalization, and sign alignment.

#### TW-1: finite-time eigenvalues

> Leading triple-well BSO multipliers at $T_{\mathrm{eff}}=3\tau=0.9$ compared with finer-grid finite-difference references mapped to the same horizon. Blue crosses show the 12 largest-$|\mu|$ BSO values and red circles the six references; one-to-one generator matching gives mean absolute error 0.0649.

#### TW-2: metastable eigenfunctions

> Leading triple-well BSO eigenfunctions on the triple-well state-space domain. The stationary mode is followed by an odd left--right partition and an even upper-versus-lower partition of the three wells; panels use independent display scales and visualization postprocessing, including symmetry constraints and smoothing.

#### L63-1: Lorenz-63 relative forecast error

> Pooled relative $L^2$ forecast error over 64 overlapping origins from the held-out Lorenz-63 sensor trajectory. The autoregressive frozen FNO and 140-dimensional polynomial-PCA BSO are shown over short, middle, and long lead panels with independent vertical scales; the short panel is logarithmic. BSO is comparable at short and intermediate leads and has lower long-range mean error.

#### L96-1: Lorenz-96 relative forecast error

> Pooled relative $L^2$ forecast error over 64 overlapping origins from the held-out Lorenz-96 sensor trajectory. The autoregressive frozen FNO and 30-dimensional quadratic-PCA BSO are shown over short, middle, and long lead panels with independent linear vertical scales. BSO is lower at every positive lead and has 33.3% lower all-lead mean error.

#### SST-1: 1--90-day forecast skill

> Global SST forecast skill over 96 held-out 2020--2022 origins after fitting on 1993--2019. Eleven ocean--atmosphere fields initialize learned methods, but only SST is verified. Panels show area-weighted RMSE and MAE through day 30, and RMSE and centered anomaly ACC through day 90. BSO is competitive with damped persistence at short leads and best among complete-band methods over days 31--90; SFNO ends when complete-cohort scores become unavailable.

#### SST-2: 540-day numerical rollout stability

> Area-weighted physical SST RMSE in a 540-day numerical stability test over 48 origins, shown over full and zoomed logarithmic ranges. BSO remains bounded, FNO reaches 816.327 K at day 540, and SFNO becomes nonfinite at day 171. FNO/SFNO validity uses the full 11-channel state, whereas BSO uses decoded SST; this is not a long-range skill ranking.

#### EEG-1: four-second resting-state rollout

> Cumulative relative $L^2$ error for a four-second rollout treated as autonomous on the S029R02 eyes-closed resting EEG record. Each step advances 0.1 seconds, and all methods use the same 15--19 second forecast window. BSO has the lowest step-40 error, 14.219% below direct FNO; the result covers one subject, one run, and one rollout.

## 9. Reporting summary

The benchmarks answer different scientific questions and should not be ranked
by one universal metric.

| Experiment | Principal reported result | Main interpretation limit |
|---|---|---|
| OU | Fixed-dictionary MAE `0.001338` over the first six nonconstant generator modes; learned-dictionary MAE `0.1660` over targets 0–5 | The fixed dictionary supplies the analytic function family; learned eigenfunction panels include postprocessing |
| Triple well | Six-mode generator MAE `0.0649`, maximum error `0.2768`; leading eigenfunctions recover the three-well partition qualitatively | No quantitative eigenfunction reference or repeated-seed interval |
| Lorenz-63 | All-lead mean relative $L^2$: BSO `0.94820`, FNO `1.11007` (14.6% reduction) | Short-band BSO mean is 9.1% higher; the held-out tail is not a newly sealed blind test |
| Lorenz-96 | All-lead mean relative $L^2$: BSO `0.96680`, FNO `1.44955` (33.3% reduction) | Long-range values measure stable statistical behavior, not phase-accurate trajectories |
| SST | Over days 31–90, BSO gives `0.4933 K / 0.3601 K / 0.598` RMSE/MAE/ACC and remains bounded in the separate 540-day stress test | One chronological split and seed; the 540-day result is numerical stability, not forecast skill |
| EEG | Four-second cumulative relative $L^2$: BSO `0.749440902`, 14.219% below direct FNO | One subject, one run, one forecast window, and no uncertainty interval |

## 10. Reproducibility and scope limits

- Reported values use one fixed run/seed configuration per experiment; no
  repeated-seed ensemble, confidence interval, or significance test is
  reported.
- Generated spectral benchmarks, chronological Lorenz/SST forecasts, and the
  single EEG window answer different questions. Their metrics are not directly
  interchangeable.
- Forecast origins overlap in Lorenz, SST, and EEG. Phase-B refinements reuse
  fitting data, and fitting-set diagnostics are not held-out scores.
- The Lorenz chronological tails and EEG window were available during method
  development; they are held out from fitting but are not newly sealed blind
  tests.
- SST auxiliary variables are launch-state inputs, not forecast targets, and
  no future auxiliary observations are supplied during rollout.
- Raw data, checkpoints, and runtime result bundles are ignored by Git.
  Committed figures and notebook outputs preserve the reference presentation.
  Full Lorenz and SST numerical regeneration requires the corresponding local
  artifacts.
- Available compact checkpoints do not retain every optimizer state,
  validation history, or generation seed. The fixed recipes therefore combine
  current code, notebook calls, checkpoints, and executed outputs.
- NumPy, PyTorch, and CUDA are seeded where applicable, but deterministic CUDA
  algorithms are not enforced; retraining is not guaranteed to be bitwise
  identical across hardware or library builds.
- A manuscript or archived release should cite an immutable commit, tag, or
  release DOI rather than a mutable working tree.

The six `nb_*.ipynb` files are the presentation entry points. OU,
Triple-well, and EEG recompute directly; EEG downloads its public record when
needed. Lorenz-63, Lorenz-96, and SST use local numerical artifacts when
available and otherwise display committed reference figures. Full training
and evaluation are explicit command-line tasks:

```text
python -m experiments.ou.exp_ou --help
python -m experiments.triple_well.exp_triple_well --help
python -m experiments.lorenz63.exp_lorenz63 --help
python -m experiments.lorenz96.exp_lorenz96 --help
python -m experiments.sst.download_sst_data --help
python -m experiments.sst.exp_sst --help
python -m experiments.sst.exp_sst evaluation <fit|preflight-evaluate|evaluate> --help
python -m experiments.eeg.exp_eeg_resting_reproduction --help
```
