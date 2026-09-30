# Contents
- [On MLP inputs](#on-mlp-inputs)
  - [Verification of duplication-compatible models](#verification-of-duplication-compatible-models)
  - [Verification of the matrix-product models](#verification-of-the-matrix-product-models) 
  - [Verification of scale equivariance](#verification-of-scale-equivariance) 
- [On CNN inputs](#on-cnn-inputs)
  - [Verification of duplication-compatible models](#verification-of-duplication-compatible-models-1)
  - [Verification of the matrix-product models](#verification-of-the-matrix-product-models-1)
  - [Verification of scale equivariance](#verification-of-scale-equivariance-1)
---
# On MLP inputs
## Verification of duplication-compatible models

Three modifications make a GNN meta-network respect **uniform duplication equivalence** end-to-end,
i.e. invariance under
$$
W^{(l)}_{\mathrm{up}} = W^{(l)} \otimes \frac{1}{k_{l-1}}\,\mathbf{1}_{k_l}\mathbf{1}_{k_{l-1}}^{\top},
\qquad
b^{(l)}_{\mathrm{up}} = b^{(l)} \otimes \mathbf{1}_{k_l}
$$
(eq:widened-duplication), where $k_{l-1}, k_l$ are the duplication factors of layers $l-1, l$ and
$\mathbf{1}_k \in \mathbb{R}^k$ denotes the all-ones vector:

1. **Fan-in rescaling**: edge features initialised as $d_{l-1} \cdot W^{(l)}$.
2. **Mean aggregation**: replaces default sum aggregation in message passing.
3. **Layer-wise mean readout** (`LayerWiseMeanReadout`): per-layer mean pooling over hidden nodes, replacing sum pooling.

All three are enabled together via `--duplication-equiv` in `train_sizegen.py`.

This section covers the two **duplication-compatible** model families used in the experiments below,
ScaleGMN and plain GMN, both of which retain their edgewise-MLP message functions
(script: `scripts/check_duplication_equiv.py`). Uniform duplication is the *only* member of the
equivalence hierarchy they respect; the strictly stronger guarantees of the matrix-product models are verified separately in [Verification of the matrix-product models](#verification-of-the-matrix-product-models).

Each table below is the full ablation: the reference condition, then each modification switched off
in turn, per direction. At node level only the first two apply (the readout is downstream), so it
has three rows per direction and the graph level four.

Scope: 5 w32 SP INRs, one pair of hidden-layer multipliers $k \in \{2,3,4\}$ drawn per INR,
$d_{\mathrm{hid}} = 32$, 2 message-passing rounds, untrained, seed 0, tolerance $10^{-5}$, float32.

**Date**: 2026-09-15

### Plain GMN (`symmetry: permutation`)

Plain GNN layers (`GNN_layer`) with concat+MLP message/update functions — no scale equivariance.
`DeepSet` readout (sum pooling) is the default; `LayerWiseMeanReadout` replaces it for the
duplication-equivariant variant (using a plain MLP instead of `InvariantLayer` since no
canonicalization is needed for permutation symmetry).
Tested on ReLU INR data (`--symmetry permutation --data-dir .../w32_sp`).

**Node-level (before readout):**

| Direction | Condition | Max error | Result |
|---|---|---|---|
| forward | fan-in + mean | $4.5\times10^{-7}$ | **PASS** |
| forward | no fan-in + mean | $9.2\times10^{-2}$ | FAIL |
| forward | fan-in + sum | $7.1\times10^{-2}$ | FAIL |
| bidirectional | fan-in + mean | $4.8\times10^{-7}$ | **PASS** |
| bidirectional | no fan-in + mean | $1.4\times10^{-1}$ | FAIL |
| bidirectional | fan-in + sum | $1.6\times10^{-1}$ | FAIL |

**Graph-level output:**

| Direction | Condition | Max error | Result |
|---|---|---|---|
| forward | fan-in + mean + lw-readout | $1.8\times10^{-7}$ | **PASS** |
| forward | no fan-in + mean + lw-readout | $4.3\times10^{-3}$ | FAIL |
| forward | fan-in + sum + lw-readout | $1.3\times10^{-2}$ | FAIL |
| forward | fan-in + mean + sum-readout | $1.8\times10^{-1}$ | FAIL |
| bidirectional | fan-in + mean + lw-readout | $2.4\times10^{-7}$ | **PASS** |
| bidirectional | no fan-in + mean + lw-readout | $9.9\times10^{-3}$ | FAIL |
| bidirectional | fan-in + sum + lw-readout | $3.6\times10^{-2}$ | FAIL |
| bidirectional | fan-in + mean + sum-readout | $1.2\times10^{-1}$ | FAIL |

### ScaleGMN (`symmetry: scale`)

Scale-equivariant GNN layers (`ScaleEq_GNN_layer`) with `EquivariantNet` message/update functions.
Tested on ReLU INR data (`--symmetry scale --data-dir .../w32_sp`).

**Node-level (before readout):**

| Direction | Condition | Max error | Result |
|---|---|---|---|
| forward | fan-in + mean | $1.5\times10^{-7}$ | **PASS** |
| forward | no fan-in + mean | $1.9\times10^{-3}$ | FAIL |
| forward | fan-in + sum | $5.8$ | FAIL |
| bidirectional | fan-in + mean | $1.8\times10^{-7}$ | **PASS** |
| bidirectional | no fan-in + mean | $9.0\times10^{-3}$ | FAIL |
| bidirectional | fan-in + sum | $1.9\times10^{1}$ | FAIL |

**Graph-level output:**

| Direction | Condition | Max error | Result |
|---|---|---|---|
| forward | fan-in + mean + lw-readout | $1.9\times10^{-7}$ | **PASS** |
| forward | no fan-in + mean + lw-readout | $1.2\times10^{-3}$ | FAIL |
| forward | fan-in + sum + lw-readout | $6.7\times10^{-1}$ | FAIL |
| forward | fan-in + mean + sum-readout | $1.5\times10^{-1}$ | FAIL |
| bidirectional | fan-in + mean + lw-readout | $3.0\times10^{-7}$ | **PASS** |
| bidirectional | no fan-in + mean + lw-readout | $3.2\times10^{-3}$ | FAIL |
| bidirectional | fan-in + sum + lw-readout | $5.2\times10^{-1}$ | FAIL |
| bidirectional | fan-in + mean + sum-readout | $3.1\times10^{-2}$ | FAIL |

---

## Verification of the matrix-product models

Two models constrain the MSG function to a multiplication by the raw scalar weight,
$$
\mathrm{msg} = \bar W^{(l)}_{ij} \cdot (h_j A),
\qquad A \in \mathbb{R}^{d_h \times d_h}\ \text{bias-free},
$$
so that the input weights enter the metanetwork **only through matrix multiplication**:

- the **matrix-product GMN** (`message_fn_type: matrix_product`, `mpgmn_*` configs) — the dup-equiv
  plain GMN with that constraint, keeping its concat-MLP node update (paper §"Matrix-product GMN");
- the **matrix-product ScaleGMN** (`message_fn_type: matrix_product_scale`, `mpsgmn_*` configs) — the
  same constraint on top of **ScaleGMN's** scale-equivariant node states and update instead (paper
  §"Forward matrix-product ScaleGMN"). For $\ell \ge 1$:
  $$
  Z_{\mathrm{sc}}^{(\ell),\mathrm{in}}
  = \frac{1}{n_{\ell-1}} \bar W^{(\ell)} H_{\mathrm{sc}}^{(\ell-1)} A_{\ell,\mathrm{sc}}^{\mathrm{in}},
  \qquad
  \tilde H_{\mathrm{sc}}^{(\ell)}
  = \Psi^{\mathrm{sc}}_{h,\ell}\!\left(H_{\mathrm{sc}}^{(\ell)},\, Z_{\mathrm{sc}}^{(\ell),\mathrm{in}}\right),
  $$
  i.e. the ScaleGMN message $\Psi_m(w_e(e) \odot w_v(h))$ restricted to $w_e = \mathrm{id}$,
  $w_v = \mathrm{id}$, $\Psi_m(x) = xA$, with the update $\Psi^{\mathrm{sc}}_{h,\ell}$ still
  ScaleGMN's `EquivariantNet` on $\mathrm{cat}(\text{node state},\ \text{aggregate})$.

Both keep all three modifications of the previous section — the constraint sits on top of them, so
both are always run with `--duplication-equiv True` and neither has a baseline variant. Both are
implemented by rebinding the layer classes in the `src.scalegmn.models` namespace inside a context
manager (`src/models/conv_matrix_product_layer.py`, `src/models/matrix_product_scale_layer.py`), so
the git subtree is untouched. What the constraint buys, and what this section verifies, are two
properties the edgewise-MLP GMNs do not have: the **general blockwise** equivalence and **spectral
continuity**.

**Directions.** The matrix-product GMN is run in both directions (MNIST v3/v4 `mpgmn_*`), so it is
verified in both. The matrix-product ScaleGMN is **forward-only in every training experiment**
(`FORWARD_ONLY` in the checkers) and has no bidirectional column in the spectral-continuity or
uniform-duplication-equivalence checks below. The one exception is the equivalence-class table just
below (subtable (b)): its bidirectional column exists purely to complete that contrast, via a
bidirectional MSG+AGGR layer (`MatrixProductScale_GNN_layer_aggr`) that the code provides for
exactly this and that no training experiment uses (`--include-bidir-scalegmn`).

Scripts: `scripts/check_matrix_product_gmn.py [--family {gmn,scalegmn}] --n-inrs 10` (untrained
models, $d_{\mathrm{hid}} = 32$, 2 message-passing rounds, 10 w16 SP INRs, seed 0, tolerance
$10^{-5}$) and, for the ablation of the three duplication components,
`scripts/check_duplication_equiv.py --template-conf ... --forward-only`. Each family carries its own
edgewise-MLP sibling (`gmn_sizegen_sp.yml`, `scalegmn_sizegen_sp_v3.yml`) as a contrast column, so
the constrained and unconstrained MSG functions are compared under identical conditions. All checks
below pass.

**Date**: 2026-08-29. Test 2 and Test 3 numbers depend on how much RNG earlier tests consume;
everything below is from a single run per family at `--n-inrs 10`.

**Update 2026-09-15**: the `gmn` family's contrast column was forward-only (`plain GMN` above);
`scripts/check_matrix_product_gmn.py` now also builds a bidirectional edgewise-MLP GMN and scores
it in the same Test 2 loop (`plain GMN bd` column), so the `mp-GMN fw`/`mp-GMN bd`/`plain GMN fw`
numbers in the table below were re-measured together with it in one run (`--family gmn --n-inrs 10`,
same script args as before). This is now subtable (a) below.

**Update 2026-09-15 (2)**: added `--include-bidir-scalegmn`, which also builds a bidirectional
matrix-product ScaleGMN and a bidirectional plain ScaleGMN in Test 2, via
`MatrixProductScale_GNN_layer_aggr` (`src/models/matrix_product_scale_layer.py`) — provided in the
code specifically for this contrast, even though no training experiment uses either bidirectional
model. Split the table into (a) GMN family and (b) ScaleGMN family so (b) can carry its own 4
columns; all four of (b)'s numbers (`mp-ScaleGMN fw`, `mp-ScaleGMN bd`, `plain ScaleGMN fw`,
`plain ScaleGMN bd`) are freshly measured (`--family scalegmn --n-inrs 10 --include-bidir-scalegmn`),
superseding the old forward-only `mp-ScaleGMN fw`/`plain ScaleGMN` pair.

### Matrix-product identity

The aggregate produced by PyG message passing matches the closed form, and input-layer nodes
receive the empty aggregate $z^{(0),\mathrm{in}} := 0$ (layer 0 has no incoming weight matrix):

| Check | mp-GMN | mp-ScaleGMN | Result |
|---|---|---|---|
| aggregate $= \dfrac{1}{n_{l-1}} \bar W^{(l)} H^{(l-1)} A$ | $3.8\times10^{-6}$ | $1.9\times10^{-6}$ | **PASS** |
| input-layer aggregate $= 0$ | $0$ | $0$ | **PASS** |

### Equivalence classes: forward vs bidirectional

Two blockwise conditions on the widened weight blocks $W^{(l)}_{\mathrm{up},ij} \in \mathbb{R}^{k_l \times k_{l-1}}$
matter, the *row*-sum one required by the forward aggregate and the *column*-sum one additionally
required by the backward aggregate. Five families of functionally equivalent widenings, all with
$k_0 = k_L = 1$ and $b^{(l)}_{\mathrm{up}} = b^{(l)} \otimes \mathbf{1}_{k_l}$ (and, where a single
Kronecker factor is used, $P^{(l)} \in \mathbb{R}^{k_l \times k_{l-1}}$):

| Family | Widened weights | Satisfies |
|---|---|---|
| uniform | $W^{(l)}_{\mathrm{up}} = W^{(l)} \otimes \dfrac{1}{k_{l-1}}\mathbf{1}_{k_l}\mathbf{1}_{k_{l-1}}^{\top}$ (eq:widened-duplication) | row + col |
| row-stoch | $W^{(l)}_{\mathrm{up}} = W^{(l)} \otimes P^{(l)}$, $P^{(l)}\mathbf{1}_{k_{l-1}} = \mathbf{1}_{k_l}$ (eq:widened-row-stochastic) | row |
| doubly-stoch | as above, additionally $(P^{(l)})^{\top}\mathbf{1}_{k_l} = \dfrac{k_l}{k_{l-1}}\mathbf{1}_{k_{l-1}}$ | row + col |
| general-bidir | blockwise, non-Kronecker: independent random block per $(i,j)$, subject to (row) $W^{(l)}_{\mathrm{up},ij}\,\mathbf{1}_{k_{l-1}} = W^{(l)}_{ij}\,\mathbf{1}_{k_l}$ and (col) $\mathbf{1}_{k_l}^{\top} W^{(l)}_{\mathrm{up},ij} = \dfrac{k_l}{k_{l-1}} W^{(l)}_{ij}\,\mathbf{1}_{k_{l-1}}^{\top}$ (paper §"The bidirectional variant … only a subset of them") | row + col |
| general | blockwise, non-Kronecker: independent random block per $(i,j)$, subject to (row) only (eq:widened-general-duplication-equivalence) | row |

Graph-level output error between base (w16) and widened ($k = 4$ per hidden layer, i.e. w64).
No training experiment uses a bidirectional matrix-product ScaleGMN or a bidirectional plain
ScaleGMN (the paper defines only the forward mpsgmn), but `MatrixProductScale_GNN_layer_aggr`
(`src/models/matrix_product_scale_layer.py`) exists specifically so this contrast can be completed
here, via `scripts/check_matrix_product_gmn.py --family scalegmn --include-bidir-scalegmn`.

**(a) GMN family** (`--family gmn`):

| Family | row | col | mp-GMN fw | mp-GMN bd | plain GMN fw | plain GMN bd |
|---|---|---|---|---|---|---|
| uniform | ✓ | ✓ | $2.4\times10^{-7}$ **P** | $3.3\times10^{-7}$ **P** | $1.8\times10^{-7}$ **P** | $2.7\times10^{-7}$ **P** |
| row-stoch | ✓ | ✗ | $2.4\times10^{-7}$ **P** | $1.6\times10^{-1}$ F | $2.9\times10^{-2}$ F | $2.5\times10^{-2}$ F |
| doubly-stoch | ✓ | ✓ | $3.0\times10^{-7}$ **P** | $2.4\times10^{-7}$ **P** | $4.4\times10^{-3}$ F | $9.8\times10^{-3}$ F |
| general-bidir | ✓ | ✓ | $2.4\times10^{-7}$ **P** | $2.7\times10^{-7}$ **P** | $8.9\times10^{-3}$ F | $2.0\times10^{-2}$ F |
| general | ✓ | ✗ | $2.4\times10^{-7}$ **P** | $1.4\times10^{-1}$ F | $3.1\times10^{-2}$ F | $3.0\times10^{-2}$ F |

**(b) ScaleGMN family** (`--family scalegmn --include-bidir-scalegmn`):

| Family | row | col | mp-ScaleGMN fw | mp-ScaleGMN bd | plain ScaleGMN fw | plain ScaleGMN bd |
|---|---|---|---|---|---|---|
| uniform | ✓ | ✓ | $6.6\times10^{-7}$ **P** | $7.2\times10^{-7}$ **P** | $2.1\times10^{-7}$ **P** | $1.8\times10^{-7}$ **P** |
| row-stoch | ✓ | ✗ | $8.6\times10^{-7}$ **P** | $6.0\times10^{-1}$ F | $8.8\times10^{-2}$ F | $1.0\times10^{-1}$ F |
| doubly-stoch | ✓ | ✓ | $4.7\times10^{-7}$ **P** | $6.6\times10^{-7}$ **P** | $7.6\times10^{-2}$ F | $6.9\times10^{-2}$ F |
| general-bidir | ✓ | ✓ | $4.8\times10^{-7}$ **P** | $7.5\times10^{-7}$ **P** | $1.0\times10^{-1}$ F | $7.2\times10^{-2}$ F |
| general | ✓ | ✗ | $5.0\times10^{-7}$ **P** | $4.6\times10^{-1}$ F | $1.0\times10^{-1}$ F | $8.0\times10^{-2}$ F |

This is §"Compatibility with the general blockwise equivalence", for **both** matrix-product models:

1. **Forward variant preserves the full general equivalence** — all five families pass, in both
   (a) and (b). The matrix product by a duplicated node-state matrix only ever sees the block row
   sums, so (row) alone suffices.
2. **Bidirectional variant preserves exactly the row+col subclass** — the backward aggregate uses
   $(\bar W^{(r+1)})^{\top}$, so it needs (col) as well: it passes on all three row+col families and
   fails on both row-only ones, in both (a) and (b). general-bidir vs. general — same non-Kronecker
   block structure, differing only in (col) — isolates (col) as the binding condition, not
   "Kronecker with doubly-stochastic $P$". `mp-ScaleGMN bd` in (b) is not used by any training
   experiment (see above) but shows the same pattern as `mp-GMN bd`, confirming the property is a
   statement about the matrix-product message, not about ScaleGMN's node states/update on top of it.
3. **The edgewise-MLP models are restricted to uniform duplication** — fan-in rescaling makes
   the widened edge features individually equal only for uniform $P$; any other family changes
   the individual scalars fed to the message MLP, so both `plain GMN fw` and `plain ScaleGMN fw`
   fail on the other four. Their bidirectional counterparts (`plain GMN bd`, `plain ScaleGMN bd`)
   fail the same four families: the restriction sits on the edge features feeding the MSG network,
   not on the message-passing direction, so it does not narrow further when the backward messages
   are added.

The constraint **composes** with ScaleGMN's node states and update: the scale-equivariant update is
applied *after* aggregation and identically to every copy of a node, so it cannot break an
equivalence the aggregate already respects.

Repeating with $k = 3$ on the (a) GMN family: same pass/fail pattern, passing errors
$\sim 2$–$3\times10^{-7}$, failures $\sim 4\times10^{-3}$–$1.3\times10^{-1}$ (not re-checked for (b)).

All three points are also measured as task accuracy in `results/EXPERIMENTS_MNIST_INR_classification.md`
§ Experiment 2, which scores the 11 trained MNIST-INR conditions on w24 test INRs widened to
w48–w240 under both `general` and `uniform`. Forward matrix-product: max_drop $0.0$% under
`general` ($10\times$ widening); bidirectional mp-GMN: $0.0$% under `uniform`, $70.7$–$72.5$% under
`general`; edgewise-MLP dup-equiv: $0.0$% under `uniform`, $70.2$–$83.9$% under `general`.

### Spectral continuity (Hadamard family)

Set $\bar W^{(2)} = H_n$ (Sylvester Hadamard) and all other weights/biases to zero, so the
normalized spectral norm is $\|\theta_n\| = \|H_n\|/n = 1/\sqrt{n} \to 0$. A model Lipschitz w.r.t.
that norm must satisfy $f(\theta_n) \to f(0)$; the paper's counterexample shows the edgewise-MLP
GMN need not.

| $n$ | $1/\sqrt{n}$ | mp-GMN $\lvert f(\theta_n) - f(0)\rvert$ | plain GMN $\lvert f(\theta_n) - f(0)\rvert$ |
|---|---|---|---|
| $16$ | $0.2500$ | $2.08\times10^{-3}$ | $4.99\times10^{-3}$ |
| $32$ | $0.1768$ | $1.04\times10^{-3}$ | $5.05\times10^{-3}$ |
| $64$ | $0.1250$ | $5.21\times10^{-4}$ | $5.08\times10^{-3}$ |
| $128$ | $0.0884$ | $2.60\times10^{-4}$ | $5.09\times10^{-3}$ |
| $256$ | $0.0625$ | $1.30\times10^{-4}$ | $5.10\times10^{-3}$ |

Matrix-product GMN: decays $16\times$ over the range, $\sim 1/n$ (at least as fast as the
$1/\sqrt{n}$ Lipschitz bound). Plain GMN: flat. Absolute magnitudes are small since the models are
untrained; only the trend in $n$ is scored.

This zero-context family is **vacuous under `symmetry: scale`**, i.e. for both models of the
ScaleGMN family: all five rows are exactly $0$ for both. ScaleGMN's message and update are
degree-one homogeneous, so with all biases zero and $W^{(1)} = 0$ every hidden state is exactly $0$
and $f(\theta_n) = f(0) = 0$ identically — the comparison says nothing about either model. The
checker detects this (both models' first-width difference below tolerance), reports the family as
vacuous and does not score it.

The **`hadamard+context`** family is the fix, and is what the ScaleGMN family is scored on: perturb
only layer 2 exactly as before, but keep a fixed, width-independent context in the other layers
whose fan-in-rescaled edge features are $O(1)$. The layer-2 normalized spectral norm is still
exactly $1/\sqrt{n}$, and the states are no longer identically zero. Both families behave the same
way on it:

| $n$ | $1/\sqrt{n}$ | mp-GMN | plain GMN | mp-ScaleGMN | plain ScaleGMN |
|---|---|---|---|---|---|
| $16$ | $0.2500$ | $3.46\times10^{-3}$ | $3.69\times10^{-3}$ | $5.39\times10^{-3}$ | $2.89\times10^{-3}$ |
| $32$ | $0.1768$ | $1.73\times10^{-3}$ | $4.01\times10^{-3}$ | $2.70\times10^{-3}$ | $2.87\times10^{-3}$ |
| $64$ | $0.1250$ | $8.67\times10^{-4}$ | $4.16\times10^{-3}$ | $1.35\times10^{-3}$ | $2.85\times10^{-3}$ |
| $128$ | $0.0884$ | $4.34\times10^{-4}$ | $4.24\times10^{-3}$ | $6.74\times10^{-4}$ | $2.85\times10^{-3}$ |
| $256$ | $0.0625$ | $2.17\times10^{-4}$ | $4.28\times10^{-3}$ | $3.37\times10^{-4}$ | $2.84\times10^{-3}$ |

**PASS** — exact halving per width doubling for both matrix-product models ($\sim 1/n$, at least as
fast as the $1/\sqrt{n}$ the Lipschitz bound guarantees), a $16\times$ decay over the range, while
both edgewise-MLP siblings stay flat (ratios $1.16$ and $0.98$ over the same $16\times$ width range)
despite the input converging to the zero network.

### Uniform duplication equivalence: all three components are still required

The constraint is *on top of* fan-in rescaling, mean aggregation and the layer-wise mean readout, so
those must still each be load-bearing. Script:
`scripts/check_duplication_equiv.py --template-conf configs/mnist_cls/mpsgmn_sizegen_sp_v3.yml
--symmetry scale --data-dir .../mnist_inrs/w32_sp --forward-only` (5 w32 SP INRs, $k \in \{2,3,4\}$
per hidden layer, $d_{\mathrm{hid}} = 32$, 2 rounds, tolerance $10^{-5}$) — the same ablation as
[Verification of duplication equivalence](#verification-of-duplication-compatible-models), on the
matrix-product ScaleGMN's MSG function. `--forward-only` drops the bidirectional half of the
ablation; it leaves the forward numbers untouched:

| Level | Condition | Max error | Result |
|---|---|---|---|
| node | fan-in + mean | $1.3\times10^{-6}$ | **PASS** |
| node | no fan-in + mean | $6.3\times10^{-2}$ | FAIL |
| node | fan-in + sum | $1.4\times10^{4}$ | FAIL |
| graph | fan-in + mean + lw-readout | $2.8\times10^{-7}$ | **PASS** |
| graph | no fan-in + mean + lw-readout | $1.7\times10^{-2}$ | FAIL |
| graph | fan-in + sum + lw-readout | $8.6\times10^{-3}$ | FAIL |
| graph | fan-in + mean + sum-readout | $7.1\times10^{-2}$ | FAIL |

**PASS** — the reference condition passes and each of the three components, switched off on its own,
breaks the equivalence, so all three are load-bearing here too. Node-level
passing errors are an order of magnitude larger than the edgewise ScaleGMN's ($\sim 10^{-6}$ vs
$\sim 10^{-7}$) and the failing ones far larger ($10^{4}$ vs $10^{1}$) simply because the message is
the fan-in-rescaled raw weight $d_{l-1} W^{(l)}_{ij}$ multiplied straight into $h_j A$, with no MSG
network to compress its scale; the graph-level errors, after mean pooling, are back at
$\sim 3\times10^{-7}$. The matrix-product GMN needs no separate table — its `uniform` row in the
equivalence-class table above is the same statement, measured under the same three components.

---

## Verification of scale equivariance

Everything above concerns *widening*. This section checks the symmetry ScaleGMN is actually named
after: the neuron-wise positive rescaling of a ReLU MLP,
$$
W^{(l)}[i,j] \mapsto \frac{\lambda^{(l)}_j}{\lambda^{(l-1)}_i} W^{(l)}[i,j],
\qquad
b^{(l)}_j \mapsto \lambda^{(l)}_j b^{(l)}_j,
\qquad
\lambda^{(0)} = \mathbf{1}_{d_0},\ \ \lambda^{(L)} = \mathbf{1}_{d_L},\ \ \lambda^{(l)} > 0,
$$
which leaves the represented function unchanged because ReLU is positively homogeneous. Under test:
the two `symmetry: scale` models (edgewise ScaleGMN, matrix-product ScaleGMN) are **equivariant** at
node level ($h_v \mapsto \lambda_v h_v$) and **invariant** at graph level; the two
`symmetry: permutation` models (plain GMN, matrix-product GMN) are neither.

Script: `scripts/check_scale_equiv.py` (5 w24 SP INRs, layout $[2,24,24,1]$, untrained,
$d_{\mathrm{hid}} = 32$, 2 rounds, $\lambda \sim \mathrm{LogUniform}[1/4, 4]$ per hidden neuron,
seed 0, tolerance $10^{-5}$, float32). Unlike the duplication checkers it reads `symmetry` and
`message_fn_type` off each model's own config rather than overriding them — here they *are* the
model under test.

**Error metric.** $\max\lvert \text{got} - \text{want}\rvert / \max(1, \max\lvert
\text{want}\rvert)$ throughout: relative where the reference is large, absolute where it is small.
Neither pure metric works here — a random readout can put the graph output near zero, so a relative
error turns float32 noise into a $10^{-5}$ "failure", while $\lambda$ up to $4$ per layer inflates
the hidden states, making the same noise look large in an absolute one. Reference magnitudes are
quoted per model so the numbers can be read either way.

**Date**: 2026-08-29

### The transform is function-preserving

$\max\lvert f(\theta_\lambda)(x) - f(\theta)(x)\rvert = 2.65\times10^{-7}$ over 256 random inputs in
$[-1,1]^2$, $\lambda \in [0.251, 3.968]$. **PASS** — the sampled pairs really are the same function,
so whatever the tables below measure is the metanetwork's doing.

### Node-level equivariance and graph-level invariance

`node hidden` is $\max\lvert h_v(\theta_\lambda) - \lambda_v h_v(\theta)\rvert$ over hidden nodes,
`node I/O` the same over input/output nodes ($\lambda = 1$, so an invariance check — reported, not
scored), `graph upstm` the upstream `PermScaleInvariantReadout` and `graph lw-mean`
`LayerWiseMeanReadout`.

`bidirectional` means the model **as defined**: with the reciprocal backward edge feature $1/W[u,v]$
a bidirectional scale model requires, installed by `install_bidir_reciprocal`
([src/models/bidir_reciprocal.py](../src/models/bidir_reciprocal.py)) exactly as every entry point
does. It is not an option — without it the backward messages carry $\lambda_v^2/\lambda_u$ and the
model is not scale-equivariant at all, which is a broken model rather than a variant of one. The
matrix-product ScaleGMN has no bidirectional row: it is forward-only in every experiment.

| Model | Direction | Edges / aggr | node hidden | node I/O | graph upstm | graph lw-mean | Result |
|---|---|---|---|---|---|---|---|
| ScaleGMN | forward | fan-in + mean | $2.3\times10^{-7}$ | $1.1\times10^{-7}$ | $1.8\times10^{-7}$ | $1.2\times10^{-7}$ | **PASS** |
| ScaleGMN | forward | raw + add | $2.7\times10^{-7}$ | $5.7\times10^{-8}$ | $1.9\times10^{-7}$ | $1.8\times10^{-7}$ | **PASS** |
| ScaleGMN | bidirectional | fan-in + mean | $3.0\times10^{-7}$ | $8.9\times10^{-8}$ | $2.4\times10^{-7}$ | $1.5\times10^{-7}$ | **PASS** |
| ScaleGMN | bidirectional | raw + add | $6.7\times10^{-7}$ | $5.2\times10^{-7}$ | $2.4\times10^{-7}$ | $2.7\times10^{-7}$ | **PASS** |
| mp-ScaleGMN | forward | fan-in + mean | $4.8\times10^{-7}$ | $8.3\times10^{-7}$ | $2.2\times10^{-7}$ | $1.8\times10^{-7}$ | **PASS** |
| plain GMN | forward | fan-in + mean | $7.8\times10^{-1}$ | $2.6\times10^{-2}$ | $6.8\times10^{-2}$ | $3.9\times10^{-2}$ | FAIL (expected) |
| plain GMN | forward | raw + add | $7.0\times10^{-1}$ | $4.5\times10^{-2}$ | $3.4\times10^{-2}$ | $2.2\times10^{-2}$ | FAIL (expected) |
| plain GMN | bidirectional | fan-in + mean | $7.6\times10^{-1}$ | $3.6\times10^{-2}$ | $5.0\times10^{-2}$ | $2.1\times10^{-2}$ | FAIL (expected) |
| plain GMN | bidirectional | raw + add | $6.4\times10^{-1}$ | $6.8\times10^{-2}$ | $5.8\times10^{-2}$ | $5.1\times10^{-2}$ | FAIL (expected) |
| mp-GMN | forward | fan-in + mean | $9.0\times10^{-1}$ | $2.4\times10^{-1}$ | $5.6\times10^{-2}$ | $8.7\times10^{-2}$ | FAIL (expected) |
| mp-GMN | bidirectional | fan-in + mean | $8.8\times10^{-1}$ | $3.2\times10^{-1}$ | $1.7\times10^{-1}$ | $2.4\times10^{-1}$ | FAIL (expected) |

Reference magnitudes, $\max\lvert \lambda_v h_v\rvert$ / $\max\lvert f\rvert$: ScaleGMN
$5.0$ / $0.86$, mp-ScaleGMN $7.4$ / $1.1$, plain GMN $6.8$ / $1.2$, mp-GMN $6.9$ / $0.87$.
`mp-*` rows have no `raw + add` variant because those models are dup-equiv only, and the
`symmetry: permutation` rows need no reciprocal — it is a no-op there by construction
(`models.py:305` gates it on `symmetry != 'permutation'`).

**PASS** — both `symmetry: scale` models are exactly equivariant/invariant at float32 noise level, in
every direction, independent of fan-in rescaling and of mean vs. add. Permutation-symmetry models
fail by 5–6 orders of magnitude.

**The upstream code path, for the record.** Upstream assigns a `reciprocal` flag (`models.py:305`)
and never reads it — `models.py:418` hardcodes `bw_edge_attr = batch.edge_attr` — so its backward
messages carry $\lambda_v^2/\lambda_u$. On the same weights that path scores $9.6\times10^{-1}$ node
/ $8.2\times10^{-2}$ graph against $4.1\times10^{-7}$ / $3.0\times10^{-7}$ with
`install_bidir_reciprocal`. It is invisible upstream because their INR configs are `symmetry: sign`,
where $1/\lambda = \lambda$. The script keeps the uncorrected path only as a must-FAIL regression
guard (printed `bd:raw`), not as a configuration of the model. Bidirectional `symmetry: scale` runs
require `install_bidir_reciprocal` to be measurements of a scale-equivariant model:
`scripts/run_{mnist_cls,fmnist_cls}_scalegmn_bidir_recip.sh` (same configs, LR grid, epochs, splits).
`scripts/run_cifar10_predgen_scalegmn_bidir_recip.sh` exists but is **not** a valid retrain path —
running it trains, not just checks, on real CNN-zoo weights, and NaNs by step ~13 at every LR: the
smallest nonzero weight across `cnn_zoo/w16_mup16` (~1.2e-15) has a reciprocal (~8.4e14) that overflows
float32 inside `EquivariantNet`'s unnormalized io-node branch after a few backward-round threadings —
the same elementwise-division instability the ScaleGMN paper (arXiv:2406.10685, Appendix A.2) reports
and never resolved for positive-scale symmetry (their only bidirectional result is sign-symmetric).
This check's Test 4 above, on small synthetic CNNs, is unaffected — it is a forward-pass exactness
check, not a training run. See
[`src/models/bidir_reciprocal.py`](../src/models/bidir_reciprocal.py)
for the implementation. Forward results are unaffected, `symmetry: permutation` rows need nothing, and
`mpsgmn` is forward-only.

### The error does not grow with the size of the rescaling

$\max\lvert f(\theta_\lambda) - f(\theta)\rvert$ at graph level (`LayerWiseMeanReadout`), forward,
for $\lambda \sim \mathrm{LogUniform}[1/r, r]$:

| Model | $r = 2$ | $r = 4$ | $r = 16$ | $r = 64$ |
|---|---|---|---|---|
| ScaleGMN | $1.2\times10^{-7}$ | $1.2\times10^{-7}$ | $1.3\times10^{-7}$ | $8.9\times10^{-8}$ |
| mp-ScaleGMN | $2.5\times10^{-7}$ | $2.4\times10^{-7}$ | $2.2\times10^{-7}$ | $1.9\times10^{-7}$ |

**PASS** — flat over a $32\times$ range, i.e. what Tests 1–2 measure is float32 noise and not a
small-perturbation artefact of $r = 4$.

---

# On CNN inputs

Everything above was measured on MLP graphs (MNIST INRs). This section re-verifies the same two
properties on **convolutional** graphs, for the CNN accuracy-prediction experiments
([EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md](EXPERIMENTS_CIFAR10_CNN_accuracy_prediction.md)).

Base architecture: the Unterthiner et al. small CNN zoo network
$[1] \to \text{conv}_{3\times3/s2} \to [c] \to \text{conv}_{3\times3/s2} \to [c] \to \text{conv}_{3\times3/s2} \to [c] \to \text{GAP} \to \text{dense} \to [10]$,
ReLU ($4970$ parameters at $c = 16$). Implementation: `src/data/zoo_cnn.py`.

## Verification of function preservating

Script: `scripts/check_cnn_widening_equiv.py`. Widths $\{16, 32\}$ $\times$ multipliers
$k \in \{2, 3, 4\}$ $\times$ 3 random CNNs $\times$ 5 families, compared on random $32\times32$
inputs.

Widening acts on the **channel** axis, in exactly the same way as the MLP case: the five families of
[Equivalence classes](#equivalence-classes-forward-vs-bidirectional) carry over verbatim, with each
scalar weight $W^{(l)}_{ij}$ replaced by its $3\times3$ kernel and the row / column conditions
imposed per kernel offset. Nothing in the argument changes either — a convolution is linear in its
input channels, ReLU is elementwise, and global average pooling is linear and per-channel, so it
commutes with the duplication, and the dense head behaves like an MLP layer ($k_L = 1$). As before,
function preservation needs only the **row** condition; the **column** condition is what the
bidirectional matrix-product GMN additionally requires.

| Family | $\max\lvert f_{\mathrm{base}} - f_{\mathrm{wide}}\rvert$ | row cond err | col cond err | non-uniformity |
|---|---|---|---|---|
| uniform | $8.33\times10^{-17}$ | $2.78\times10^{-17}$ | $0$ | $0$ |
| row-stoch | $1.39\times10^{-16}$ | $1.11\times10^{-16}$ | $2.96\times10^{-1}$ | $2.96\times10^{-1}$ |
| doubly-stoch | $5.55\times10^{-17}$ | $6.94\times10^{-17}$ | $4.86\times10^{-17}$ | $1.03\times10^{-1}$ |
| general-bidir | $4.44\times10^{-16}$ | $1.89\times10^{-15}$ | $1.33\times10^{-15}$ | $3.65$ |
| general | $3.34\times10^{-13}$ | $1.55\times10^{-15}$ | $7.95$ | $4.09$ |

**PASS** — all five families preserve the function; non-uniformity confirms none collapsed to the
uniform widening. col-cond: violated by the two row-only families, satisfied by the other three.

Measured in **float64** (the script's default). In float32 the four Kronecker / near-Kronecker
families still pass at $3\times10^{-8}$–$2\times10^{-7}$, but general reaches $2.07\times10^{-4}$ —
genuine catastrophic cancellation, not a logic error: its blocks are $O(1)$ random entries whose row
sums cancel down to the $O(0.1)$ base weights, while its *row condition* error stays at
$1.18\times10^{-6}$. Hence the float64 default; `--dtype float32` reproduces the failure.

## Verification of duplication-compatible models

Script: `scripts/check_duplication_equiv_cnn.py`. Base width 16, uniform widening with $k \in \{2, 3\}$,
3 random CNNs per $k$, $d_{\mathrm{hid}} = 32$, 2 GNN layers, tolerance $10^{-6}$, both directions
(`--forward-only` drops the bidirectional half, for models with no bidirectional variant — mpsgmn_*;
same split as [Verification of duplication-compatible models](#verification-of-duplication-compatible-models)
on MLP graphs). Uniform duplication duplicates every edge feature identically regardless of
direction, so no `install_bidir_reciprocal` fix is needed for this property — that fix is what makes
*scale equivariance* hold bidirectionally (see [Verification of scale equivariance](#verification-of-scale-equivariance-1)), a different property from the one here.

**Date**: 2026-09-15

The three modifications carry over unchanged, with one detail specific to convolutions: after
upstream's `_transform_weights_biases` a layer's weight has shape $[n_{\mathrm{in}}, n_{\mathrm{out}}, k_h k_w]$
and the graph puts **one edge per input channel**, carrying the whole zero-padded $\mathbb{R}^9$
kernel. Fan-in rescaling is therefore by $n_{\mathrm{in}}$, not $n_{\mathrm{in}} \cdot k_h k_w$ —
mean aggregation averages over incoming *edges*, and there are $n_{\mathrm{in}}$ of them
(`src/data/cnn_zoo_dataset.py`, `FanInCNNZooDataset`).

Two readouts are checked, since the accuracy-prediction task reads the output layer rather than
pooling the whole graph:

- `layerwise_mean` — `LayerWiseMeanReadout`, as on the INR side;
- `last_layer` — `LastLayerReadout` (`src/models/last_layer_readout.py`), which reads only the 10
  output nodes. Those are never widened (since $k_L = 1$), so this readout is duplication-invariant
  *by construction* and needs only fan-in + mean. It exists because upstream's `readout_range:
  last_layer` path reshapes with the *train* width's `num_nodes` and so crashes at OOD widths; ours
  selects output nodes from `node2type`, which is width-independent.
- `sum-ro` — the upstream sum-pooling readout (`PermScaleInvariantReadout` / `DeepSet`), included as
  the negative control.

| Level / readout | Direction | Condition | ScaleGMN max err | plain GMN max err | Result |
|---|---|---|---|---|---|
| node | forward | fan-in + mean | $3.58\times10^{-7}$ | $3.87\times10^{-7}$ | **PASS** |
| node | forward | no fan-in + mean | $1.19\times10^{-4}$ | $2.97\times10^{-3}$ | FAIL |
| node | forward | fan-in + sum | $5.18\times10^{-2}$ | $1.22\times10^{-1}$ | FAIL |
| node | bidirectional | fan-in + mean | $3.28\times10^{-7}$ | $4.17\times10^{-7}$ | **PASS** |
| node | bidirectional | no fan-in + mean | $1.65\times10^{-4}$ | $1.33\times10^{-2}$ | FAIL |
| node | bidirectional | fan-in + sum | $3.55\times10^{-2}$ | $3.55\times10^{-1}$ | FAIL |
| graph / sum-ro | forward | fan-in + mean | $1.06\times10^{-2}$ | $1.37\times10^{-2}$ | FAIL |
| graph / sum-ro | bidirectional | fan-in + mean | $3.38\times10^{-2}$ | $2.57\times10^{-2}$ | FAIL |
| graph / layerwise_mean | forward | fan-in + mean | $1.19\times10^{-7}$ | $1.64\times10^{-7}$ | **PASS** |
| graph / layerwise_mean | forward | fan-in + sum | $3.40\times10^{-2}$ | $2.17\times10^{-2}$ | FAIL |
| graph / layerwise_mean | forward | no fan-in + mean | $3.67\times10^{-4}$ | $7.44\times10^{-4}$ | FAIL |
| graph / layerwise_mean | bidirectional | fan-in + mean | $2.61\times10^{-7}$ | $1.64\times10^{-7}$ | **PASS** |
| graph / layerwise_mean | bidirectional | fan-in + sum | $1.80\times10^{-2}$ | $3.26\times10^{-2}$ | FAIL |
| graph / layerwise_mean | bidirectional | no fan-in + mean | $4.59\times10^{-3}$ | $9.60\times10^{-4}$ | FAIL |
| graph / last_layer | forward | fan-in + mean | $1.94\times10^{-7}$ | $2.38\times10^{-7}$ | **PASS** |
| graph / last_layer | forward | fan-in + sum | $1.07\times10^{-3}$ | $3.89\times10^{-2}$ | FAIL |
| graph / last_layer | forward | no fan-in + mean | $2.62\times10^{-5}$ | $8.45\times10^{-4}$ | FAIL |
| graph / last_layer | bidirectional | fan-in + mean | $2.91\times10^{-7}$ | $1.49\times10^{-7}$ | **PASS** |
| graph / last_layer | bidirectional | fan-in + sum | $1.18\times10^{-2}$ | $2.34\times10^{-2}$ | FAIL |
| graph / last_layer | bidirectional | no fan-in + mean | $7.09\times10^{-5}$ | $3.44\times10^{-4}$ | FAIL |

**PASS** — for both symmetries, both readouts and both directions: equivalence holds only with
fan-in + mean; every ablation breaks it by at least one order of magnitude (smallest failure
$2.62\times10^{-5}$ vs largest pass $4.17\times10^{-7}$). Passing errors are float32 noise amplified
by the fan-in factors (up to $144$ here). Bidirectional passes by the same margin as forward — the
backward aggregate duplicates every widened edge feature identically to the forward one, so mean
aggregation collapses it the same way regardless of direction.

## Verification of the matrix-product models

Script: `scripts/check_matrix_product_gmn_cnn.py [--family {gmn,scalegmn}]`. Base width 8 (layout
$[1,8,8,8,10]$), `d_hid = 32`, 2 GNN layers, mean aggregation, tolerance $10^{-5}$, all in float32.
Configs under test: `configs/cifar10_predgen/mpgmn_sizegen_sp_v1.yml` and
`mpsgmn_sizegen_sp_v1.yml`, each against its edgewise-MLP sibling (`gmn_sizegen_sp_v1.yml`,
`scalegmn_sizegen_sp_v1.yml`) as a contrast column. As on MLP graphs, the conv matrix-product
**ScaleGMN** is forward-only in every *training* experiment (`FORWARD_ONLY` in the checkers); Test 2
below is the one place a bidirectional variant is built anyway, via `--include-bidir-scalegmn` (see
the 2026-09-15 update below). Test 2 consumes RNG ahead of Test 3, so its numbers are from a single
run at this scope (measured 2026-08-29).

**Update 2026-09-15**: the `gmn` family's edgewise-MLP contrast was forward-only (`plain GMN
(forward)` below); `scripts/check_matrix_product_gmn_cnn.py` now also builds a bidirectional
edgewise-MLP GMN and scores it in the same Test 2 loop (`plain GMN (bidirectional)` column). Test 1
and the `gmn` family's Test 2 numbers below were re-measured together with it in one full run
(`--family gmn`, same script defaults as before, `--only` unset so RNG consumption matches the
original methodology). This is now subtable (a) below.

**Update 2026-09-15 (2)**: switched Test 2's scoring readout from `last_layer` to `layerwise_mean`
(`--readout layerwise_mean`, new flag) and added `--include-bidir-scalegmn` (same mechanism as the
MLP script, via `MatrixProductScale_GNN_layer_aggr`) to also build a bidirectional matrix-product
ScaleGMN and a bidirectional plain ScaleGMN. Split the table into (a) GMN family and (b) ScaleGMN
family; (a)'s numbers were re-measured under `layerwise_mean` (superseding the `last_layer` numbers
above), and (b) is entirely new — its old forward-only `mp-forward`/`plain ScaleGMN (forward)` pair
is superseded by all four of its columns, freshly measured together in one run
(`--family scalegmn --readout layerwise_mean --include-bidir-scalegmn`).

**Update 2026-09-16**: synced into `paper/appendix.tex`'s `tab:verify-blockwise` — the MLP block's
GMN numbers were already current there (MLP never had a readout choice); the CNN block's GMN
numbers were updated to these `layerwise_mean` values, and two new blocks (`MLP weights, ScaleGMN
family`, `CNN weights, ScaleGMN family`) were added from subtable (b) above, using generic
`Matrix-product`/`Duplication-compatible` column headers (dropping the `GMN` suffix) since they now
head both a GMN and a ScaleGMN block.

On a CNN graph an edge carries a whole kernel in $\mathbb{R}^9$, so the scalar matrix-product
message has to become **bilinear** in the kernel and the source state, with one learned map per
kernel offset (`src/models/conv_matrix_product_layer.py`):
$$
\mathrm{msg}_{u\to v} = \sum_{s=1}^{9} a_{u,v}[s]\,(h_u A_s),
\qquad
A \in \mathbb{R}^{9 \times d_{\mathrm{in}}(v) \times d_{\mathrm{hid}}},
$$
so with mean aggregation the forward aggregate is a **sum of 9 matrix products**, one per spatial
offset:
$$
Z^{(l)} = \frac{1}{n_{l-1}} \sum_{s=1}^{9} \bar W_s^{(l)} H^{(l-1)} A_s.
$$
The input weights therefore enter the metanetwork only through matrix multiplication, which is what
makes it Lipschitz w.r.t. the normalized spectral norm. The layer is installed by rebinding
`GNN_layer` / `GNN_layer_aggr` (or, for the ScaleGMN family, `ScaleEq_GNN_layer` /
`ScaleGMN_GNN_layer_aggr`) in the `src.scalegmn.models` namespace inside a context manager, so the
git subtree is untouched. The ScaleGMN variant
(`message_fn_type: matrix_product_conv_scale`) uses this same message; only its node states and node
update are ScaleGMN's.

### Matrix-product identity
The PyG aggregate matches the closed form to $2.2\times10^{-7}$
(conv mp-GMN) and $1.2\times10^{-7}$ (conv mp-ScaleGMN), and is $0$ on the input layer for both (no
incoming weight matrix). **PASS**

### Equivalence classes: forward vs bidirectional
Channel multipliers $k \in \{2, 3\}$, 3 random CNNs per $k$, all
five families of `src/data/zoo_cnn.py`. Each family is first re-confirmed to be a genuine functional
equivalence (max $\lvert f_{\mathrm{base}} - f_{\mathrm{wide}}\rvert \le 3.2\times10^{-5}$ in
float32; see the float64 numbers above), to satisfy the row condition ($\le 7.9\times10^{-7}$), and
— except uniform — to be measurably non-uniform ($\ge 2.4\times10^{-2}$).

Node-level / graph-level max error, scored on the `layerwise_mean` readout (`--readout
layerwise_mean`; `last_layer` is the accuracy-prediction task's default readout, but see the caveat
below on why this section now uses `layerwise_mean` instead).

**(a) GMN family** (`--family gmn --readout layerwise_mean`):

| Family | col cond | mp-forward | mp-bidirectional | plain GMN (forward) | plain GMN (bidirectional) |
|---|---|---|---|---|---|
| uniform | ✓ | $4.2\times10^{-7}$ / $1.3\times10^{-7}$ **PASS** | $4.2\times10^{-7}$ / $1.9\times10^{-7}$ **PASS** | $4.5\times10^{-7}$ / $1.5\times10^{-7}$ **PASS** | $5.2\times10^{-7}$ / $4.5\times10^{-7}$ **PASS** |
| row-stoch | ✗ | $4.5\times10^{-7}$ / $1.9\times10^{-7}$ **PASS** | $6.1\times10^{-1}$ / $1.5\times10^{-2}$ FAIL | $1.8\times10^{-1}$ / $4.6\times10^{-3}$ FAIL | $3.8\times10^{-1}$ / $3.5\times10^{-2}$ FAIL |
| doubly-stoch | ✓ | $4.8\times10^{-7}$ / $1.2\times10^{-7}$ **PASS** | $5.2\times10^{-7}$ / $2.4\times10^{-7}$ **PASS** | $1.9\times10^{-1}$ / $3.1\times10^{-3}$ FAIL | $2.6\times10^{-1}$ / $4.6\times10^{-2}$ FAIL |
| general-bidir | ✓ | $1.2\times10^{-6}$ / $6.7\times10^{-8}$ **PASS** | $2.3\times10^{-6}$ / $2.4\times10^{-7}$ **PASS** | $4.3\times10^{-1}$ / $3.4\times10^{-2}$ FAIL | $5.4\times10^{-1}$ / $1.6\times10^{-1}$ FAIL |
| general | ✗ | $1.4\times10^{-6}$ / $2.7\times10^{-7}$ **PASS** | $1.4$ / $1.4\times10^{-1}$ FAIL | $4.6\times10^{-1}$ / $4.8\times10^{-2}$ FAIL | $6.1\times10^{-1}$ / $2.4\times10^{-1}$ FAIL |

**(b) ScaleGMN family** (`--family scalegmn --readout layerwise_mean --include-bidir-scalegmn`):

| Family | col cond | mp-forward | mp-bidirectional | plain ScaleGMN (forward) | plain ScaleGMN (bidirectional) |
|---|---|---|---|---|---|
| uniform | ✓ | $4.1\times10^{-7}$ / $2.4\times10^{-7}$ **PASS** | $3.7\times10^{-7}$ / $2.2\times10^{-7}$ **PASS** | $4.0\times10^{-7}$ / $1.8\times10^{-7}$ **PASS** | $3.4\times10^{-7}$ / $3.0\times10^{-7}$ **PASS** |
| row-stoch | ✗ | $4.1\times10^{-7}$ / $1.2\times10^{-7}$ **PASS** | $6.6\times10^{-2}$ / $1.7\times10^{-2}$ FAIL | $8.8\times10^{-4}$ / $5.1\times10^{-3}$ FAIL* | $4.0\times10^{-3}$ / $8.7\times10^{-3}$ FAIL* |
| doubly-stoch | ✓ | $4.1\times10^{-7}$ / $2.1\times10^{-7}$ **PASS** | $3.7\times10^{-7}$ / $1.2\times10^{-7}$ **PASS** | $6.9\times10^{-5}$ / $6.9\times10^{-4}$ FAIL* | $5.7\times10^{-5}$ / $1.6\times10^{-3}$ FAIL* |
| general-bidir | ✓ | $4.1\times10^{-7}$ / $2.4\times10^{-7}$ **PASS** | $3.7\times10^{-7}$ / $2.5\times10^{-7}$ **PASS** | $1.9\times10^{-3}$ / $2.1\times10^{-2}$ FAIL* | $2.6\times10^{-3}$ / $1.8\times10^{-2}$ FAIL* |
| general | ✗ | $3.6\times10^{-7}$ / $4.8\times10^{-7}$ **PASS** | $5.9\times10^{-1}$ / $7.6\times10^{-2}$ FAIL | $1.5\times10^{-2}$ / $1.3\times10^{-2}$ FAIL* | $1.0\times10^{-2}$ / $1.6\times10^{-2}$ FAIL* |

\* plain-ScaleGMN columns are reported, not scored (the script asserts pass/fail only for
`--family gmn`'s contrast column) — see the caveat below.

**PASS** — the forward conv matrix-product message respects all five families, the bidirectional
variant exactly the three that also satisfy the column condition, and the edgewise-MLP siblings only
uniform duplication — the same pattern as on MLP graphs, now confirmed in both directions for the
ScaleGMN family too. Node level is reported alongside graph level as the readout-independent
statement.

One caveat specific to the ScaleGMN family: its plain-ScaleGMN columns (forward and bidirectional)
are **reported but not scored**. On *conv* graphs the edgewise ScaleGMN message is degree-one
homogeneous in the kernel, so whenever the column condition holds it is invariant to *first order* in
the widening perturbation, which can make a column-condition family look like it passes at graph
level while it still fails at node level — under the `last_layer` readout an earlier run showed
exactly this at `doubly-stoch` ($3.6\times10^{-7}$ at graph level against $2.1\times10^{-4}$ at node
level at `num_layers = 2`, the gap narrowing but not closing at `--num-layers 4`). Under
`layerwise_mean` (pooling every layer instead of only the 10 output nodes) this run shows no such
crossing: every non-uniform family fails clearly at *both* levels, in *both* directions, so the
policy stays conservative (reported, not asserted) as a property of the readout choice, not because
this particular run needed the hedge.

### Spectral continuity.
Conv Hadamard family: $\bar W^{(2)}[:,:,\text{centre}] = H_n$
(i.e. $W^{(2)} = H_n/n$ on the centre kernel offset), every other weight and bias zero, so the
normalized spectral norm is $\sqrt{n_1/n_2}\cdot\|W^{(2)}\|_2 = 1/\sqrt{n} \to 0$ (single-offset
kernel, so this also bounds the conv operator norm). Scored on the `layerwise_mean` readout:

| $n$ | $\|\theta_n\|$ | mp-GMN $\lvert f(\theta_n) - f(0)\rvert$ | plain GMN |
|---|---|---|---|
| $16$ | $0.2500$ | $8.76\times10^{-4}$ | $6.68\times10^{-4}$ |
| $32$ | $0.1768$ | $4.38\times10^{-4}$ | $6.42\times10^{-4}$ |
| $64$ | $0.1250$ | $2.19\times10^{-4}$ | $6.30\times10^{-4}$ |
| $128$ | $0.0884$ | $1.10\times10^{-4}$ | $6.23\times10^{-4}$ |

**PASS** — the matrix-product model's output collapses onto $f(0)$ (exactly halving per width
doubling, i.e. $\propto 1/\sqrt{n}$, so it tracks the norm), while the plain GMN stays flat (ratio
$0.93$ over an $8\times$ width range) despite the input converging to the zero network. The
`last_layer` readout is reported but not scored: with $W^{(L)} = 0$ on this family the
matrix-product messages into the output nodes are identically zero at any depth, so the comparison
is vacuous.

This zero-context family is **vacuous under `symmetry: scale`**, i.e. for the ScaleGMN family, for
the same homogeneity reason as on MLP graphs: all four rows are exactly $0$ for both models under
both readouts. The `hadamard+context` family (fixed width-independent context outside layer 2) is
what the ScaleGMN family is scored on, and it gives the same verdict for both families on
`layerwise_mean`:

| $n$ | $\|\theta_n\|$ | mp-GMN | plain GMN | mp-ScaleGMN | plain ScaleGMN |
|---|---|---|---|---|---|
| $16$ | $0.2500$ | $3.24\times10^{-4}$ | $3.12\times10^{-4}$ | $4.42\times10^{-3}$ | $5.06\times10^{-4}$ |
| $32$ | $0.1768$ | $1.61\times10^{-4}$ | $2.07\times10^{-4}$ | $2.20\times10^{-3}$ | $5.06\times10^{-4}$ |
| $64$ | $0.1250$ | $7.99\times10^{-5}$ | $1.54\times10^{-4}$ | $1.10\times10^{-3}$ | $5.07\times10^{-4}$ |
| $128$ | $0.0884$ | $3.99\times10^{-5}$ | $1.28\times10^{-4}$ | $5.47\times10^{-4}$ | $5.07\times10^{-4}$ |

**PASS** — both matrix-product models halve per width doubling; the plain GMN decays only to ratio
$0.41$ (and not with the norm) and the plain ScaleGMN is flat to ratio $1.00$. `last_layer` is again
vacuous on this family for the matrix-product models: at `num_layers = 2` the layer-2 perturbation is
two hops upstream of the output nodes and cannot reach them at all.

### Uniform duplication equivalence
 (conv matrix-product ScaleGMN; the conv mp-GMN's `uniform` row in
Test 2 above is the same statement for that family).

 Script:
`scripts/check_duplication_equiv_cnn.py --template-conf configs/cifar10_predgen/mpsgmn_sizegen_sp_v1.yml
--symmetry scale` (base width 16, $k \in \{2,3\}$, 3 CNNs per $k$, tolerance $10^{-6}$):

| Level / readout | Condition | Max error | Result |
|---|---|---|---|
| node | fan-in + mean | $1.6\times10^{-7}$ | **PASS** |
| node | no fan-in + mean | $5.7\times10^{-4}$ | FAIL |
| node | fan-in + sum | $4.8\times10^{-1}$ | FAIL |
| graph / upstream sum-pooling | fan-in + mean | $1.3\times10^{-2}$ | FAIL |
| graph / `layerwise_mean` | fan-in + mean | $1.2\times10^{-7}$ | **PASS** |
| graph / `layerwise_mean` | fan-in + sum | $1.6\times10^{-1}$ | FAIL |
| graph / `layerwise_mean` | no fan-in + mean | $3.6\times10^{-3}$ | FAIL |
| graph / `last_layer` | fan-in + mean | $1.2\times10^{-7}$ | **PASS** |
| graph / `last_layer` | fan-in + sum | $2.4\times10^{-1}$ | FAIL |
| graph / `last_layer` | no fan-in + mean | $1.4\times10^{-4}$ | FAIL |

**PASS** — same pattern under both width-agnostic readouts, including `last_layer` (the default for
the accuracy-prediction task).

## Verification of scale equivariance

The MLP-graph section above has the derivation and the error-metric rationale; this is the same check
on convolutional graphs. Script: `scripts/check_scale_equiv_cnn.py` (3 random zoo CNNs at $c = 16$,
layout $[1,16,16,16,10]$, untrained, $d_{\mathrm{hid}} = 32$, 2 rounds,
$\lambda \sim \mathrm{LogUniform}[1/4, 4]$ per channel, seed 0, tolerance $10^{-5}$, float32 with
TF32 explicitly off).

The symmetry is per **channel**: $W^{(l)}[i,j,:,:] \mapsto (\lambda^{(l)}_i/\lambda^{(l-1)}_j)
W^{(l)}[i,j,:,:]$ on the convs, $W^{(L)}[i,j] \mapsto W^{(L)}[i,j] / \lambda^{(L-1)}_j$ on the dense
head, $b^{(l)}_i \mapsto \lambda^{(l)}_i b^{(l)}_i$, with $\lambda = 1$ on the single input channel
and the 10 logits. It preserves the function because a convolution is linear in its input channels
and GAP is linear and per-channel, so the head's $1/\lambda$ cancels the $\lambda$ carried by the
pooled features. On the graph an edge carries the whole zero-padded $\mathbb{R}^9$ kernel of one
(in-channel, out-channel) pair, so it scales by $\lambda_v/\lambda_u$ exactly as a scalar MLP weight
does, and the conv bilinear message $\sum_s a_{u,v}[s]\,(h_u A_s)$ scales by $\lambda_v$ term by term.

### The transform is function-preserving
$\max\lvert f(\theta_\lambda)(x) - f(\theta)(x)\rvert = 2.98\times10^{-8}$ on random
$32\times32$ inputs, $\lambda \in [0.251, 3.968]$. **PASS**

### Node-level equivariance and graph-level invariance
Both CNN readouts are scored (`layerwise_mean` and `last_layer`, the latter being
the default for the accuracy-prediction task); `node I/O` is reported, not scored. As on MLP graphs,
`bidirectional` means the model **as defined**, i.e. with the reciprocal backward edge feature
$1/W[u,v]$ installed by `install_bidir_reciprocal`, which `scripts/train_sizegen_predgen.py` and
`scripts/eval_sizegen_predgen_duplicated.py` call.
The conv matrix-product ScaleGMN has no bidirectional row at all: it is forward-only in every experiment (`FORWARD_ONLY`).

| Model | Direction | Edges / aggr | node hidden | node I/O | graph lw-mean | graph last-layer | Result |
|---|---|---|---|---|---|---|---|
| ScaleGMN | forward | fan-in + mean | $2.1\times10^{-8}$ | $2.7\times10^{-7}$ | $1.8\times10^{-7}$ | $1.8\times10^{-7}$ | **PASS** |
| ScaleGMN | forward | raw + add | $1.9\times10^{-8}$ | $5.2\times10^{-8}$ | $3.0\times10^{-8}$ | $1.5\times10^{-8}$ | **PASS** |
| ScaleGMN | bidirectional | fan-in + mean | $1.8\times10^{-8}$ | $2.3\times10^{-7}$ | $3.6\times10^{-7}$ | $2.1\times10^{-7}$ | **PASS** |
| ScaleGMN | bidirectional | raw + add | $1.3\times10^{-6}$ | $3.0\times10^{-7}$ | $3.0\times10^{-8}$ | $6.0\times10^{-8}$ | **PASS** |
| conv mp-ScaleGMN | forward | fan-in + mean | $4.1\times10^{-8}$ | $1.4\times10^{-7}$ | $2.4\times10^{-7}$ | $3.0\times10^{-8}$ | **PASS** |
| plain GMN | forward | fan-in + mean | $7.5\times10^{-1}$ | $1.0\times10^{-1}$ | $3.2\times10^{-2}$ | $3.7\times10^{-2}$ | FAIL (expected) |
| plain GMN | forward | raw + add | $7.4\times10^{-1}$ | $2.0\times10^{-2}$ | $3.2\times10^{-3}$ | $6.7\times10^{-3}$ | FAIL (expected) |
| plain GMN | bidirectional | fan-in + mean | $7.6\times10^{-1}$ | $9.5\times10^{-2}$ | $2.9\times10^{-2}$ | $3.4\times10^{-2}$ | FAIL (expected) |
| plain GMN | bidirectional | raw + add | $7.4\times10^{-1}$ | $3.6\times10^{-2}$ | $8.5\times10^{-3}$ | $1.8\times10^{-3}$ | FAIL (expected) |
| conv mp-GMN | forward | fan-in + mean | $8.0\times10^{-1}$ | $2.3\times10^{-1}$ | $4.4\times10^{-2}$ | $2.0\times10^{-2}$ | FAIL (expected) |
| conv mp-GMN | bidirectional | fan-in + mean | $8.0\times10^{-1}$ | $3.6\times10^{-1}$ | $1.1\times10^{-1}$ | $6.0\times10^{-2}$ | FAIL (expected) |

Reference magnitudes, $\max\lvert \lambda_v h_v\rvert$ / $\max\lvert f\rvert$: ScaleGMN
$40.1$ / $0.39$, conv mp-ScaleGMN $1.08$ / $0.35$, plain GMN $3.72$ / $0.87$, conv mp-GMN
$4.31$ / $0.45$. The `symmetry: permutation` rows need no reciprocal — it is a no-op there by
construction — and have no homogeneity structure for it to fix.

**PASS** — both `symmetry: scale` models are exactly equivariant and invariant under both readouts,
in every direction, independent of fan-in rescaling and of mean vs. add; the reciprocal backward
feature is what makes the bidirectional rows pass. Permutation-symmetry models fail by 5–7 orders of
magnitude.

### Insensitivity to the size of the rescaling.

Graph-level (`layerwise_mean`), forward:

| Model | $r = 2$ | $r = 4$ | $r = 16$ | $r = 64$ |
|---|---|---|---|---|
| ScaleGMN | $3.0\times10^{-8}$ | $1.8\times10^{-7}$ | $1.2\times10^{-7}$ | $1.8\times10^{-7}$ |
| conv mp-ScaleGMN | $8.9\times10^{-8}$ | $1.2\times10^{-7}$ | $3.0\times10^{-8}$ | $1.2\times10^{-7}$ |

**PASS** — flat over a $32\times$ range.