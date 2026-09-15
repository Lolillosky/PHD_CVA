# CCR Deep Learning Research Project

## Purpose

This project supports the development of a research paper and PhD thesis
chapter on **deep learning methods for path-dependent counterparty
credit risk (CCR) exposure modelling**.

The work should be treated as academic research aimed at a strong
quantitative finance / machine learning journal. Responses should
therefore be mathematically precise, critical, and research-oriented
rather than introductory.

The main methodological themes are:

-   Monte Carlo regression for conditional valuation and CCR exposure.
-   Temporal-Difference (TD) / Bellman learning.
-   Differential Machine Learning (DML).
-   Low-dimensional **current-state / local differential labels** rather
    than full historical-sequence differentials.
-   Feed-forward neural networks with engineered Markov states.
-   Recurrent neural networks, particularly LSTM/GRU.
-   Causal Transformers for learned path representations.
-   Path-dependent derivatives.
-   Rolling model updating / warm-start retraining as products age.
-   Sample efficiency and computational efficiency.

A central objective is to identify a coherent and defensible
methodological contribution rather than simply comparing many
machine-learning architectures.

------------------------------------------------------------------------

## Research Question

A useful overarching research question is:

> **How should conditional-expectation regressors for path-dependent
> counterparty exposures exploit information across the temporal and
> differential dimensions?**

A possible central hypothesis is:

> **Path-dependent counterparty exposure can be learned more efficiently
> by combining temporal bootstrapping with low-dimensional current-state
> differential information, while sequence models encode historical
> state.**

The paper should not claim novelty merely from using neural networks, TD
learning, DML, LSTMs, or Transformers individually. The potentially
novel contribution lies in their combination and, especially, in the use
of **local/current-state differential information for path-dependent
CCR**.

------------------------------------------------------------------------

## Core CCR Framework

At a CCR observation date $t$, define the derivative value as

$$ V_t = \mathbb{E}_t^Q \left[
\sum_{u>t} D(t,u) C_u
\right]. $$

Here:

-   $C_u$ denotes future contractual cash flows.
-   $D(t,u)$ is the discount factor from $u$ to $t$.
-   Valuation/continuation expectations are under the risk-neutral
    measure $Q$.
-   Exposure distributions may subsequently be generated under the
    physical measure $P$, depending on the CCR framework.

Past paid cash flows do not enter the current MtM.

Relevant CCR quantities may include:

-   $V_t$
-   Positive exposure $V_t^+ = \max(V_t,0)$
-   Expected Exposure (EE)
-   Potential Future Exposure (PFE)
-   Expected Positive Exposure / EPE
-   CVA or CVA-related quantities where appropriate
-   Distributional and tail errors

The distinction between $P$ and $Q$ must always be handled carefully.

------------------------------------------------------------------------

## Observation, Fixing, and Payment Dates

CCR observation dates need not coincide with fixing or payment dates.

Let

$$ \tau_0 < \tau_1 < \cdots <
\tau_N $$

be CCR observation dates and $T_k$ contractual payment dates.

For the TD transition from $\tau_i$ to $\tau_{i+1}$,
define the immediate reward as

$$ R_i =
\sum_{T_k \in (\tau_i,\tau_{i+1}]}
D(\tau_i,T_k) C_k. $$

The Bellman recursion is then

$$ V_{\tau_i} = \mathbb{E}_{\tau_i}^Q \left[
R_i +
D(\tau_i,\tau_{i+1})V_{\tau_{i+1}}
\right]. $$

Important interpretation:

> **Fixing dates modify the information state; payment dates generate TD
> rewards.**

If a coupon has already fixed but has not yet been paid, it remains part
of the MtM.

Most CCR transitions may therefore have

$$ R_i=0, $$

while transitions containing payment dates have non-zero rewards.

This naturally produces economically meaningful changes in MtM around
payment dates.

------------------------------------------------------------------------

## Monte Carlo Targets

A conventional Monte Carlo regression target at time $t$ is

$$ Y_t^{MC} = \sum_{u>t}D(t,u)C_u. $$

The neural network approximates

$$ \hat V_t \approx
\mathbb{E}_t^Q[Y_t^{MC}]. $$

This serves as the main benchmark against TD learning.

------------------------------------------------------------------------

## Temporal-Difference Learning

The one-step TD/Bellman target is

$$ Y_t^{TD} = R_{t,t+\Delta} +
D(t,t+\Delta)\hat V_{t+\Delta}. $$

Thus,

$$ \hat V_t \approx \mathbb{E}_t^Q \left[
R_{t,t+\Delta}
+
D(t,t+\Delta)\hat V_{t+\Delta}
\right]. $$

Intermediate cash flows are **not mathematically required** for TD
learning. However, the proposed synthetic derivative deliberately
contains intermediate annual cash flows so that the experiment has
meaningful intermediate rewards and realistic value discontinuities.

Important research questions include:

-   Bias/variance trade-off between MC and TD targets.
-   Sample efficiency.
-   Stability of bootstrapping.
-   Error propagation backward through time.
-   Accuracy of exposure distributions and tails.
-   Interaction between TD learning and differential supervision.

------------------------------------------------------------------------

# Synthetic Path-Dependent Derivative

## Horizon and Observation Structure

Use a derivative with approximately:

-   Maturity: 5 years.
-   Monthly underlying observations: 12 per year.
-   Total underlying observations: 60.
-   Annual contractual payment dates.
-   CCR observation dates may be different from fixing/payment dates.

The underlying may be a basket.

For $d$ assets, define a geometric basket such as

$$ B_t = \left( \prod_{j=1}^{d}S_t^j
\right)^{1/d}. $$

------------------------------------------------------------------------

## Annual Path Statistic

For year $k$, define a geometric average/performance statistic using the
12 monthly observations in that year, for example

$$ G_k^{yr} = \left( \prod_{m=12(k-1)+1}^{12k}
\frac{B_{t_m}}{B_{T_{k-1}}} \right)^{1/12}. $$

The exact normalization may be refined later.

------------------------------------------------------------------------

## Annual Cash Flows

Use annual call-like cash flows for all five years:

$$ C_k^{YoY} = \alpha \left( G_k^{yr}-1
\right)^+, \qquad k=1,\ldots,5. $$

The parameter $\alpha$ is a contractual scaling/notional
coefficient, not a learned parameter.

A simple initial value can be chosen, or $\alpha$ can be
calibrated so that the initial PV of the annual component is comparable
to that of the terminal component.

------------------------------------------------------------------------

## Full-History Terminal Statistic

Define a statistic depending on all 60 observations:

$$ A^H = \left( \prod_{m=1}^{60}
\frac{B_{t_m}}{B_0} \right)^{1/60}. $$

The terminal payoff should add long-memory dependence and allow both
positive and negative MtM regions.

A simple candidate is a risk-reversal-like payoff:

$$ C_5^H = \beta \left[
(A^H-K_C)^+
-
(K_P-A^H)^+
\right]. $$

A simple initial specification is

$$ K_P=0.9, \qquad K_C=1.0, \qquad \beta=1. $$

The total year-five cash flow is

$$ C_5=C_5^{YoY}+C_5^H. $$

This gives:

-   Local/yearly path dependence through annual averages.
-   Long-memory dependence through the full-history statistic.
-   Positive and negative value regions.
-   A payoff that remains relatively simple and suitable for AAD/DML.

Avoid barriers, extrema, and unnecessary discontinuities in the initial
experiments because they complicate differential labels and can obscure
the methodological comparison.

------------------------------------------------------------------------

# Markovianity and Sequence Models

An important conceptual point is that the geometric-average product can
be made Markovian by augmenting the current state with the appropriate
running statistic.

If $A_t$ is a sufficient running geometric-average state,

$$ V_t = V(t,X_t,A_t), $$

where $X_t$ contains the current market risk factors.

Therefore an RNN or Transformer is **not mathematically necessary** for
this particular synthetic product.

This should be acknowledged explicitly.

## Engineered-State Benchmark

Use a feed-forward neural network:

$$ \hat V_t = f_\theta(X_t,A_t,t). $$

Because the sufficient statistic is known, this is an important
information-efficient benchmark.

## Learned Sequential Representation

For LSTM/Transformer models, the main experiment can instead supply the
raw sequence

$$ X_0,\ldots,X_t $$

without explicitly providing the sufficient geometric-average statistic.

The question then becomes:

> Can a sequence architecture learn the relevant path representation
> directly from the raw history, and at what sample/computational cost
> relative to explicit Markov state engineering?

This makes the simple geometric-average product scientifically useful
because the true sufficient statistic is known.

A useful paper statement is:

> *The geometric-average contract is intentionally chosen because its
> path dependence admits a known low-dimensional Markovian
> representation. This provides a controlled benchmark against which the
> ability of recurrent and attention-based architectures to learn
> path-dependent representations directly from historical risk-factor
> sequences can be assessed.*

For more complex products/portfolios, manual construction of sufficient
state variables can become cumbersome, which provides the practical
motivation for generic sequence models.

------------------------------------------------------------------------

# Neural Architectures

The main architecture comparison should remain focused.

Recommended principal architectures:

1.  **FFNN + engineered sufficient state**
2.  **LSTM**
3.  **Causal Transformer**

GRU or vanilla RNN can be secondary robustness checks if needed, but
avoid turning the paper into an architecture catalogue.

------------------------------------------------------------------------

## Many-to-Many Learning

The sequence models should produce values at every relevant CCR date:

$$ (X_0,\ldots,X_T) \longrightarrow
(\hat V_0,\ldots,\hat V_T). $$

This is a many-to-many problem rather than sequence-to-one.

------------------------------------------------------------------------

## Causal Transformer

The Transformer must be causal.

The output at time $t$ may attend only to

$$ X_0,\ldots,X_t $$

and never to future states.

Use a triangular causal attention mask.

Sequence length is only around 60, so quadratic attention complexity is
not an important limitation.

------------------------------------------------------------------------

# Positional Encoding for the Transformer

A Transformer has no intrinsic ordering, so positional information must
be introduced.

A standard sinusoidal encoding is

$$ PE(t,2i) = \sin \left(
\frac{t}{10000^{2i/d_{\text{model}}}} \right), $$

$$ PE(t,2i+1) = \cos \left(
\frac{t}{10000^{2i/d_{\text{model}}}} \right). $$

Alternatively, learned positional embeddings can be used.

Because the maximum sequence length is only about 60, learned embeddings
are inexpensive and are a reasonable main specification, with sinusoidal
encoding as a robustness check.

Distinguish **sequence position** from **economic time**.

The Transformer representation can be

$$ z_t = W_xx_t+p_t, $$

where $p_t$ is positional encoding.

In addition, explicitly provide economic time variables such as

$$ t/T $$

and/or

$$ (T-t)/T. $$

Positional encoding tells the Transformer **where a token occurs in the
sequence**; time-to-maturity tells it **where the derivative is in its
economic life**.

------------------------------------------------------------------------

# Differential Machine Learning

The reference concept is Differential Machine Learning in the sense of
Huge & Savine.

Standard DML augments value labels with derivatives of the target with
respect to model inputs.

For path-dependent sequences, full historical differential supervision
could involve

$$ \left\{ \frac{\partial Y_t}
{\partial X_{t_j}} \right\}_{j\leq t}, $$

which scales approximately as $O(Td)$ and can become impractical.

------------------------------------------------------------------------

## Proposed Local / Current-State Differential Labels

The proposed approach is to use only derivatives with respect to the
**current market state**, holding realized history fixed:

$$ \Delta_t^{loc} = \left. \frac{\partial Y_t}
{\partial X_t} \right|_{\text{history fixed}}.
$$

Possible terminology:

-   current-state differential labels;
-   local differential labels;
-   conditional local differentials.

Avoid describing these as approximations of the complete historical
gradient. They deliberately represent a different, economically
meaningful sensitivity.

The differential target dimension is approximately

$$ O(d) $$

rather than

$$ O(Td). $$

This is potentially one of the strongest methodological contributions.

------------------------------------------------------------------------

## Differential Loss

A simple joint objective is

$$ \mathcal{L} = \mathcal{L}_V +
\lambda_{\Delta}\mathcal{L}_{\Delta}, $$

with

$$ \mathcal{L}_V = \left| \hat V_t-Y_t
\right|^2 $$

and

$$ \mathcal{L}_{\Delta} = \left|
\nabla_{X_t}\hat V_t -
\widehat{\Delta}_t^{loc} \right|^2. $$

The sequence model must **not detach or ignore historical information**
merely because differential labels are only supplied for the current
state.

History should continue to influence the learned hidden representation.

The derivative loss only constrains

$$ \frac{\partial\hat V_t}{\partial X_t}. $$

------------------------------------------------------------------------

## Economic Interpretation

For CCR, the local derivative has a natural interpretation:

> The sensitivity of today's MtM to today's market state, conditional on
> the history already realized.

Retroactively perturbing historical fixings is less directly relevant to
many operational risk-management questions.

A central empirical question is:

> **How much of the sample-efficiency benefit of Differential Machine
> Learning survives when only $O(d)$ current-state sensitivities are
> used instead of $O(Td)$ full-path sensitivities?**

If computationally feasible, full-path differentials can be included as
an ablation/upper benchmark.

------------------------------------------------------------------------

# AAD

Automatic Adjoint Differentiation / reverse-mode AD should be used where
appropriate to generate differential labels efficiently.

Keep conceptually separate:

1.  Established AAD methodology.
2.  Standard Differential Machine Learning.
3.  The proposed use of low-dimensional current-state differential
    labels in a path-dependent sequence-learning setting.

Do not present the latter as standard DML without qualification.

------------------------------------------------------------------------

# Experimental Design

A clean principal grid is

$$ { \text{FFNN+state}, \text{LSTM},
\text{causal Transformer} } \times { \text{MC},
\text{TD} } \times { \text{value-only},
\text{value+local differential} }. $$

This gives 12 principal configurations.

Analyze these deeply rather than adding many additional architectures.

------------------------------------------------------------------------

## Fair Architecture Comparison

Use:

-   Same underlying simulated paths.
-   Same train/validation/test partitions.
-   Same information set, except where explicitly testing engineered
    state versus learned history.
-   Same dates.
-   Same target definitions.
-   Same normalization.
-   Same loss definitions.
-   Same differential weighting conventions.
-   Approximately matched parameter counts, e.g. within 10--15%.
-   Same optimizer family where sensible.
-   Same early-stopping criteria.
-   Same random seeds.
-   Same hyperparameter-search budget.

Do not force identical hidden dimensions merely to appear fair.

Do not necessarily force identical learning rates if architectures
require different optimal values; instead give each model the same
tuning opportunity.

Report both controlled-capacity and, if useful, best-tuned comparisons.

------------------------------------------------------------------------

# Evaluation

Do not restrict evaluation to pricing RMSE.

Important metrics include:

## Valuation Accuracy

-   RMSE / MAE of $V_t$.
-   Error by time-to-maturity.
-   Error around fixing and payment dates.

## Exposure Accuracy

-   EE.
-   PFE at relevant quantiles.
-   EPE.
-   Exposure distributions.
-   Tail errors.
-   CVA impact if included.

## Differential Accuracy

-   Error in current-state sensitivities.
-   Stability of sensitivities through time.
-   Hedging/risk interpretation if appropriate.

## Sample Efficiency

A particularly important result should be accuracy as a function of the
number of training paths:

$$ \text{error} = f(N_{\text{training paths}}). $$

Demonstrate whether TD and/or local differential supervision achieve a
target accuracy with materially fewer simulations.

## Computational Efficiency

Report:

-   Simulation cost.
-   Label-generation cost.
-   Training cost.
-   Inference cost.
-   Differential-label cost.
-   Total computational budget.

This is important because a method requiring fewer paths but
substantially more expensive labels may not be superior overall.

------------------------------------------------------------------------

# Aging Product and Rolling Warm-Start Training

A practical extension is to allow calendar time to pass and update the
model as the derivative ages.

Let the trained parameters at month $t$ be

$$ \theta_t^*. $$

At month $t+1$, initialize with

$$ \theta_{t+1}^{(0)} = \theta_t^* $$

and fine-tune using a relatively small new simulation set.

The preferred terminology is:

> **rolling warm-start retraining**

rather than generic "transfer learning."

The economic motivation is that the valuation function should usually
evolve smoothly between adjacent dates, although realized fixings,
calibration changes, and reduced maturity alter the problem.

------------------------------------------------------------------------

## Fixed Architecture

Do not resize the network every month.

Keep a model capable of handling the maximum sequence length and use:

-   masking;
-   padding;
-   variable-length sequences.

This preserves the parameter space and makes warm-starting
straightforward.

For an aging contract:

-   realized historical information increases;
-   remaining future horizon decreases.

These should not be confused.

------------------------------------------------------------------------

## Rolling Experiments

Compare at least:

1.  Full retraining from scratch.
2.  Warm start using the full new training set.
3.  Warm start using a small update set.

Potential additional ablation:

-   retain all previous parameters;
-   retain sequence-representation layers but reinitialize the final
    pricing head.

Naive warm-starting should not be assumed to work automatically; it must
be tested for generalization.

A potentially strong practical hypothesis is:

> Local differential supervision reduces the amount of new simulation
> required to update an aging CCR surrogate.

This could provide an interesting interaction between:

$$ \text{aging} + \text{warm start} +
\text{TD} + \text{local differential labels}. $$

Treat this initially as an important practical extension rather than
necessarily the central contribution.

------------------------------------------------------------------------

# Novelty Positioning

Do not claim that any of the following alone is novel:

-   Neural networks for CCR.
-   DML.
-   TD learning for derivative pricing.
-   LSTM/Transformer pricing.
-   AAD.

Relevant prior work includes:

-   Huge & Savine on Differential Machine Learning.
-   Neural approaches to CVA/CCR.
-   Deep BSDE / xVA methods.
-   Prior TD methods for derivative pricing.
-   Existing applications of DML to credit exposure.

The potentially distinctive intersection is:

$$ \boxed{
\text{path-dependent CCR}
+
\text{sequence representation}
+
\text{TD}
+
\text{low-dimensional local DML}
} $$

with sample/computational efficiency as an important empirical
contribution.

Before making definitive novelty claims, perform a systematic and
current literature review.

------------------------------------------------------------------------

# Possible Paper Titles

Current preferred title:

> **Learning Path-Dependent Counterparty Exposures: Temporal-Difference
> and Differential Methods**

Other candidates:

> **Deep Learning for Path-Dependent Counterparty Exposures: From Markov
> States to Sequential Representations**

> **Efficient Learning of Path-Dependent Counterparty Exposures with
> Temporal and Differential Information**

> **Learning Dynamic Counterparty Exposures from Temporal and
> Differential Information**

If the local-differential contribution becomes central:

> **Efficient Learning of Path-Dependent Counterparty Exposures with
> Temporal-Difference and Local Differential Information**

Prefer **counterparty exposures** to **counterparty credit risk** in the
title if the paper primarily models MtM/exposure profiles rather than
default/CVA itself.

------------------------------------------------------------------------

# Relationship to Previous Research

The researcher's previous paper studies deep learning for the FRTB-IMA
P&L Attribution Test (PLAT), including neural approximation,
distributional accuracy, tail metrics, control-variate ideas, and hedged
versus unhedged portfolios.

The new CCR paper should be clearly distinct but can share a broader
thesis theme:

> **Machine-learning methods for computationally demanding financial
> risk measurement and valuation.**

The first paper concerns market risk/FRTB.

The second paper concerns dynamic/path-dependent CCR exposure.

------------------------------------------------------------------------

# Proposed PhD Thesis Structure

A coherent thesis structure is:

1.  Introduction.
2.  Market risk: metrics and regulation.
3.  Counterparty credit risk: metrics and regulation.
4.  Automatic Adjoint Differentiation.
5.  Neural networks and the architectures used in the research.
6.  Research Paper 1: FRTB-IMA PLAT.
7.  Research Paper 2: Path-dependent CCR exposure learning.
8.  Conclusions and future research.

An alternative pedagogical ordering is to introduce neural
networks/backpropagation before AAD, but either structure can be
defended.

TD learning and the current-state differential proposal belong primarily
in the second research chapter rather than being over-developed in
generic background chapters.

------------------------------------------------------------------------

# Guidance for ChatGPT Within This Project

When assisting with this project:

-   Assume an advanced quantitative-finance and machine-learning
    audience.
-   Do not over-explain elementary ML, stochastic calculus, or CCR
    concepts unless requested.
-   Use mathematical notation when it improves precision.
-   Critically challenge ideas rather than automatically agreeing.
-   Distinguish clearly between:
    -   mathematical necessity;
    -   computational convenience;
    -   empirical hypothesis;
    -   methodological novelty;
    -   application novelty.
-   Identify likely reviewer objections.
-   Prefer controlled experiments and ablations over adding unnecessary
    model complexity.
-   Preserve the distinction between $P$-measure exposure simulation and
    $Q$-measure conditional valuation.
-   Be precise about ex-dividend versus cum-dividend valuation around
    payment dates.
-   Treat fixing dates as information-state events and payment dates as
    cash-flow/reward events.
-   Do not imply that sequence models are required when a known
    sufficient Markov state exists.
-   Use the engineered-state FFNN as a serious benchmark, not a straw
    man.
-   Treat the simple path-dependent derivative as a controlled
    laboratory in which the sufficient statistic is known.
-   Focus strongly on sample efficiency, distribution/tail accuracy, and
    total computational cost.
-   When discussing literature or novelty, search current academic
    sources before making definitive claims.
-   When proposing paper text, aim for language appropriate for a Q1
    quantitative finance / machine-learning journal.

------------------------------------------------------------------------

# Open Research Decisions

The following remain deliberately open and should be refined through
experiments:

-   Exact normalization of annual geometric averages.
-   Calibration/choice of $\alpha$ and $\beta$.
-   Exact strikes of the terminal risk-reversal component.
-   Underlying stochastic dynamics and number of risk factors.
-   Correlation structure.
-   Whether stochastic volatility/rates are introduced.
-   Exact CCR observation grid.
-   Exact fixing/payment lag.
-   LSTM versus GRU as the recurrent main model.
-   Learned versus sinusoidal Transformer positional encoding.
-   TD horizon: one-step versus possible multi-step variants.
-   Differential-loss weighting $\lambda_\Delta$.
-   Whether full-path differentials are computationally feasible as an
    ablation.
-   Whether CVA itself is included or exposure metrics remain the
    primary outputs.
-   Size of rolling warm-start update samples.
-   Whether rolling warm-start results belong in the main contribution
    or an extension.

These decisions should be driven by the scientific question and clean
experimental design rather than by adding complexity for its own sake.
