# L1: Introduction to Generative Modeling — Study Notes

CSE-5025 Generative AI · Source: `Slide/l1.pdf` (45 slides)
Topics: generative vs. discriminative learning, probability toolkit, latent-variable models, the map of generative model families, evaluation metrics.

> **How to read these notes.** Each topic has the core statement first (what the slide says), then **Why it matters** (where it shows up later in the course), worked numbers, and ⚠️ common pitfalls. Sections marked *(beyond slides)* add background that helps with understanding but goes past what was shown in lecture.

---

## 0. Learning objectives (what you should be able to do)

1. Define a generative model and contrast it with a discriminative one.
2. Explain what modelling p(x) enables: **sample, score, fill in, represent, condition**.
3. Write down a latent-variable model and name its **prior, likelihood, posterior**.
4. Show that **maximum likelihood = minimising KL(p_data ‖ p_θ)**.
5. Place autoregressive models, flows, VAEs, GANs and diffusion in one taxonomy.

### Course roadmap (context)
| Block | Lectures | Content |
|---|---|---|
| Foundations | L1–3 | latent variables, EM (GMM/HMM), variational inference |
| Deep generative models | L4–9 | AEs/VAEs, GANs, diffusion |
| Language & grounding | L10–12 | Transformers, fine-tuning/alignment, RAG |
| Evaluation, ethics, multimodal | L13–14 | |

How L1 feeds the rest: the probability toolkit (§1) is used in every lecture; latent-variable models (§3) are the subject of L2–5; the ML ⇔ KL result (§4) is the objective behind almost every model; the taxonomy (§5) is the map for L4–11; evaluation (§6) returns in L13 and in A3 (FID).

### Notation
| Symbol | Meaning |
|---|---|
| x ∈ ℝᴰ | observed data point (bold = vector) |
| y | label, if one exists |
| z ∈ ℝᵈ | latent (hidden) variable, usually d ≪ D |
| p_data(x) | true, unknown data distribution |
| p_θ(x) | our model, parameters θ |
| q_φ(z\|x) | approximate posterior (encoder) |
| E_p[f(x)] | expectation of f under p |
| D_KL(p‖q) | KL divergence |

Conventions: "log" means natural log (nats) unless stated; log₂ gives bits. x⁽ˢ⁾ is the s-th sample, xₙ the n-th data point, xₜ the t-th element of a sequence.

---

## 1. Probability toolkit

Everything in this course is a statement about probability distributions, so this section is the vocabulary. You need to be fluent with the sum rule, product rule, Bayes, expectations and KL before L2.

### Discrete vs. continuous
- **PMF** (discrete): p(x) ≥ 0, Σ p(x) = 1. *Categorical*: p(z=k) = π_k, Σπ_k = 1 (die, word, cluster label).
- **PDF** (continuous): p(x) ≥ 0, ∫p(x)dx = 1. *Gaussian*: N(x; μ, σ²) = (1/√(2πσ²)) · exp(−(x−μ)²/(2σ²)).
- ⚠️ **A density can exceed 1**: N(0; 0, 0.1²) ≈ 3.99. Only *areas* under a density are probabilities.

**Explanation.** A PMF value *is* a probability: p(word = "ভাত") = 0.002 means 0.2% of the time. A PDF value is a probability **per unit of x**: P(a ≤ x ≤ b) = ∫ₐᵇ p(x)dx, and P(x = exactly 1.0) = 0 for any continuous x. A narrow Gaussian (σ = 0.1) squeezes all its area into a width of about 0.4, so its height must be large (≈ 4) for the area to equal 1. Consequence for later: a **log-density can be positive**, so "negative log-likelihood" for continuous data can be negative. This is not a bug.

**Why it matters:** the categorical is the output of every language model (softmax over the vocabulary); the Gaussian is the prior and noise model in VAEs and diffusion.

### The two rules
- **Sum rule (marginalise):** p(x) = Σ_y p(x,y)  (or ∫ p(x,y) dy)
- **Product rule (factorise):** p(x,y) = p(x|y)p(y) = p(y|x)p(x)
- Equate the two factorisations → **Bayes' rule**.
- **Independence:** p(x,y) = p(x)p(y).

**Explanation.**
- *Marginalising* = "I don't care about y, so add up over all its values." It removes a variable.
- *Conditioning* = "I know y, so restrict to the slice where y holds and renormalise so it sums to 1": p(x|y) = p(x,y)/p(y).
- *Bayes*, derived in one line: p(x|y)p(y) = p(y|x)p(x) ⇒ p(y|x) = p(x|y)p(y)/p(x). It **reverses the direction** of a conditional, and that reversal is how we go from "how the data are generated" to "what caused this data point".
- *Conditional independence* *(beyond slides, needed for L2)*: x ⊥ y | z means p(x,y|z) = p(x|z)p(y|z). Naive Bayes assumes features are independent given the class; an HMM assumes each observation depends only on the current hidden state. These assumptions are what make the models tractable.

**Worked example (Dhaka monsoon):**

| | Umbrella | None | p(w) |
|---|---|---|---|
| Rain | 0.35 | 0.05 | 0.40 |
| Dry | 0.12 | 0.48 | 0.60 |
| p(u) | 0.47 | 0.53 | 1 |

- Marginal: p(Rain) = 0.35 + 0.05 = 0.40 (sum along the row).
- Conditional: p(Rain | Umbrella) = 0.35 / 0.47 ≈ **0.74** (take the Umbrella column, renormalise).
- Reverse direction: p(Umbrella | Rain) = 0.35 / 0.40 = **0.875**. Note 0.74 ≠ 0.875: p(a|b) and p(b|a) are different quantities. Mixing them up is the most common Bayes mistake.
- Independence check: p(Rain)·p(Umbrella) = 0.40 × 0.47 = 0.188 ≠ 0.35 ⇒ weather and umbrella are **dependent** (as expected).

### Chain rule
p(x₁,…,x_T) = ∏ₜ p(xₜ | x₍<t₎)
- Exact for **any** distribution. This is how GPT factorises a sentence: one token at a time.

**Explanation.** Apply the product rule repeatedly: p(x₁,x₂,x₃) = p(x₁) · p(x₂|x₁) · p(x₃|x₁,x₂). No assumption is made, so nothing is lost. The difficulty is that the last factor conditions on everything before it. A lookup table for p(xₜ | x₍<t₎) with vocabulary V needs Vᵗ⁻¹ rows, which is impossible for real vocabularies. Neural autoregressive models solve this by using **one shared network** that reads the prefix and outputs a distribution over the next token.

- Example: p("আমি ভাত খাই") = p(আমি) · p(ভাত | আমি) · p(খাই | আমি, ভাত).
- The order is a choice (left-to-right is natural for text; images use raster order in PixelCNN).
- Markov assumptions (HMMs, n-gram models) *truncate* the history: p(xₜ | x₍<t₎) ≈ p(xₜ | xₜ₋₁). That makes the model cheaper but less exact.

### Expectation & Monte Carlo
E_p[f(x)] = ∫ f(x)p(x)dx ≈ (1/S) Σₛ f(x⁽ˢ⁾), x⁽ˢ⁾ ~ p
- If we can **sample**, we can estimate averages without solving the integral. VAE and diffusion training rely on this.

**Explanation.**
- The estimator is **unbiased** (its average is the true expectation), and its standard error is σ_f/√S. So 4× more samples halve the error, and the rate **does not depend on the dimension of x**. That is why Monte Carlo works in high-dimensional spaces where grids of points fail.
- Example: estimate E[x²] for x ~ N(0,1) (true answer 1). Draw samples, square them, average; with S = 10,000 the estimate is typically within ±0.015 of 1.
- In this course: the VAE's reconstruction term E_q[log p(x|z)] is estimated with **one** sample of z per data point; diffusion training samples a random timestep t and random noise ε per example. Both are single-sample Monte Carlo estimates of an expectation inside the loss.
- ⚠️ Monte Carlo has high variance when f is large only in a region where p puts very little mass. That is exactly the problem with estimating p(x) = E_p(z)[p(x|z)] directly (see §3, "central obstacle").

### KL divergence
D_KL(p‖q) = E_p[log p(x)/q(x)] = Σ p(x) log(p(x)/q(x))
- **≥ 0**, **= 0 iff p = q**, **not symmetric** → not a true distance.

**Explanation.**
- Read it as: "on average, under samples from p, how much more likely are they under p than under q?" It measures how badly q stands in for p.
- Link to information theory *(beyond slides)*: D_KL(p‖q) = H(p,q) − H(p), where H(p) = −Σ p log p is the **entropy** and H(p,q) = −Σ p log q is the **cross-entropy**. It is the extra cost (in nats or bits) of encoding samples from p using a code designed for q. The cross-entropy loss you use to train classifiers and LLMs is H(p,q), and minimising it over q is the same as minimising the KL because H(p) is fixed.
- Why ≥ 0: by Jensen's inequality, −D_KL = E_p[log q/p] ≤ log E_p[q/p] = log Σ q = log 1 = 0.
- ⚠️ If q(x) = 0 anywhere that p(x) > 0, then D_KL(p‖q) = ∞. Keep this in mind for §4 (mode covering).

**Worked example:** p = (0.5, 0.5) fair coin, q = (0.9, 0.1) biased.
- D_KL(p‖q) = 0.5 ln(0.5/0.9) + 0.5 ln(0.5/0.1) = −0.294 + 0.805 ≈ **0.511** nats
- D_KL(q‖p) = 0.9 ln(0.9/0.5) + 0.1 ln(0.1/0.5) = 0.529 − 0.161 ≈ **0.368** nats
- Check through entropy: H(p) = ln 2 ≈ 0.693; H(p,q) = −0.5 ln 0.9 − 0.5 ln 0.1 ≈ 1.204; difference ≈ 0.511 ✓.
- Why they differ: under p, tails happen half the time, and q gives tails only 0.1, so p‖q is penalised heavily. Under q, tails are rare, so that same mismatch is weighted only by 0.1.
- Order matters → shapes model behaviour (see §4).

### Likelihood (coin example)
10 flips: 7 H, 3 T. Model p(H) = θ.
- L(θ) = θ⁷(1−θ)³, log L = 7 log θ + 3 log(1−θ)
- d/dθ: 7/θ − 3/(1−θ) = 0 → **θ̂ = 0.7**
- **Probability:** fix θ, vary data. **Likelihood:** fix data, vary θ.

**Explanation.**
- The likelihood is *not* a distribution over θ. It doesn't integrate to 1 over θ. It is a score: "how well does this θ explain the data we saw?"
- **Why take logs:** (1) a product of N small numbers underflows to 0 in floating point; a sum of logs does not; (2) sums are easier to differentiate; (3) log is monotone, so the arg max does not change.
- Numbers: log L(0.7) = −6.109 vs. log L(0.5) = −6.931. θ = 0.7 explains the data better, by a factor e^0.82 ≈ 2.3.
- Check it is a maximum: d²/dθ² = −7/θ² − 3/(1−θ)² < 0 ✓. In general, θ̂ = h/n.
- ⚠️ MLE overfits small data *(beyond slides)*: 3 heads in 3 flips gives θ̂ = 1, so the model says tails are impossible (probability 0, log-probability −∞). Remedies: a prior (MAP estimation) or smoothing (e.g. (h+1)/(n+2)). The same issue in language models is why unseen words need nonzero probability.

### MLE for a Gaussian
log L(μ,σ²) = −(N/2) log(2πσ²) − (1/(2σ²)) Σ (xₙ − μ)²
- **μ̂ = (1/N) Σ xₙ**,  **σ̂² = (1/N) Σ (xₙ − μ̂)²**  (note: 1/N, not 1/(N−1))
- Example: data {2, 4, 6} → μ̂ = 4, σ̂² = (4+0+4)/3 = 8/3 ≈ **2.67**

**Derivation.**
- ∂/∂μ: (1/σ²) Σ (xₙ − μ) = 0 ⇒ μ̂ = sample mean.
- ∂/∂σ²: −N/(2σ²) + (1/(2σ⁴)) Σ (xₙ − μ)² = 0 ⇒ σ̂² = mean squared deviation.
- Note that maximising the μ-part means **minimising squared error**. "Gaussian likelihood ⇔ MSE loss" is why the VAE decoder and the diffusion noise-prediction loss use MSE.
- The 1/N estimator is **biased**: E[σ̂²] = ((N−1)/N) σ². It is still the MLE; the unbiased 1/(N−1) version is not the maximiser of the likelihood.

> **The pattern for the whole course:** write log p_θ(data), then maximise it — in closed form (here), by **EM** for mixtures (L2), by **gradient descent** for neural nets.

---

## 2. What is a generative model?

**Definition:** a generative model learns the probability distribution the data came from, so it can produce new data like it.

x₁,…,x_N ~ p_data(x)  ⟹ (learning)  p_θ(x) ≈ p_data(x)  ⟹ (generation)  x_new ~ p_θ(x)

- We **never see p_data**, only samples from it.
- p_θ is a parametric family; learning chooses θ.

**Explanation.** Three ingredients are always present: (1) a **model family** {p_θ} (a Gaussian, a mixture, a neural network that outputs a distribution); (2) a **training objective** that measures how close p_θ is to p_data using only the samples (usually likelihood); (3) a **sampling procedure** to draw x_new. Every model family in §5 is a different choice of these three. "Like it" means *statistically* similar, not copies. A model that memorises and replays training data has failed at generalisation even if its samples look perfect.

### Discriminative (x → y)
- Models **p(y|x)** directly, or just a decision rule.
- Examples: logistic/softmax regression, NN classifier → p(y|x); SVM → boundary; decision tree / random forest → rules.
- **Limitation:** tells *which label* fits x, but **cannot produce a new x**.

Explanation: a discriminative model spends all its capacity on the question it is asked. It never needs to know what typical inputs look like, only which side of the boundary they fall on. That is efficient for classification, but it means the model has no notion of whether an input is plausible at all. It will confidently label random noise as a "cat".

### Generative
- **With labels:** learn the joint p(x,y) = p(x|y)p(y). p(y) = class frequency, p(x|y) = what class-y data look like. Generate y → x (e.g. x ~ p(x|y=Cat)).
- **Without labels:** learn p(x) — an *unconditional* generative model. **Most of the course lives here.**

Explanation: the joint is the "complete story" of the data. Once you have it, everything else is computable: p(y) and p(x) by marginalising, p(y|x) by Bayes, new (x, y) pairs by ancestral sampling (first y, then x). The price is that modelling x is much harder than modelling y. x is high-dimensional (an image) while y is one number.

### Side by side (memorise this table)
| | Discriminative | Generative |
|---|---|---|
| Learns | p(y\|x) or rule x↦y | p(x) or p(x,y) |
| Direction | x → y | y → x (or z → x) |
| Typical task | "Is this a cat or dog?" | "Generate a new cat image" |
| Produces new x? | No | Yes |
| Uses unlabelled data? | Not directly | Yes, naturally |
| Examples | Logistic reg., SVM, CNN | GMM, Naive Bayes, VAE, GPT |

- **Conditional generation:** model p(x|c) for a label/prompt c (text-to-image, translation, class-conditional sampling).
- One line: **discriminative = predict; generative = model and generate.**

⚠️ Conditional generation p(x|c) is still *generative*, even though it has an input c. The test is what gets modelled: a distribution over the high-dimensional x (generative) or over the label y (discriminative). A translation model p(Bangla sentence | English sentence) is generative; a sentiment classifier p(positive | review) is discriminative.

### Bayes' rule: the bridge
p(y|x) = p(x|y) p(y) / p(x),  p(x) = Σ_y p(x|y)p(y)
(posterior = likelihood × prior / evidence)
- A generative model can **also classify**.
  - Generative classifier: fit p(x|y), p(y), apply Bayes → **Naive Bayes, Gaussian Discriminant Analysis**.
  - Discriminative classifier: fit p(y|x) directly → **logistic regression**.
- **Ng & Jordan (2001):** generative classifiers may reach their best error with **less data**; discriminative classifiers often reach a **lower final error**.

**Explanation of Ng & Jordan.** Naive Bayes makes a strong assumption (features independent given class). With little data, that assumption acts like regularisation: few parameters, low variance, so it learns fast. But the assumption is usually false, so with lots of data its error plateaus (high bias). Logistic regression makes fewer assumptions, needs more data to pin down its parameters, but eventually gets closer to the true boundary. Practical reading: in low-resource settings (e.g. a small labelled Bangla dataset), a generative classifier can be a strong baseline.

Also note that p(x) in the denominator is the same for every class, so for **choosing** a class you only need to compare p(x|y)p(y). You need p(x) only if you want calibrated probabilities.

**Worked example:**
| Class | p(y) | p(x\|y) | p(x\|y)p(y) |
|---|---|---|---|
| Cat | 0.6 | 0.020 | 0.012 |
| Dog | 0.4 | 0.005 | 0.002 |
| Evidence p(x) | | | 0.014 |

p(Cat|x) = 0.012/0.014 ≈ **0.857**, p(Dog|x) ≈ 0.143 → predict **Cat**. The same model can also *sample* a new cat from p(x|Cat).

A bonus of having p(x): if p(x) = 0.014 is far lower than for typical training images, the input is probably neither a cat nor a dog. A discriminative classifier would still force it into one of the two classes.

### Same data, two views
- Discriminative asks "given x, which label?" — only the **boundary** matters.
- Generative asks "how are x and y generated?" — each class gets a full **density**; can sample new points, flag outliers; the boundary follows.

Picture two 2-D point clouds. Logistic regression draws a line between them and ignores how the clouds are shaped. A generative classifier (e.g. GDA) fits an ellipse-shaped Gaussian to each cloud; the boundary is wherever the two weighted densities are equal (a straight line if both Gaussians share a covariance, a curve otherwise).

### Why model p(x)? Five capabilities
| Capability | Operation | Example |
|---|---|---|
| **Sample** | draw x ~ p_θ(x) | image/text/audio generation |
| **Score** | evaluate p_θ(x) | anomaly detection |
| **Fill in** | p_θ(x_miss \| x_obs) | inpainting, imputation |
| **Represent** | infer latent x → z | unsupervised features |
| **Condition** | model p_θ(x\|c) | "cat" → cat image |

What each one means in practice:
- **Sample:** generate new data, including synthetic training data for low-resource tasks (e.g. augmenting a small Bangla handwriting dataset).
- **Score:** a low p_θ(x) flags an unusual input: fraud, a faulty sensor reading, an out-of-distribution test image. ⚠️ Deep models sometimes give *higher* likelihood to out-of-distribution data (a known failure), so scoring needs care.
- **Fill in:** condition on the observed part and generate the rest: image inpainting, completing a sentence, imputing missing values in survey or health records.
- **Represent:** the latent z of a data point is a compressed feature vector you can use for clustering, retrieval or a downstream classifier with few labels.
- **Condition:** controllable generation. Text-to-image, translation, speech synthesis and chatbots are all p(x|c).

No single model family is best at all five — this tension shapes every model in the course. (GANs sample well but cannot score; autoregressive models score exactly but sample slowly and have no z for representation; VAEs represent and sample fast but give only a bound for scoring.)

### The difficulty: data spaces are astronomically large
| Image | Values D | # possible images (256 levels = 2⁸ᴰ) | ≈ in decimal |
|---|---|---|---|
| 28×28 grey | 784 | 2^6,272 | 10^1,888 |
| 64×64 RGB | 12,288 | 2^98,304 | 10^29,592 |
| 224×224 RGB | 150,528 | 2^1,204,224 | 10^362,507 |

- Can't store a probability per image; almost all look like noise.
- Real images fill a tiny, structured region → p_θ must be a **compact parametric function**, not a lookup table.

For scale: there are about 10^80 atoms in the observable universe, and MNIST has 6 × 10⁴ training images. The model has to **generalise** from a vanishingly small sample to a sensible probability for every point of that space. It can only do that by exploiting structure (nearby pixels are correlated, objects have parts, etc.), and neural network architectures (convolutions, attention) build that structure in.

### Manifold hypothesis
Real high-dimensional data concentrate near a **low-dimensional manifold** in the ambient space.
- Analogy: a curled sheet of paper is 2-D inside 3-D space.
- A face x ∈ ℝ¹⁵⁰⁵²⁸ varies by a few factors (pose, lighting, expression, identity).
- So learn z ∈ ℝᵈ, d ≪ D: **x → z → x**.

Explanation: pick 150,528 pixel values uniformly at random and you get static, never a face. So faces occupy a negligible fraction of ℝᴰ. Moving smoothly along the manifold (turning the head slightly) changes the image smoothly, while moving off it (adding random noise) quickly produces non-faces. This motivates **latent-variable models**: describe each image by a few coordinates z on the manifold and learn a decoder that maps z back to pixels. Diffusion models use the same picture differently: noise pushes data off the manifold, and the network learns to push it back.

---

## 3. Latent-variable models

### Intuition
- x = what we observe (pixels); z = hidden factors that produced it, never labelled.
- Joint: **p(x,z) = p(x|z) p(z)**
- To score x, consider every explanation z: **p(x) = ∫ p(x|z) p(z) dz**
- Two directions: **Generation z → x (decoder)**; **Inference x → z (encoder)**.

Explanation: write down a *story* of how the data were made: "first nature picks hidden settings z, then produces x from them". The integral is the sum rule: since we don't know which z produced x, we average p(x|z) over all z, weighted by how plausible each z is a priori. A data point gets high p(x) if *some* plausible z explains it well.

Examples of z: the topic of a news article; the speaker in an audio clip; the digit identity and stroke thickness in MNIST; the hidden "regime" (bull/bear market) in an HMM for stock prices.

### Definition
p_θ(x) = ∫ p_θ(x|z) p(z) dz
| Part | Name | Notes |
|---|---|---|
| p(z) | **prior** | simple, e.g. N(0, I) or Cat(π) |
| p_θ(x\|z) | **likelihood / decoder** | how z produces x |
| p_θ(z\|x) | **posterior** | which z explains this x |
| p_θ(x) | **evidence / marginal likelihood** | |

- Graphical model: z → x, θ outside, plate over N; shaded node = observed.
- **Ancestral sampling:** draw z ~ p(z), then x ~ p_θ(x|z). *Simple parts, rich marginal.*

**"Simple parts, rich marginal" explained.** Each piece is easy: the prior is a standard Gaussian or a die roll, and p(x|z) is a Gaussian whose mean depends on z. But averaging infinitely many simple Gaussians, each centred at a different place, can produce a very complicated p(x): multimodal, curved, concentrated on a manifold. That is the whole trick behind GMMs, VAEs and (in a sense) diffusion.

Reading the graphical model: the arrow z → x encodes p(x|z); the plate (box labelled N) means "repeat for each data point, independently"; θ sits outside the plate because all data points share it; x is shaded because we observe it, and z is unshaded because we don't.

### Example 1 — Gaussian Mixture Model (discrete z)
- z ~ Cat(π₁,…,π_K); x | z=k ~ N(μ_k, Σ_k)
- **p(x) = Σₖ π_k N(x; μ_k, Σ_k)**
- z = which component made x (the cluster label nobody gave us). K simple Gaussians → multimodal density.
- Inference via Bayes: **p(z=k|x) = π_k N(x; μ_k, Σ_k) / p(x)** — these are the **responsibilities** computed in the **E-step of EM** (L2).
- Fitting uses **EM** (L2).

Parameters: θ = {π_k, μ_k, Σ_k}. π_k is the fraction of data each cluster accounts for, μ_k its centre, Σ_k its shape. Compared with k-means, a GMM gives **soft** assignments (a point can be 70% cluster A, 30% cluster B) and per-cluster shapes. k-means is the limit of a GMM with equal, shrinking spherical covariances.

**Ancestral sampling, one draw** (π = (0.3, 0.5, 0.2), μ = (−2, 1, 4), s.d. = (0.6, 1, 0.5)):
1. Roll 3-sided die z ~ Cat(0.3, 0.5, 0.2), say z = 2.
2. Draw x ~ N(1, 1²), say x = 1.4.
3. Keep x, **forget z**. Repeat → histogram traces p(x) with three bumps.

Step 3 is the key: the dataset we receive contains only the x's. Learning has to recover π, μ, σ *without* knowing which die roll produced each point, and that is why we need the posterior.

**Worked example: which component made x = 0?**
| k | π_k | N(0; μ_k, σ_k²) | π_k·N | p(z=k\|x=0) |
|---|---|---|---|---|
| 1 | 0.3 | 0.00257 | 0.000771 | 0.006 |
| 2 | 0.5 | 0.24197 | 0.120985 | **0.994** |
| 3 | 0.2 | 1.0×10⁻¹⁴ | 2.0×10⁻¹⁵ | 0.000 |
| Sum = p(x=0) | | | **0.121756** | 1 |

- x = 0 is 2 units (≈ 3.3 s.d.) from comp 1, 1 s.d. from comp 2, 8 s.d. from comp 3.
- Here the sum has only K = 3 terms; with continuous high-dimensional z it becomes an integral.

How to compute a row: N(0; −2, 0.6²) = (1/(0.6√(2π))) · exp(−(0−(−2))²/(2·0.36)) = 0.665 × exp(−5.56) ≈ 0.00257. Then multiply by π_k, add the column to get p(x), and divide each entry by that sum. Distance measured in **standard deviations**, not raw units, is what matters: comp 3 is closer than comp 1 in raw units only by a factor of 2, but its narrow σ makes x = 0 essentially impossible under it.

**Second example (the self-check drill), x = 4:**
| k | π_k·N(4; μ_k, σ_k²) | p(z=k\|x=4) |
|---|---|---|
| 1 | 3.9×10⁻²³ | 0.000 |
| 2 | 0.00222 | 0.014 |
| 3 | 0.15958 | **0.986** |
| Sum | 0.16179 | 1 |

### Example 2 — Probabilistic PCA (continuous z)
- z ~ N(0, I_d), d ≪ D;  x = Wz + μ + ε,  ε ~ N(0, σ² I_D)
- **p(x) = N(x; μ, WWᵀ + σ²I)**
- Linear-Gaussian ⇒ p(x) and p(z|x) are **exact** (closed form).
- Example: 2-D points along a line; a 1-D z captures the main variation.

**Explanation.**
- Where the covariance comes from: x is a linear function of Gaussians, so it is Gaussian, with mean E[x] = μ and Cov(x) = W·Cov(z)·Wᵀ + Cov(ε) = WWᵀ + σ²I.
- Interpretation: the columns of W are the d main directions of variation (the "manifold", here a flat d-dimensional plane); σ² is the noise around the plane.
- Parameter savings: a full Gaussian in D dimensions needs D(D+1)/2 covariance numbers (≈ 307,000 for D = 784). PPCA needs D·d + 1 (≈ 7,800 for d = 10). This is the manifold hypothesis turned into a model.
- The exact posterior *(beyond slides)*: p(z|x) = N(M⁻¹Wᵀ(x−μ), σ²M⁻¹) with M = WᵀW + σ²I. As σ² → 0, its mean becomes the standard PCA projection.
- PPCA is a **linear VAE**: the VAE (L4) keeps this structure but replaces Wz + μ with a neural network f_θ(z). That gains expressiveness but loses the closed-form posterior.

### Progression to the VAE (L4)
| Model | Latent | Meaning |
|---|---|---|
| GMM | discrete z | hidden cluster identity |
| PPCA | continuous z + **linear** map Wz | hidden coordinates |
| VAE | continuous z + **nonlinear** decoder f_θ(z) | learned features |

### The central obstacle: inference
p_θ(z|x) = p_θ(x|z)p(z) / ∫ p_θ(x|z′)p(z′)dz′
- Generation z → x is easy; inference x → z needs the **evidence** in the denominator.
- **Tractable:** discrete z with few values (finite sum: GMM, HMM); linear-Gaussian (PPCA).
- **Intractable:** nonlinear neural decoders, high-dimensional continuous z — no closed form.
- **VAE answer:** learn an encoder **q_φ(z|x) ≈ p_θ(z|x)** → variational inference & **ELBO** (L3), amortised inference (L4).

**Why exactly is it hard?**
- The numerator is easy: plug in z and evaluate. The denominator is the integral over *all* z.
- Grids fail: with d = 50 latent dimensions and only 10 grid points per dimension, you need 10⁵⁰ evaluations of a neural network.
- Naive Monte Carlo (sample z ~ p(z), average p(x|z)) fails too: for a specific image x, almost every random z decodes to something unrelated, so p(x|z) ≈ 0 for nearly all samples and the estimate is dominated by rare lucky draws (huge variance).
- Fix idea: sample z from a distribution that already concentrates on "z's that explain x". That distribution is q_φ(z|x). Learning it is the subject of L3–4.
- Why we care about the posterior at all: (1) **to learn** (EM needs it in the E-step; the ELBO needs q close to it); (2) **to represent** (the encoding of x *is* the posterior).

HMM note (for A1): in an HMM, z is a *sequence* of discrete states, so the sum has Kᵀ terms. That would be intractable, but the chain structure lets dynamic programming (forward–backward, Viterbi) compute it in O(T·K²). Tractability depends on **structure**, not only on the size of the latent space.

---

## 4. Learning: maximum likelihood = KL minimisation

max_θ ∏ₙ p_θ(xₙ) ⟺ max_θ (1/N) Σ log p_θ(xₙ) → (N→∞) max_θ E_p_data[log p_θ(x)]

D_KL(p_data ‖ p_θ) = E_p_data[log p_data(x)] − E_p_data[log p_θ(x)]
- First term is **constant in θ** (entropy of the data) → drops out of the arg min.
- ⇒ **θ* = argmin_θ D_KL(p_data ‖ p_θ)**. This is what "learning p(x)" means.
- Forward KL penalises missing any data mode → ML models tend to **cover all modes**, sometimes spreading mass too thinly.
- For latent-variable models p_θ(x) = ∫… is hard to evaluate ⇒ **ELBO**.

**The argument step by step.**
1. Data are i.i.d., so the likelihood of the dataset is a product: ∏ₙ p_θ(xₙ).
2. Take log and divide by N (neither changes the arg max): (1/N) Σ log p_θ(xₙ).
3. This is a Monte Carlo estimate (§1) of E_p_data[log p_θ(x)], and it becomes exact as N → ∞.
4. Expand the KL: D_KL(p_data‖p_θ) = E_p_data[log p_data] − E_p_data[log p_θ]. The first term is (minus) the entropy of the data. It is fixed by nature and does not depend on θ.
5. So minimising the KL over θ = maximising E_p_data[log p_θ] = maximum likelihood. ∎

**Why it matters.** It says that maximum likelihood is a principled choice, not only a convenient one: it is the best fit in a precise information-theoretic sense. It also tells you **which** KL you are minimising (the forward one), and that predicts the model's failure mode. LLM pre-training (next-token cross-entropy), VAEs (via the ELBO) and diffusion (via a bound) are all maximum-likelihood methods. GANs are the main exception.

### Forward vs. reverse KL (frequent exam topic)
| | Forward D_KL(p_data ‖ p_θ) | Reverse D_KL(p_θ ‖ p_data) |
|---|---|---|
| Behaviour | **Mode covering** | **Mode seeking** |
| Why | p_θ must put mass wherever p_data does | p_θ punished for mass where p_data has none |
| Fit to bimodal data with 1 Gaussian | wide Gaussian over both modes | locks onto one mode |
| Used by / related to | Max likelihood, VAEs | GAN behaviour |
| Failure | blurry samples | **mode collapse** |

**Explanation.**
- *Forward* = E_p_data[log p_data/p_θ]. The average is taken over **data samples**. If some data region gets p_θ ≈ 0, log(p_data/p_θ) → ∞, an enormous penalty. So the model stretches to cover every mode, even if that puts mass in the empty gap between modes. With a limited model this gives over-dispersed, "blurry" samples (images that average several plausible images).
- *Reverse* = E_p_θ[log p_θ/p_data]. The average is taken over **model samples**. Missing a data mode costs nothing (the model never samples there), but putting mass where p_data ≈ 0 is heavily penalised. So the model prefers to fit one mode well: sharp but less diverse samples, in the extreme **mode collapse**.
- Sketch to remember: p_data has two bumps at −3 and +3; forward-KL Gaussian ≈ N(0, ~3²) spanning both; reverse-KL Gaussian ≈ N(+3, 1) sitting on one.
- GAN connection *(beyond slides)*: the original GAN objective, with an optimal discriminator, equals 2·JS(p_data‖p_g) − log 4, where JS is the Jensen–Shannon divergence (a symmetrised KL). In practice GANs behave in a mode-seeking way, which is why "related to" rather than "equals" appears in the table.
- With an infinitely flexible model, both KLs are minimised at p_θ = p_data. The difference only matters when the model can't represent p_data exactly, which is always the case in practice.

---

## 5. The landscape of generative models

### Taxonomy (after Goodfellow, NIPS 2016) — how does a model access p(x)?
- **Explicit, tractable density:** GMM, PPCA, autoregressive (GPT), normalising flows → exact log p(x)
- **Explicit, intractable / approximate:** VAEs, diffusion → bound (ELBO) on log p(x); energy-based models (EBMs)
- **Implicit (sample only):** GANs → z ~ p(z), x = G_θ(z); no p(x) to evaluate

**Explanation of the axis.** The question is "can you give me the number p_θ(x) for this x?"
- *Tractable:* yes, exactly. So you can train by maximum likelihood directly and report test likelihood.
- *Approximate:* the model defines p(x) mathematically, but computing it needs an intractable integral (VAE, diffusion) or a normalising constant (EBM: p(x) = exp(−E_θ(x))/Z_θ, where Z_θ sums over all possible x). You train on a bound or an approximation instead.
- *Implicit:* the model is only a sampling procedure. It defines a distribution (the distribution of G(z)), but there is no formula for its density, so training needs a likelihood-free method such as a discriminator.

### Model families at a glance
| Family | Likelihood | Sampling | Lectures |
|---|---|---|---|
| Autoregressive | Exact | Sequential | L10–11 |
| Normalising flow | Exact | One pass (invertible) | overview |
| VAE | Lower bound (ELBO) | One pass | L4–5 |
| GAN | None | One pass | L6–7 |
| Diffusion | Lower bound | Iterative, many steps | L8–9 |

Each family trades off **likelihood, sampling speed, sample quality**.

| Family | Main strength | Main weakness | Typical use today |
|---|---|---|---|
| Autoregressive | exact likelihood, very high quality on text | slow sequential sampling, no latent z | LLMs (GPT, Llama), audio |
| Flow | exact likelihood + fast sampling | architecture restricted to invertible maps, dim z = dim x | density estimation, some audio |
| VAE | fast sampling, useful latent space | samples often blurry, only a bound | representation learning; latent space of Stable Diffusion |
| GAN | sharp samples, fast | unstable training, mode collapse, no likelihood | super-resolution, style transfer, fast image synthesis |
| Diffusion | state-of-the-art image quality and diversity, stable training | slow (many denoising steps) | image/video/audio generation |

*(beyond slides)* This is sometimes called the **generative learning trilemma**: high sample quality, mode coverage and fast sampling. GANs give quality + speed, VAEs coverage + speed, diffusion quality + coverage.

### How each works
- **Autoregressive (GPT, PixelCNN):** p_θ(x) = ∏ₜ p_θ(xₜ|x₍<t₎). Predict next token from previous; sample one at a time. e.g. "The capital of Bangladesh is" → p(Dhaka|·) = 0.97.
  - Training is **parallel**: the true prefix is known for every position, so all T predictions are computed in one pass (teacher forcing, causal masking in Transformers).
  - Sampling is **sequential**: token t+1 needs token t, so generating T tokens takes T network calls. That is why LLM generation streams word by word.
  - Exact log p(x) is just the sum of the per-token log-probabilities, which gives the perplexity metric (§6).
- **Normalising flows (RealNVP, Glow):** invertible x = f_θ(z), z ~ N(0,I); change of variables:
  p_θ(x) = p(z) · |det ∂f_θ⁻¹(x)/∂x|, z = f_θ⁻¹(x).
  Exact likelihood + one-pass sampling, but **dim z = dim x** and f must be **invertible**.
  - Why the determinant: stretching space spreads the same probability over more volume, so density drops. The Jacobian determinant measures that local volume change.
  - 1-D example: x = 2z with z ~ N(0,1). Then f⁻¹(x) = x/2, |d f⁻¹/dx| = 1/2, so p(x) = N(x/2; 0, 1)·½ = N(x; 0, 2²) ✓ (twice as wide, half as tall).
  - The architecture is designed (e.g. coupling layers) so that both the inverse and the determinant are cheap. Computing a general D×D determinant costs O(D³).
- **VAE (L4–5):** x →q_φ(z|x)→ z →p_θ(x|z)→ x̂
  **log p_θ(x) ≥ E_q_φ[log p_θ(x|z)] − D_KL(q_φ(z|x) ‖ p(z))**
  (reconstruction term − "stay near prior" term)
  - Reconstruction term: encode x, sample z, decode; reward if x̂ is close to x (with a Gaussian decoder this is MSE, see §1).
  - KL term: keep each encoding close to N(0, I), so that sampling z ~ N(0, I) at generation time lands in regions the decoder knows.
  - The gap between log p_θ(x) and the bound is exactly D_KL(q_φ(z|x) ‖ p_θ(z|x)): the better the encoder approximates the true posterior, the tighter the bound (derived in L3).
  - Gradients flow through the sampling step thanks to the **reparameterisation trick**: z = μ_φ(x) + σ_φ(x)·ε, ε ~ N(0, I) (L4).
  - β-VAE (A2) multiplies the KL term by β > 1 to encourage disentangled latents, at some cost in reconstruction.
- **GAN (L6–7):** generator vs. discriminator
  **min_G max_D E_p_data[log D(x)] + E_z[log(1 − D(G(z)))]**
  - D(x) ∈ [0,1] is the discriminator's probability that x is real. D wants it high on real data and low on fakes; G wants its fakes to score high.
  - For a fixed G, the optimal discriminator is D*(x) = p_data(x)/(p_data(x) + p_g(x)). The discriminator effectively estimates the density *ratio* without either density being written down. That is how a GAN learns without a likelihood.
  - Training is a two-player game, not a simple minimisation, so it can oscillate or collapse. WGAN-GP (A2) replaces the objective with a Wasserstein distance plus a gradient penalty to make training more stable.
- **Diffusion (L8–9):** forward adds noise step by step, **q(xₜ|xₜ₋₁) = N(√(1−βₜ) xₜ₋₁, βₜ I)** until x_T ≈ N(0, I). Reverse: network learns p_θ(xₜ₋₁|xₜ); generate by denoising from pure noise.
  - βₜ is a small noise schedule (e.g. from 10⁻⁴ to 0.02 over T = 1000 steps in DDPM). The √(1−βₜ) factor shrinks the signal so the variance stays bounded.
  - Useful closed form *(L8)*: you can jump to any step directly, q(xₜ|x₀) = N(√ᾱₜ x₀, (1−ᾱₜ) I) with ᾱₜ = ∏ₛ≤ₜ (1−βₛ). Training picks a random t, adds noise in one go, and the network predicts that noise with an MSE loss.
  - Sampling runs the network once per step (hundreds to a thousand steps), which is why it is slow. DDIM (A3) uses a deterministic sampler that skips steps to make it much faster.
  - Latent diffusion (Stable Diffusion) runs this process in a VAE's latent space instead of pixel space. Families are often combined in practice.

### Short history (timeline)
| Year | Milestone |
|---|---|
| 1977 | EM algorithm (Dempster, Laird & Rubin); HMMs also predate deep learning |
| 2013 | VAE (Kingma & Welling) |
| 2014 | GAN (Goodfellow et al.) |
| 2015 | Diffusion idea (Sohl-Dickstein et al.) |
| 2017 | Transformer (Vaswani et al.) |
| 2020 | DDPM (Ho et al.), GPT-3 (Brown et al.) |
| 2021 | CLIP (Radford et al.) |
| 2022 | Stable Diffusion (Rombach et al.), ChatGPT (OpenAI) |

Ideas are older than the hype; neural nets made them scale.

Reading the timeline: the probabilistic machinery (latent variables, EM, Markov chains) is decades old. What changed after ~2012 was (1) neural networks as flexible p(x|z) or p(xₜ|x₍<t₎), (2) GPUs and large datasets, (3) training tricks like reparameterisation and stable architectures. The 2015 diffusion paper was largely ignored until DDPM (2020) made it work well. Good ideas can wait years for the engineering to catch up.

---

## 6. Evaluation (preview of L13)

**Three criteria:** likelihood (held-out log-likelihood, perplexity) · sample quality (FID, humans) · diversity/coverage.
⚠️ They can disagree: **high likelihood ≠ good samples** (Theis et al., 2016).

| Metric | Definition | Better |
|---|---|---|
| NLL | −(1/N) Σ log p_θ(xₙ) on held-out data | lower |
| Bits per dimension | NLL / (D ln 2), for images | lower |
| Perplexity | exp(−(1/T) Σ log p_θ(xₜ\|x₍<t₎)) | lower |
| FID | ‖μ_r − μ_g‖² + Tr(Σ_r + Σ_g − 2(Σ_rΣ_g)^½) on Inception features | lower |

**Explanations.**
- **NLL** must be measured on **held-out** data. On training data a model that memorises gets a great score. By §4, lower test NLL means smaller estimated KL from the data distribution. Only explicit models (and bounds, for VAEs/diffusion) can report it; GANs cannot.
- **Bits per dimension** normalises NLL so models on different image sizes can be compared: divide the per-image NLL (nats) by D and by ln 2 to convert to bits. Example: NLL = 2,000 nats on a 28×28 image → 2000 / (784 × 0.693) ≈ **3.68 bits/dim**. For reference, storing raw 8-bit pixels costs 8 bits/dim, so the model "compresses" each pixel to 3.68 bits.
- **Perplexity** is exp(per-token cross-entropy) for language models. Interpret it as an "effective number of choices" per token: a uniform guess over a vocabulary of V words has perplexity V. ⚠️ Perplexities are only comparable between models with the **same tokeniser**, because a different tokenisation changes what counts as one token (important for Bangla, where tokenisers vary a lot).
- **FID (Fréchet Inception Distance):** pass real and generated images through a pretrained Inception-v3 network, take the 2048-dimensional pooled features, fit a Gaussian (mean μ, covariance Σ) to each set, and compute the Fréchet (2-Wasserstein) distance between the two Gaussians. The first term compares average content; the trace term compares spread, so it penalises both bad quality **and** low diversity. ⚠️ FID is biased for small sample sizes (use the same number of samples, commonly 10k–50k, when comparing models); it depends on an ImageNet-trained network, so it is less meaningful for very different domains. You will compute it in A3.
- **Human evaluation** is still the gold standard for quality, but it is expensive and hard to reproduce.

- **Reading perplexity:** if every correct next token gets probability 1/4, perplexity = 4 → as unsure as picking uniformly among 4 words.

**Why high likelihood ≠ good samples (Theis et al.).** Two directions:
1. *Good likelihood, bad samples:* take a good model p and mix it with junk: 0.01·p + 0.99·noise. Its log-likelihood drops by at most log 100 ≈ 4.6 nats per image, which is tiny compared with thousands of nats per image (well under 0.01 bits/dim). Yet 99% of its samples are noise.
2. *Bad likelihood, good samples:* a model that outputs training images exactly has perfect-looking samples but assigns zero likelihood to any new test image. GANs can produce excellent samples while having no likelihood at all.
Lesson: report several metrics, and match the metric to the use case (likelihood for compression/scoring, FID/human ratings for generation quality, coverage metrics for diversity).

---

## 7. Summary (the six take-aways)
1. Generative learns p(x) or p(x,y); discriminative learns p(y|x).
2. Modelling p(x) lets us sample, score, fill in, represent, condition.
3. Latent-variable models: p(x) = ∫ p(x|z)p(z)dz — simple parts, rich marginal.
4. The posterior p(z|x) is usually intractable → approximate inference.
5. Maximum likelihood ≡ minimising D_KL(p_data ‖ p_θ).
6. Families differ in how they access p(x): exactly, via a bound, or not at all.

The one-paragraph version: we want p_θ ≈ p_data, and the principled measure of "≈" is KL, which turns into maximum likelihood. For high-dimensional data, we get expressive p_θ by introducing latent variables (simple pieces, complicated marginal), but then computing p(x) and p(z|x) requires integrals we cannot do. Each model family is a different way around that integral: avoid latents entirely (autoregressive), make the map invertible (flows), bound the integral (VAE, diffusion), or skip the density altogether (GAN).

---

## 8. Key terms (flash cards)
| Term | Meaning |
|---|---|
| Generative model | Learns p(x) or p(x,y); can sample new data |
| Discriminative model | Learns p(y\|x) or a decision rule; predicts labels |
| Latent variable z | Hidden cause of x, never observed |
| Prior / likelihood / posterior | p(z) / p(x\|z) / p(z\|x) |
| Evidence (marginal likelihood) | p(x) = ∫ p(x\|z)p(z)dz |
| Ancestral sampling | Sample parents first: z ~ p(z), then x ~ p(x\|z) |
| Manifold hypothesis | Real data lie near a low-dim surface in ℝᴰ |
| Maximum likelihood | Choose θ to maximise Σ log p_θ(xₙ) |
| KL divergence | Non-negative, asymmetric gap between distributions |
| Explicit / implicit density | Gives p(x) (exact or bounded) / only samples |
| ELBO | Evidence lower bound on log p(x); trains VAEs |
| Responsibility | Posterior p(z=k\|x) in a mixture; the soft cluster assignment |
| Cross-entropy | H(p,q) = −E_p[log q]; = H(p) + D_KL(p‖q) |
| Mode collapse | Generator covers only a few modes of p_data |
| Monte Carlo estimate | Replace an expectation with an average over samples |
| Change of variables | p(x) = p(z)·\|det ∂z/∂x\|; basis of normalising flows |
| Perplexity | exp(per-token NLL); effective number of choices per token |
| FID | Fréchet distance between Gaussians fit to Inception features of real vs. generated images |

---

## 9. Practice questions

### From the slides (with answers)
1. **A model outputs p(spam|email). Generative or discriminative? Can it write a new email?**
   Discriminative. No — it never models p(email).
2. **p(Cat)=0.3, p(x|Cat)=0.04, p(x|Dog)=0.01. Find p(Cat|x).**
   p(Dog)=0.7 → 0.012 / (0.012 + 0.007) = 0.012/0.019 ≈ **0.63**. (Note how the likelihood ratio of 4:1 for Cat is partly cancelled by the prior of 3:7 against it.)
3. **Why do we need p(z|x) to fit a GMM?**
   Cluster labels are hidden; we must estimate which component made each point before re-fitting the means (E-step).
4. **Why is p(z|x) intractable in a VAE but not a GMM?**
   GMM: z takes K values → finite sum. VAE: z continuous + neural decoder → ∫ has no closed form.
5. **Why does E_p_data[log p_data] drop out of the ML/KL argument?**
   It does not depend on θ, so it doesn't change the arg min.
6. **Which family cannot report log p(x) for a test image?**
   GANs — they define x = G(z) only as a sampler, no density.

### Extra practice *(beyond slides, with answers)*
7. **A density takes the value 5 at some point. Is the model broken?** No. Densities can exceed 1; only the integral must be 1 (e.g. a Gaussian with σ ≈ 0.08 has peak ≈ 5).
8. **Why is a text-to-image model generative even though it takes an input?** It models a distribution over the high-dimensional output x given the prompt c, i.e. p(x|c), and it can sample new x.
9. **You fit one Gaussian to bimodal data by maximum likelihood. Where does its mean go, and why?** Between the modes, with a large variance. Maximum likelihood = forward KL, which is mode covering.
10. **What does a perplexity of 1 mean?** The model assigns probability 1 to every correct next token: perfect prediction on that data (and on a held-out set, almost certainly a bug or data leak).
11. **Why can a normalising flow not have d ≪ D?** It must be invertible, and an invertible map cannot change dimension, so dim z = dim x.

### Think–Pair–Share (discussion; prepare short answers)
1. For a chatbot, what is x, and what does sampling from p(x) mean? *(x = a token sequence; sampling = generating text token by token via the chain rule.)*
2. Does an LLM have a latent variable z? *(Argue: no explicit z — it is a fully observed autoregressive model; or yes — hidden states act as a deterministic internal representation, but not a probabilistic latent.)*
3. A task where discriminative suffices vs. one needing generative. *(e.g. Bangla spam / crop-disease classification vs. Bangla text generation, synthetic data, imputation of missing health records.)*

### Self-check drills (do these by hand)
- Compute D_KL(p‖q) and D_KL(q‖p) for the coin example; explain why they differ. *(0.511 vs. 0.368 nats; see §1.)*
- Derive θ̂ for a coin with h heads in n flips (answer: h/n).
- Derive μ̂ and σ̂² for a Gaussian by setting derivatives to zero. *(See §1, "Derivation".)*
- Fill in the GMM responsibility table for a different x (e.g. x = 4). *(Answer in §3: ≈ 0.000, 0.014, 0.986.)*
- Write the ELBO, the GAN objective, and the diffusion forward step from memory.
- Draw the explicit/implicit taxonomy and place all five families.
- Explain mode covering vs. mode seeking with a sketch of a bimodal p_data.

---

## Reading before L2
- Lecture 1 Reading Material (Sections 2–5 in detail)
- Bishop & Bishop, *Deep Learning: Foundations and Concepts* — probability & mixture models
- Goodfellow, *NIPS 2016 Tutorial on GANs* — taxonomy section

**Next:** GMMs & EM, HMMs. **Assignment A1** released (EM for a GMM + HMM decoder), due Lecture 4.

For A1, the parts of L1 you will use directly: the GMM definition and responsibilities (§3), Gaussian MLE formulas (§1, these become the M-step), log-likelihood monitoring (the EM log-likelihood should never decrease), and the HMM tractability note (§3). Work in log space (log-sum-exp) to avoid the underflow mentioned in §1.
