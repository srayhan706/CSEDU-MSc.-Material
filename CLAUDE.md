# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this directory is

This is not a software project. It is a personal coursework folder for an MSc in Computer Science and Engineering at the University of Dhaka (CSEDU). It holds course materials (PDF outlines and lecture slides) and, later, assignments and projects. It is not a git repository, and it has no build, lint or test tooling at the top level.

Layout: `<Semester>/<Course>/`, for example `Fall/GenAI/`. Each course folder keeps lecture slides in `Slide/` (`l1.pdf`, `l2.pdf`, …) and the course outline at its root.

Fall 2026 courses:
- `Fall/GenAI/`: CSE-5025 Generative Artificial Intelligence. `gencourseout.pdf` is the course outline.
- `Fall/VLSI/`: empty for now.
- `Fall/Res. Meth./`: Research Methodology, empty for now. Note the space and the period in the folder name, so quote paths.

The top-level folder name `CSEDU MSc.` also contains a space and a trailing period, so always quote absolute paths.

## CSE-5025 (GenAI): what Claude needs to know

- **Stack set by the course:** Python, PyTorch, Hugging Face Transformers / Diffusers / PEFT, FAISS or Chroma for vector search, and W&B or TensorBoard for experiment tracking.
- **Compute limits:** assignments must run on a single consumer GPU or on free Colab/Kaggle notebooks. LLM work uses models of 7B parameters or fewer, with LoRA or 4-bit quantization. Size models, datasets and step counts to fit these limits.
- **Assignments:**
  - A1: EM for a GMM and an HMM decoder, and monitoring the ELBO.
  - A2: VAE and β-VAE on MNIST/CelebA, then DCGAN and WGAN-GP.
  - A3: a small DDPM, comparing DDPM and DDIM sampling with FID.
  - A4: LoRA fine-tuning of a small open LLM, plus a RAG pipeline with factuality and hallucination rates.
- **Term project** (teams of 2–3): preferably a problem relevant to Bangladesh, such as Bangla text generation or Bangla RAG. The final report is 8 pages in conference format and must include an ethics and limitations section.
- **Generative AI policy (important):** AI help is allowed for coding and language editing only if the student discloses it in a short statement with each submission. It is **not** allowed for writing the paper review or in exams. When you help with an assignment, remind the user about the disclosure statement. Do not write the paper review.
- **Late penalty:** 10% per day, for up to 3 days.
