# AI-Driven Code Review & Auto-Documentation Pipeline

> **"From Code to Clarity: AI-driven review + AI-driven docs in one pipeline."**

This repository is a hands-on demo for the STTP (Short-Term Training Programme) workshop. Every time you open a Pull Request, two AI systems activate automatically:

1. **CodeRabbit** — posts an AI code review with line-level comments and quality suggestions.
2. **Hugging Face Doc-Gen** — generates Google-style docstrings for every function you changed and posts them as a PR comment and commits `docs/README-docs.md`.

---

## 🗺️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GitHub Pull Request                        │
└────────────────────────┬────────────────────────────────────────┘
                         │ triggers
          ┌──────────────┴──────────────┐
          ▼                             ▼
  ┌───────────────┐           ┌──────────────────────┐
  │  CodeRabbit   │           │   GitHub Actions       │
  │  (App-based)  │           │   (doc-gen.yml)        │
  └───────┬───────┘           └──────────┬─────────────┘
          │                              │
          │ posts review                 │ runs
          │ comments                     ▼
          │                   ┌──────────────────────┐
          │                   │ scripts/generate_docs │
          │                   │  ├─ git diff → .py    │
          │                   │  ├─ ast.parse()        │
          │                   │  ├─ HF Inference API   │
          │                   │  ├─ post PR comment    │
          │                   │  └─ commit docs/       │
          │                   └──────────────────────┘
          ▼                              ▼
  ┌─────────────────────────────────────────────┐
  │          GitHub PR Page (End Result)        │
  │  ✅ CodeRabbit review comments              │
  │  ✅ AI-generated docstring comment          │
  │  ✅ docs/README-docs.md committed           │
  └─────────────────────────────────────────────┘
```

---

## ⚙️ Setup Guide

Complete the following 4 steps in **under 15 minutes** — no prior CI/CD experience needed.

### Step 1 — Fork this Repository

1. Click the **Fork** button at the top of this page.
2. Keep the default settings and click **Create fork**.

### Step 2 — Add GitHub Secrets

The doc-gen pipeline needs your Hugging Face API token.

1. Go to your forked repo → **Settings** → **Secrets and variables** → **Actions**.
2. Click **New repository secret** and add:

| Secret Name     | Value                                   | Required |
|-----------------|-----------------------------------------|----------|
| `HF_API_TOKEN`  | Your token from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | ✅ Yes |
| `HF_MODEL_ID`   | `Qwen/Qwen2.5-Coder-7B-Instruct`       | ⬜ Optional (this is the default) |

> **How to get a Hugging Face token:**
> 1. Sign up free at [huggingface.co](https://huggingface.co).
> 2. Go to **Settings → Access Tokens → New token**.
> 3. Select **Read** role and copy the token.

### Step 3 — Install CodeRabbit

1. Visit [coderabbit.ai](https://coderabbit.ai) and click **Get Started for Free**.
2. Connect your GitHub account.
3. Install the CodeRabbit GitHub App on your forked repository.

> CodeRabbit is free for public repositories. No credit card required.

### Step 4 — Open a Pull Request

1. Create a new branch:
   ```bash
   git checkout -b feature/add-calculator
   ```
2. Make a small change to `calculator.py` (e.g., add a new function).
3. Push and open a PR against `main`:
   ```bash
   git add calculator.py
   git commit -m "feat: add new calculator function"
   git push origin feature/add-calculator
   ```
4. Open a PR on GitHub and watch:
   - **CodeRabbit** posts a review within ~60 seconds.
   - **doc-gen** Action completes and posts generated docstrings within ~90 seconds.

---

## 📁 Repository Structure

```
.
├── calculator.py                   # Demo Python module (intentionally flawed)
├── .coderabbit.yaml                # CodeRabbit review configuration
├── .github/
│   └── workflows/
│       └── doc-gen.yml             # GitHub Actions: triggers doc generation on PR
├── scripts/
│   └── generate_docs.py           # Python script: HF API + AST + PR comment
├── docs/
│   └── README-docs.md             # Auto-generated API docs (updated per PR)
└── README.md                      # This file
```

---

## 🔒 Security

- All API tokens are stored as **GitHub Actions Encrypted Secrets**.
- **No credentials appear** in any committed file or in Action logs.
- The `GITHUB_TOKEN` used to post PR comments is the short-lived, auto-generated Actions token — no personal token needed for that step.

---

## 🧪 Local Dry-Run (No API Token Needed)

You can test the doc-generation script locally without calling any APIs:

```bash
pip install huggingface_hub requests
python scripts/generate_docs.py --dry-run --files calculator.py
```

This will:
- Parse `calculator.py` using AST.
- Print all extracted function names and their generated prompts.
- Print the PR comment preview to stdout.
- Write a placeholder `docs/README-docs.md`.

---

## 🤖 Technology Stack

| Layer            | Technology                              |
|------------------|-----------------------------------------|
| Source Control   | GitHub (free tier)                      |
| CI/CD            | GitHub Actions (free for public repos)  |
| AI Code Review   | CodeRabbit (free for public repos)      |
| LLM Inference    | Hugging Face Inference API (free tier)  |
| Primary LLM      | `Qwen/Qwen2.5-Coder-7B-Instruct`       |
| Script Language  | Python 3.9+                             |
| Doc Format       | GitHub-Flavored Markdown                |

---

## ❓ FAQ

**Q: Why are no docstrings in `calculator.py`?**  
A: Intentional! The file is designed to showcase what CodeRabbit flags and what the doc-gen pipeline produces.

**Q: The HF API is rate-limited — what happens?**  
A: The script automatically retries up to 3 times with exponential backoff (5s → 10s → 20s).

**Q: Can I use a different Hugging Face model?**  
A: Yes — set `HF_MODEL_ID` as a repository secret with any model ID available on the public Inference API.

**Q: Does this cost anything?**  
A: No. Hugging Face Inference API (free tier), CodeRabbit (free for public repos), and GitHub Actions (free for public repos) are all $0 for this demo.

---

## 📚 Resources

- [CodeRabbit Documentation](https://docs.coderabbit.ai)
- [Hugging Face Inference API Docs](https://huggingface.co/docs/api-inference)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Qwen2.5-Coder on HF Hub](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
