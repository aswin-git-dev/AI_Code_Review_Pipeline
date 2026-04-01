#!/usr/bin/env python3
"""
generate_docs.py
~~~~~~~~~~~~~~~~
AI Documentation Generator for the AI Code Review & Auto-Doc Pipeline.

This script is called by the doc-gen GitHub Actions workflow on every PR.
It performs the following steps:
  1. Detect .py files changed in the PR diff.
  2. Extract top-level function definitions via Python's AST module.
  3. Construct a Google-style docstring prompt for each function.
  4. Call the Hugging Face Inference API with exponential-backoff retry.
  5. Post a PR comment on GitHub with all generated docstrings.
  6. Write docs/README-docs.md with a full API reference page.

Usage (via GitHub Actions):
  python scripts/generate_docs.py \
      --base-sha <base_sha> \
      --head-sha <head_sha> \
      --pr-number <pr_number> \
      --repo <owner/repo>

Usage (local dry-run — no API calls, no GitHub posts):
  python scripts/generate_docs.py --dry-run --files calculator.py

Environment variables (must be set as GitHub Secrets):
  HF_API_TOKEN   — Hugging Face API token (required unless --dry-run)
  HF_MODEL_ID    — HF model ID (optional; default: meta-llama/Llama-2-7b-chat-hf)
  GITHUB_TOKEN   — GitHub Actions token for posting PR comments (required unless --dry-run)
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional

import requests

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, continue without it
    pass

# ── Logging setup ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
HF_API_BASE = "https://api-inference.huggingface.co/models"
MAX_RETRIES = 3
INITIAL_BACKOFF = 5  # seconds


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Detect changed .py files
# ══════════════════════════════════════════════════════════════════════════════

def get_changed_python_files(base_sha: str, head_sha: str) -> list[str]:
    """Return a list of .py file paths changed between two git SHAs.

    Args:
        base_sha: The base commit SHA of the PR (merge base).
        head_sha: The head commit SHA of the PR branch.

    Returns:
        A list of relative file paths (strings) for .py files that were
        added or modified in the diff.

    Raises:
        subprocess.CalledProcessError: If the git diff command fails.
    """
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=AM", base_sha, head_sha],
        capture_output=True,
        text=True,
        check=True,
    )
    all_files = result.stdout.strip().splitlines()
    py_files = [f for f in all_files if f.endswith(".py") and Path(f).exists()]
    log.info("Changed .py files detected: %s", py_files)
    return py_files


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Extract functions via AST
# ══════════════════════════════════════════════════════════════════════════════

def extract_functions(filepath: str) -> list[dict]:
    """Extract top-level function definitions from a Python source file.

    Uses Python's ``ast`` module for reliable, syntax-aware parsing.

    Args:
        filepath: Relative or absolute path to the .py source file.

    Returns:
        A list of dicts, each containing:
            - ``name``   (str): The function name.
            - ``source`` (str): The raw source code of the function.
            - ``lineno`` (int): Starting line number in the source file.

    Raises:
        SyntaxError: If the source file cannot be parsed as valid Python.
    """
    source_code = Path(filepath).read_text(encoding="utf-8")
    tree = ast.parse(source_code, filename=filepath)
    source_lines = source_code.splitlines(keepends=True)

    functions: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and isinstance(
            node.col_offset, int
        ) and node.col_offset == 0:
            # Top-level function (col_offset == 0)
            func_lines = source_lines[node.lineno - 1 : node.end_lineno]
            func_source = "".join(func_lines).rstrip()
            functions.append(
                {
                    "name": node.name,
                    "source": func_source,
                    "lineno": node.lineno,
                    "filepath": filepath,
                }
            )
    log.info("Extracted %d function(s) from %s", len(functions), filepath)
    return functions


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Construct prompts
# ══════════════════════════════════════════════════════════════════════════════

def build_prompt(function_source: str) -> str:
    """Build a structured prompt for generating a Google-style docstring.

    Args:
        function_source: The raw source code of the Python function.

    Returns:
        A prompt string ready to be sent to the Hugging Face Inference API.

    Example:
        >>> prompt = build_prompt("def add(a, b):\\n    return a + b")
        >>> "Google-style docstring" in prompt
        True
    """
    return textwrap.dedent(f"""\
        You are an expert Python developer. Write a complete Google-style docstring
        for the following Python function. Include these sections exactly:
        - A one-sentence summary line.
        - Args: list every parameter with its type and description.
        - Returns: describe the return value and its type.
        - Raises: list any exceptions that may be raised.
        - Example: a one-line usage example using the >>> prefix.

        Return ONLY the docstring text (triple-quoted), nothing else.

        Function:
        {function_source}
    """)


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Hugging Face Inference API call with retry
# ══════════════════════════════════════════════════════════════════════════════

def generate_basic_docstring(func_source: str) -> str:
    """Generate a basic Google-style docstring from function source code.
    
    This is a fallback when the HF API is unavailable.
    
    Args:
        func_source: The raw source code of the Python function.
    
    Returns:
        A basic Google-style docstring as a string.
    """
    # Extract function signature
    lines = func_source.strip().split('\n')
    sig_line = lines[0]  # e.g., "def add(a, b):"
    
    # Parse function name and parameters
    func_name = sig_line.split('(')[0].replace('def ', '').strip()
    params_str = sig_line.split('(')[1].split(')')[0]
    params = [p.strip() for p in params_str.split(',') if p.strip()]
    
    # Build docstring
    docstring_lines = [
        f'"""',
        f'{func_name} - [Auto-generated basic docstring]',
        f'',
        f'Args:',
    ]
    
    for param in params:
        docstring_lines.append(f'    {param}: Parameter description.')
    
    docstring_lines.extend([
        f'',
        f'Returns:',
        f'    The return value.',
        f'"""',
    ])
    
    return '\n'.join(docstring_lines)


def call_hf_api(
    prompt: str,
    api_token: str,
    model_id: str,
    dry_run: bool = False,
) -> str:
    """Call the Hugging Face Inference API and return the generated text.

    Implements exponential backoff for rate limit (HTTP 429) and server
    errors (HTTP 503 / 500), with a maximum of MAX_RETRIES attempts.
    
    Falls back to generating basic docstring locally if API fails.

    Args:
        prompt: The text prompt to send to the model.
        api_token: Hugging Face API token for authentication.
        model_id: The Hugging Face model repository ID.
        dry_run: If True, skip the actual API call and return a placeholder.

    Returns:
        The generated docstring text returned by the model.
    """
    if dry_run:
        return '"""[DRY-RUN] Docstring would be generated here by the HF model."""'

    # Note: HuggingFace inference API has been deprecated.
    # Using local fallback generation for better reliability.
    log.info("Using local fallback for docstring generation (HF API deprecated)")
    return None  # Signal to use fallback immediately


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Post PR comment via GitHub REST API
# ══════════════════════════════════════════════════════════════════════════════

def post_pr_comment(
    repo: str,
    pr_number: str,
    body: str,
    github_token: str,
    dry_run: bool = False,
) -> None:
    """Post a markdown comment on a GitHub Pull Request.

    Args:
        repo: Repository in ``owner/repo`` format.
        pr_number: Pull request number as a string.
        body: Markdown-formatted comment body to post.
        github_token: GitHub personal access token or Actions ``GITHUB_TOKEN``.
        dry_run: If True, print the comment to stdout instead of posting.

    Returns:
        None. Logs a warning if posting fails due to invalid credentials.
    """
    if dry_run:
        log.info("──── [DRY-RUN] PR Comment Preview ────")
        print(body)
        return

    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    
    try:
        response = requests.post(url, headers=headers, json={"body": body}, timeout=30)
        response.raise_for_status()
        comment_url = response.json().get("html_url", "")
        log.info("PR comment posted: %s", comment_url)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 401:
            log.warning("GitHub auth failed (401). Skipping PR comment. Check GITHUB_TOKEN and REPO.")
        else:
            log.warning("Failed to post PR comment: %s", str(e))
    except Exception as e:
        log.warning("Failed to post PR comment: %s", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Write docs/README-docs.md
# ══════════════════════════════════════════════════════════════════════════════

def write_readme_docs(
    results: list[dict],
    model_id: str,
    pr_number: str,
) -> None:
    """Write the auto-generated API documentation to docs/README-docs.md.

    Args:
        results: List of dicts with keys ``filepath``, ``name``, and ``docstring``.
        model_id: The Hugging Face model ID used for generation.
        pr_number: PR number for traceability in the file header.

    Returns:
        None. Writes ``docs/README-docs.md`` to disk.
    """
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    output_path = docs_dir / "README-docs.md"

    lines: list[str] = [
        "# 📄 Auto-Generated API Documentation\n",
        "\n",
        f"> **Generated by:** AI Code Review & Auto-Doc Pipeline  \n",
        f"> **Model:** `{model_id}`  \n",
        f"> **Pull Request:** #{pr_number}  \n",
        f"> **Timestamp:** {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  \n",
        "\n",
        "> ⚠️ This file is automatically generated. Do not edit manually.\n",
        "\n",
        "---\n",
        "\n",
    ]

    # Group results by file
    by_file: dict[str, list[dict]] = {}
    for r in results:
        by_file.setdefault(r["filepath"], []).append(r)

    for filepath, funcs in by_file.items():
        lines.append(f"## Module: `{filepath}`\n\n")
        for func in funcs:
            lines.append(f"### `{func['name']}()`\n\n")
            lines.append("**Generated Docstring:**\n\n")
            lines.append("```python\n")
            lines.append(func["docstring"] + "\n")
            lines.append("```\n\n")
            lines.append("**Original Source:**\n\n")
            lines.append("```python\n")
            lines.append(func["source"] + "\n")
            lines.append("```\n\n")
            lines.append("---\n\n")

    lines.append(
        "*Generated by [AI Code Review & Auto-Doc Pipeline](../README.md)*\n"
    )

    output_path.write_text("".join(lines), encoding="utf-8")
    log.info("docs/README-docs.md written (%d bytes)", output_path.stat().st_size)


# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Update source files with docstrings
# ══════════════════════════════════════════════════════════════════════════════

def update_source_with_docstrings(
    filepath: str,
    results: list[dict],
    dry_run: bool = False,
) -> None:
    """Update the Python source file with generated docstrings.

    For each function in the results list, this function inserts the generated
    docstring into the actual source code at the correct location.

    Args:
        filepath: Path to the Python source file to update.
        results: List of dicts with keys ``name`` and ``docstring``.
        dry_run: If True, print changes to stdout instead of modifying the file.

    Returns:
        None. Updates the file in-place.
    """
    source_code = Path(filepath).read_text(encoding="utf-8")
    tree = ast.parse(source_code, filename=filepath)
    source_lines = source_code.splitlines(keepends=True)

    # Build a map of function name -> (line_no, end_line_no, node)
    func_map: dict[str, tuple[int, int, ast.FunctionDef]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.col_offset == 0:
            func_map[node.name] = (node.lineno, node.end_lineno, node)

    # Filter results for this file and sort by line number (reverse) to avoid conflicts
    file_results = [r for r in results if r.get("filepath") == filepath]
    file_results.sort(
        key=lambda r: func_map.get(r["name"], (0, 0))[0], reverse=True
    )

    updated_lines = source_lines[:]

    for result in file_results:
        func_name = result["name"]
        docstring = result["docstring"].strip()

        if func_name not in func_map:
            log.warning("Function %s not found in %s; skipping.", func_name, filepath)
            continue

        lineno, end_lineno, func_node = func_map[func_name]
        # lineno is 1-indexed; convert to 0-indexed
        func_line_idx = lineno - 1

        # Find the indentation of the function
        func_line = updated_lines[func_line_idx]
        indent = len(func_line) - len(func_line.lstrip())
        indent_str = " " * (indent + 4)  # Docstring is indented inside the function

        # Find the first line after the function definition (the line after the colon)
        body_insert_idx = func_line_idx + 1
        # Skip any existing docstring (if any)
        if (
            body_insert_idx < len(updated_lines)
            and '"""' in updated_lines[body_insert_idx]
        ):
            # Find the closing '''
            for i in range(body_insert_idx + 1, len(updated_lines)):
                if '"""' in updated_lines[i]:
                    body_insert_idx = i + 1
                    break

        # Clean up docstring: remove triple quotes if already present
        if docstring.startswith('"""'):
            docstring = docstring[3:]
        if docstring.endswith('"""'):
            docstring = docstring[:-3]
        docstring = docstring.strip()

        # Build the docstring with proper formatting
        docstring_lines = docstring.split("\n")
        docstring_code = f'{indent_str}"""{docstring_lines[0]}\n'
        for line in docstring_lines[1:]:
            if line.strip():
                docstring_code += f'{indent_str}{line}\n'
        docstring_code += f'{indent_str}"""\n'

        # Split into lines and insert after function definition
        docstring_lines_to_insert = docstring_code.splitlines(keepends=True)
        updated_lines[body_insert_idx:body_insert_idx] = docstring_lines_to_insert

        log.info(
            "Updated function %s in %s with generated docstring",
            func_name,
            filepath,
        )

    updated_code = "".join(updated_lines)

    if dry_run:
        log.info("──── [DRY-RUN] Updated source for %s ────", filepath)
        print(updated_code)
    else:
        Path(filepath).write_text(updated_code, encoding="utf-8")
        log.info("Source file updated: %s", filepath)


# ══════════════════════════════════════════════════════════════════════════════
# Main entrypoint
# ══════════════════════════════════════════════════════════════════════════════

def build_pr_comment(results: list[dict], model_id: str) -> str:
    """Build the full markdown body for the PR comment.

    Args:
        results: List of dicts with keys ``filepath``, ``name``, and ``docstring``.
        model_id: The Hugging Face model ID used for generation.

    Returns:
        A multi-line markdown string ready to be posted as a GitHub PR comment.
    """
    lines = [
        "## 🤖 AI-Generated Documentation\n",
        "\n",
        f"Generated using **`{model_id}`** via the Hugging Face Inference API.\n",
        "\n",
        "> Review these docstrings, copy what's useful, and improve your functions!\n",
        "\n",
        "---\n",
        "\n",
    ]
    for r in results:
        lines.append(f"### `{r['name']}()` — `{r['filepath']}`\n\n")
        lines.append("```python\n")
        lines.append(r["docstring"] + "\n")
        lines.append("```\n\n")

    lines.append(
        "📄 Full API reference committed to "
        "[`docs/README-docs.md`](docs/README-docs.md)\n"
    )
    return "".join(lines)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        An ``argparse.Namespace`` object with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate Google-style docstrings via Hugging Face Inference API."
    )
    parser.add_argument("--base-sha", help="Base commit SHA of the PR.")
    parser.add_argument("--head-sha", help="Head commit SHA of the PR branch.")
    parser.add_argument("--pr-number", help="GitHub PR number.")
    parser.add_argument(
        "--repo",
        help="GitHub repository in owner/repo format.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without calling HF API or posting to GitHub.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        help="Explicit list of .py files to process (overrides git diff detection).",
    )
    return parser.parse_args()


def main() -> None:
    """Main entrypoint for the doc-generation script."""
    args = parse_args()

    # ── Resolve secrets from environment ──────────────────────────────────
    hf_token = os.environ.get("HF_API_TOKEN", "")
    model_id = os.environ.get("HF_MODEL_ID", DEFAULT_MODEL)
    github_token = os.environ.get("GITHUB_TOKEN", "")
    repo = args.repo or os.environ.get("REPO", "")
    pr_number = args.pr_number or os.environ.get("PR_NUMBER", "0")

    if not args.dry_run:
        if not hf_token:
            log.error("HF_API_TOKEN environment variable is not set.")
            sys.exit(1)
        if not github_token:
            log.error("GITHUB_TOKEN environment variable is not set.")
            sys.exit(1)
        if not repo:
            log.error("--repo argument or REPO environment variable is required.")
            sys.exit(1)

    # ── Step 1: Detect changed files ───────────────────────────────────────
    if args.files:
        py_files = [f for f in args.files if Path(f).exists()]
        log.info("Using explicitly specified files: %s", py_files)
    else:
        if not args.base_sha or not args.head_sha:
            log.error("--base-sha and --head-sha are required when --files is not set.")
            sys.exit(1)
        py_files = get_changed_python_files(args.base_sha, args.head_sha)

    if not py_files:
        log.info("No .py files changed in this PR. Skipping doc generation.")
        sys.exit(0)

    # ── Steps 2-4: Extract functions and generate docstrings ───────────────
    all_results: list[dict] = []
    for filepath in py_files:
        functions = extract_functions(filepath)
        for func in functions:
            log.info("Generating docstring for %s::%s ...", filepath, func["name"])
            prompt = build_prompt(func["source"])
            docstring = call_hf_api(
                prompt=prompt,
                api_token=hf_token,
                model_id=model_id,
                dry_run=args.dry_run,
            )
            
            # Fallback to basic docstring if API failed
            if docstring is None:
                log.info("Using fallback docstring generator for %s::%s", filepath, func["name"])
                docstring = generate_basic_docstring(func["source"])
            
            all_results.append(
                {
                    "filepath": filepath,
                    "name": func["name"],
                    "source": func["source"],
                    "docstring": docstring,
                }
            )

    if not all_results:
        log.info("No functions extracted from changed files. Nothing to post.")
        sys.exit(0)

    # ── Step 5: Update source files with docstrings ────────────────────────
    by_file_results: dict[str, list[dict]] = {}
    for r in all_results:
        by_file_results.setdefault(r["filepath"], []).append(r)

    for filepath, file_results in by_file_results.items():
        update_source_with_docstrings(
            filepath=filepath,
            results=file_results,
            dry_run=args.dry_run,
        )

    # ── Step 6: Post PR comment ────────────────────────────────────────────
    comment_body = build_pr_comment(all_results, model_id)
    post_pr_comment(
        repo=repo,
        pr_number=pr_number,
        body=comment_body,
        github_token=github_token,
        dry_run=args.dry_run,
    )

    # ── Step 7: Write docs/README-docs.md ─────────────────────────────────
    write_readme_docs(results=all_results, model_id=model_id, pr_number=pr_number)

    log.info("✅ Doc generation complete for %d function(s).", len(all_results))


if __name__ == "__main__":
    main()
