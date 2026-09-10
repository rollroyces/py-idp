# Security Policy

## Supported versions

| version | supported          |
|---------|--------------------|
| 0.3.x   | ✅ active          |
| 0.2.x   | ⚠️ critical fixes only |
| < 0.2   | ❌ end-of-life     |

py-idp is pre-1.0; the public API may shift between minor versions.
Pin to a minor (`py-idp>=0.3,<0.4`) in production.

## Reporting a vulnerability

**Please do not open a public GitHub issue for security problems.**

Email **rollroyces@users.noreply.github.com** (or DM via the GitHub
profile) with:

1. A short description of the issue and impact.
2. A minimal reproduction (script, sample document, or trace).
3. The affected version (commit SHA or `pip show py-idp | grep Version`).

I aim to acknowledge within **72 hours** and ship a fix or mitigation
within **14 days** for critical issues (RCE, credential leakage,
arbitrary code execution via crafted input). Lower-severity issues
are bundled into the next regular release.

## Scope

In scope:

* Deserialization of untrusted input (documents, schemas, fixtures).
* Path traversal in `Document.from_path` and storage backends.
* Prompt injection that escapes the extraction sandbox.
* SSRF via user-supplied URLs to `compat` / `ollama` / `china:*` backends.
* The `idp.api` FastAPI server (auth, rate limit, upload-size cap).

Out of scope:

* Vulnerabilities in upstream model weights (Nanonets, Qwen, etc.).
* Issues requiring a maliciously crafted *output* of an LLM.
* Denial of service via very large legitimate documents.

## Hall of fame

_(no reports yet — be the first)_
