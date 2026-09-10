# PyPI Trusted Publisher setup — 2 minutes in your browser

The OIDC publish workflow is wired (`.github/workflows/publish.yml`) and the v0.3.1 release is tagged. The publish run failed with `invalid-publisher` because no PyPI trusted publisher matches this workflow yet. One-time setup:

## Steps

1. Open **https://pypi.org/manage/account/publishing/** (you must be signed in as the project owner — `rollroyces`).
2. Scroll to **"Pending publishers"** (or **"Add a new pending publisher"**).
3. Fill in:
   - **PyPI project name:** `py-idp`
   - **Owner:** `rollroyces`
   - **Repository name:** `py-idp`
   - **Workflow filename:** `publish.yml`
   - **Environment name:** `pypi` ← exact match required
4. Click **Add**. PyPI shows the request as "pending" — it becomes active the next time the workflow runs.
5. Tell me when done. I'll re-trigger the publish (re-run workflow `34463225636` from the Actions tab, OR delete the tag, re-push, OR I can re-run via API).

## What if the project doesn't exist on PyPI yet?

If you get "no such project", create a placeholder first:

1. Go to https://pypi.org/manage/projects/py-idp/ — you'll see "This project does not exist".
2. PyPI's UI may now offer to create it (it's been auto-claimed by the trusted-publisher registration).
3. If not, run once locally to bootstrap (needs API token, NOT trusted publishing):

   ```bash
   cd /Users/hermes/py-idp
   python -m pip install --upgrade build twine
   python -m build
   twine upload --skip-existing dist/*
   ```

   The `--skip-existing` is safe — once trusted publishing takes over, twine uploads are no longer needed.

4. Then come back and complete the trusted-publisher registration.

## Verification once setup is done

After I re-trigger the publish, you can verify the package is live:

```bash
pip index versions py-idp
# expected: py-idp 0.3.0, 0.3.1 (or just 0.3.1 if this is the first)

pip install py-idp==0.3.1
python -c "import idp; print(idp.__version__)"
# expected: 0.3.1
```

The README's PyPI badge will then resolve automatically to a green "0.3.1" status.
