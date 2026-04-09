# PyPI Release Setup

This repository is configured for publishing with **Trusted Publishing** via GitHub Actions.

Workflow file:
- `.github/workflows/publish-pypi.yml`
- `.github/workflows/release-pipeline.yml` (version-gated CI build/test pipeline)

## 1) One-time PyPI/TestPyPI setup

1. Create projects on TestPyPI and PyPI (or reserve the name `pyframe-xpy`).
2. In each index, configure a Trusted Publisher:
   - Owner: your GitHub org/user
   - Repository: `kawa-technology/FrameX`
   - Workflow: `publish-pypi.yml`
   - Environment: `testpypi` for TestPyPI, `pypi` for PyPI
3. In GitHub repository settings, create environments:
   - `testpypi`
   - `pypi`

## 2) Local preflight

```bash
python3 -m pip install -e '.[release]'
python3 -m build
python3 -m twine check dist/*
```

## 3) Publish to TestPyPI

Run workflow manually:
- Actions -> `Publish To PyPI` -> `Run workflow`
- Select `repository = testpypi`

Install test package:

```bash
python3 -m pip install --index-url https://test.pypi.org/simple/ pyframe-xpy
```

## 4) Publish to PyPI

Option A (recommended): create a GitHub Release and publish it.
- Trigger: `release: published`

Option B: manual dispatch
- Actions -> `Publish To PyPI`
- Select `repository = pypi`

## Release CI/CD pipeline

`Release Build Pipeline` runs automatically on:
- tag push matching `v*.*.*` (example: `v0.1.0`)
- published GitHub Release

Pipeline stages:
1. Validate version consistency:
   - `pyproject.toml` `project.version`
   - `framex/_version.py` `__version__`
   - git tag (if present) must match version
2. Run test matrix on Python 3.10/3.11/3.12
3. Build `sdist` + `wheel`
4. Run `twine check`
5. Smoke test wheel install/import:
   - install the built wheel in a fresh virtual environment
   - verify `import framex as fx` succeeds
6. Upload `dist/` artifact

## 5) Version bump checklist

Before each publish:
1. Update version in `pyproject.toml`
2. Update version in `framex/_version.py`
3. Build + check:

```bash
python3 -m build
python3 -m twine check dist/*
```

4. Tag/release and publish.

## Notes

- `MANIFEST.in` excludes machine-specific binaries (`*.so`, `*.dylib`, `*.pyd`) from source distributions.
- `framex/backends/c_backend.py` compiles native kernels at runtime when available.
- Package name vs import name:
  - Install: `pip install pyframe-xpy`
  - Use: `import framex as fx`
