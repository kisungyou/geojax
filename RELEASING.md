# Releasing GeoJAX

GeoJAX releases are built and uploaded manually. PyPI credentials must never
be committed to this repository or pasted into an issue, pull request, or chat.

## Account setup

1. Create separate accounts on [PyPI](https://pypi.org/account/register/) and
   [TestPyPI](https://test.pypi.org/account/register/).
2. Verify both email addresses, enable two-factor authentication, and store the
   recovery codes securely.
3. Create an account-scoped API token on TestPyPI for the first test upload.
   Once the project exists, a project-scoped token can be used instead.
4. Create the production PyPI token only when the tested release is ready to
   publish. Do not store tokens in `.pypirc`; let Twine prompt for them.

For token authentication, the username is `__token__` and the password is the
complete token, including its `pypi-` prefix. PyPI and TestPyPI use different
accounts and tokens.

## Prepare a release

1. Choose a version that has never been uploaded. Published files and version
   numbers cannot be replaced.
2. Update the version in `pyproject.toml`, `docs/conf.py`, `CITATION.cff`, the
   README release note, and `CHANGELOG.md`.
3. Run the checks and build into a clean, version-specific directory:

   ```bash
   ruff check geojax tests
   pytest
   MPLCONFIGDIR=/tmp/geojax-matplotlib make website
   python -m build --outdir dist/0.1.0
   python -m twine check dist/0.1.0/*
   ```

Replace `0.1.0` with the release being prepared.

## TestPyPI

Upload the exact artifacts intended for production:

```bash
python -m twine upload \
  --repository-url https://test.pypi.org/legacy/ \
  dist/0.1.0/*
```

At Twine's prompts, enter `__token__` and the TestPyPI token. Verify the wheel
from outside the source tree in a fresh environment. `--no-deps` is deliberate:
TestPyPI does not mirror all runtime dependencies.

```bash
python -m venv /tmp/geojax-test-release
/tmp/geojax-test-release/bin/python -m pip install --upgrade pip
/tmp/geojax-test-release/bin/python -m pip install \
  --index-url https://test.pypi.org/simple/ --no-deps geojax==0.1.0
cd /tmp
/tmp/geojax-test-release/bin/python -c \
  'from importlib.metadata import version; import geojax; print(version("geojax"))'
```

## Production

After the TestPyPI artifact has been checked, upload the unchanged files:

```bash
python -m twine upload dist/0.1.0/*
```

Verify installation from PyPI, then commit the release metadata, create the
signed or annotated tag `v0.1.0`, and push the commit and tag to GitHub. Create
a GitHub release from the changelog entry. After the first production upload,
replace the account-scoped token with a token restricted to the GeoJAX project.
