# Versions and releases

The pre-refactor code is the **1.0 baseline**, at commit
`2916774285399ea9d04026c65a828e9c1aef9923`. This names the historical source baseline;
it does not imply that a 1.0 package was published to PyPI.

**2.0.0** is the first stable release of the package-boundary refactor and the first
PyPI release. Release tags identify the source used to build the published artifacts.

## Main between releases

Main may contain commits that do not belong to a published release. Use the Git commit
SHA to identify an exact source checkout; the development version alone does not uniquely
identify intermediate commits. Record both in experiment provenance:

```bash
python -c 'import fovi; print(fovi.__version__)'
git rev-parse HEAD
git status --short
```

`fovi/_version.py` is the version source of truth. Package metadata and the documentation
read it directly. Reserve `vMAJOR.MINOR.PATCH` Git tags for releases, not routine commits.
After a stable release, advance main to a development version for the next intended release.
An installed wheel records its package version; source experiments also need their commit
SHA and any uncommitted changes. Do not publish different artifacts under the same version.

## Publishing to PyPI

`.github/workflows/publish.yml` runs on `v*` tag pushes. It calls the package CI
workflow to run the CPU regression tests, build the wheel and sdist, validate metadata,
and check the installed wheel outside the source checkout. Publishing requires a stable
version matching the tag and a commit already merged into `main`. The publish job uploads
those checked artifacts through PyPI Trusted Publishing.

Configure a GitHub Trusted Publisher in the PyPI project's settings, or a pending
publisher at <https://pypi.org/manage/account/publishing/> for the first release:

| Field | Value |
| --- | --- |
| PyPI project name | `fovi` |
| GitHub owner | `nblauch` |
| Repository | `fovi` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Once the publisher is configured, set `fovi/_version.py` to the intended stable version
and merge that change with passing CI. For the first release, tag that merged commit
`v2.0.0` and push the tag. The workflow rejects development versions and mismatched tags.
After publication, verify `pip install fovi`, `pip install 'fovi[models]'`, and
`pip install 'fovi[all]'` in clean environments, then advance main's development version.

FFCV-SSL remains a manually installed prerequisite for the built-in loaders. Its pinned
Git requirement is outside the package metadata, so it does not prevent publication.
See [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/).

## Change history

### 2.0.0

- Separate sensing/KNN, model, and training dependency boundaries within one distribution.
- Add `models`, `training`, and `all` extras. Base installs include CuPy and
  Warp. `all` includes every declared optional dependency; FFCV is installed manually.
- Move complete networks and inference loading to `fovi.models`, and training to
  `fovi.training`. Remove old model import aliases; retain training compatibility imports.
- Restore pretrained models without constructing a trainer, importing FFCV, or requiring
  research storage environment variables.
- Restore DINOv3 checkpoints across known Transformers block layouts, preserving LoRA
  parameter names and strict loading checks without changing Hub artifacts.

### 1.0 baseline

The monolithic pre-refactor source, including sensing, models, training, and Hub loading.
