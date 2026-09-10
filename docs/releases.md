# Versions and releases

The pre-refactor code is the **1.0 baseline**, at commit
`2916774285399ea9d04026c65a828e9c1aef9923`. This names the historical source baseline;
it does not imply that a 1.0 package was published to PyPI.

The package-boundary refactor is under development as **2.0.0.dev0**. The first stable
**2.0.0** release will coincide with PyPI availability. Merging the refactor does not
constitute that release.

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

## PyPI follow-up

The refactor, model parity checks, and migration documentation merge first. Publication
is a separate follow-up covering:

- Distribution of the pinned FFCV-SSL dependency. Its current Git URL is rejected by
  PyPI, including when declared in an extra. Preserve the required training behavior and
  `fovi[all] == fovi[models,training,ffcv]` when choosing its distribution mechanism.
- Package name availability, release metadata, and wheel/sdist installation checks in
  environments with only the selected extras.
- A trusted publisher workflow, a release tag, and publication of 2.0.0.
- Installation instructions using the published package instead of a source checkout.

See [setuptools' direct dependency restrictions](https://setuptools.pypa.io/en/stable/userguide/dependency_management.html#direct-url-dependencies)
and [PyPI trusted publishing](https://docs.pypi.org/trusted-publishers/).

## Change history

### Unreleased — 2.0.0.dev0

- Separate sensing/KNN, model, and training dependency boundaries within one distribution.
- Add `models`, `training`, `ffcv`, and `all` extras. Base installs include CuPy and
  Warp; the historical `warp` extra is redundant. `all` includes every optional dependency.
- Move complete networks and inference loading to `fovi.models`, and training to
  `fovi.training`, with compatibility imports at the old paths.
- Restore pretrained models without constructing a trainer, importing FFCV, or requiring
  research storage environment variables.
- Restore DINOv3 checkpoints across known Transformers block layouts, preserving LoRA
  parameter names and strict loading checks without changing Hub artifacts.

### 1.0 baseline

The monolithic pre-refactor source, including sensing, models, training, and Hub loading.
