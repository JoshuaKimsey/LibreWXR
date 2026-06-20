# Contributing to LibreWXR

Thanks for your interest in contributing. LibreWXR is an open project and
contributions are welcome, whether that is a new radar or model source, a bug
fix, documentation, or a feature.

Please read the **Licensing of contributions** section below before you open a
pull request. It is short, but it matters.

## Getting started

LibreWXR is pure Python with no GDAL or system geospatial dependencies. To set
up a development environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m librewxr.main
```

Run the test suite with `pytest` (markers are defined in `pyproject.toml`, e.g.
`pytest -m sources`). See [`README.md`](README.md) for the architecture
overview and [`docs/adding-a-source.md`](docs/adding-a-source.md) for a full
walkthrough of adding a new radar or NWP source.

A few conventions:

- Source files carry an SPDX header: `# SPDX-License-Identifier: AGPL-3.0-or-later`
  and a copyright line `# Copyright (C) 2026 Joshua Kimsey`.
- Commit messages use the imperative mood and stay concise (e.g. "Add
  precipitation motion arrows").
- Keep new code in the style of the surrounding code.

## Developer Certificate of Origin (sign-off)

Every commit must be signed off to certify that you have the right to submit it
under the project's license. This is the [Developer Certificate of
Origin](https://developercertificate.org/) (DCO): a lightweight statement of
provenance, not a copyright transfer.

Add a sign-off line to each commit with `git commit -s`, which appends:

```
Signed-off-by: Your Name <your.email@example.com>
```

The name and email must be real and must match your commit author identity.

## Licensing of contributions

LibreWXR is distributed under the **GNU Affero General Public License v3.0**
(AGPL-3.0-or-later), and it will always remain free and open under that license
for everyone.

To keep the project sustainable, a separate **commercial license** is also
offered to organisations whose use is incompatible with the AGPL's obligations
(see the *Commercial licensing* section of the [README](README.md)). For that
to be possible, the maintainer needs the right to license the *entire* codebase,
including contributions, under both the AGPL and commercial terms.

Therefore, **by submitting a contribution to LibreWXR (for example via a pull
request), you agree that:**

1. You are the author of the contribution and have the right to submit it, or
   you have the necessary rights from the actual author to submit it on their
   behalf.
2. You **retain copyright** in your contribution.
3. You grant Joshua Kimsey (the maintainer and copyright holder of LibreWXR) a
   perpetual, worldwide, non-exclusive, royalty-free, irrevocable license to
   use, reproduce, modify, distribute, and **sublicense** your contribution as
   part of LibreWXR, **under both the AGPL-3.0-or-later and under separate
   commercial license terms**, with the right to relicense it as part of the
   project.

This means you keep ownership of your work, the project stays AGPL for the
community forever, and the maintainer retains the ability to offer commercial
licenses that fund continued development. Your contribution itself is never
taken away from you, and is always available to you and everyone else under the
AGPL.

If you do not agree to these terms, please do not submit a contribution.

## Questions

For contribution questions, open an issue or discussion on
[GitHub](https://github.com/JoshuaKimsey/LibreWXR). For commercial licensing,
contact <jkimsey@proton.me>.
