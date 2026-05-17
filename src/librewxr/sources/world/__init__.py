# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Joshua Kimsey
"""Global-coverage sources.

Sources whose data is not scoped to any single region live here — for
example, the ECMWF IFS 9 km global precipitation grid (the NWP chain's
final / lowest-priority layer).

Named ``world`` rather than ``global`` because ``global`` is a Python
keyword and using it as a package name forces ``importlib`` workarounds
for any future explicit import.
"""
