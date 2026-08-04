# NOAA MRMS Merged Reflectivity QC Composite

NEXRAD + Canadian-radar quality-controlled composite reflectivity
operated by **NOAA NSSL / NCEP**, served as GRIB2 files at
`mrms.ncep.noaa.gov/2D/`.  Latest-file endpoint (`.latest.grib2.gz`)
updates every ~2 min; archive files are timestamped at the same
cadence and pruned after a rolling window upstream.

## Coverage

| Region   | Product path                                    |
| -------- | ----------------------------------------------- |
| `USCOMP` | `MergedReflectivityQCComposite` (CONUS+border)  |
| `CACOMP` | `MergedReflectivityQCComposite` (CONUS+border)  |
| `AKCOMP` | `ALASKA/MergedReflectivityQCComposite`          |
| `HICOMP` | `HAWAII/MergedReflectivityQCComposite`          |
| `PRCOMP` | `CARIB/MergedReflectivityQCComposite`           |
| `GUCOMP` | `GUAM/MergedReflectivityQCComposite`            |

Region defs live in `usa/radar/regions.py` (shared with IEM); MRMS does
not need IEM's ``live_dir`` / ``archive_dir`` fields but the canonical
RegionDef carries them harmlessly.

## Cadence & latency

- Native cadence: ~2 min (server publishes faster than our 10-min
  store cadence — we sample the nearest file via bisect).
- The `.latest` endpoint is used for live frames; archive lookups
  fetch the directory listing (cached 5 min, double-checked locking)
  and bisect to the nearest timestamped file.

## Format quirks

- `.grib2.gz` gzip-wrapped GRIB2.  Truncated downloads (server drops
  the connection mid-stream) raise `EOFError` on decompress and trip a
  one-shot retry inside `_fetch_and_parse`.
- The eccodes C library (via cfgrib) writes non-actionable `dataTime`
  truncation messages to stderr.  `librewxr.sources._helpers` redirects
  the C library's default-context logging to `/dev/null` once at import
  (`codes_context_set_logging`) to keep server logs clean — no fd
  games, so GRIB decodes can safely run inside `asyncio.to_thread`.
- No-data sentinel is `-999.0`; valid values are dBZ.  Resampling
  converts NaN → `-33.0` so the shared `_dbz_float_to_uint8` encoder
  maps it to 0.

## Cross-source role

The fetcher consumes MRMS as the primary North America source whenever
`na_source` is `mrms` or `mrms_fallback`.  In `mrms_fallback` mode it
also blends MSC Canada into CACOMP (`data/fetcher.py::_blend_cacomp`)
and falls back to IEM on any MRMS miss (`_try_fallback`).  Both helpers
keep working unchanged after this migration — they import
`MSCCanadaSource` / `IEMSource` directly from the new packages.

## Multi-product facade

`MRMSCompositeSource` is the registry-friendly wrapper: one instance
per active `MRMSCompositeSource` covers all enabled MRMS regions, and
internally maintains one `MRMSSource` per unique product path so
USCOMP+CACOMP share a single directory cache while AKCOMP/HICOMP/etc.
each get their own.

## License & attribution

NOAA MRMS data is in the public domain.  Attribution to NSSL and NCEP
is a courtesy — see the top-level `README.md` and `docs/coverage.md`.

## Stations

Per-region station map in `stations.py`.  USCOMP/CACOMP combine the
NEXRAD CONUS list with the ECCC Canadian network (MRMS ingests both
into the CONUS product); AKCOMP/HICOMP/PRCOMP/GUCOMP are NEXRAD-only.
Feeds `data/coverage.py` via the `RadarSourceContribution`.
