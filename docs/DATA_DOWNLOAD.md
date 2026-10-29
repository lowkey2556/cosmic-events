# Getting the data (manual, no scraping)

## Why manual

As of October 2026, I could not find an **official, documented** AAVSO API for bulk
downloads from the AAVSO International Database (AID). AAVSO points users to the
**Data Download** web form (free AAVSO login). AAVSO staff have said in a forum that there is
no standard API, and no newer documentation contradicts that. So this project has **no
downloader** and never scrapes aavso.org. Each star is downloaded by hand from the official
form.

## What to download

- **Stars:** the 44 stars below (also in `config/stars.yaml`). 22 are eruptive or cataclysmic and
  22 are regular periodic, all heavily observed, so observers overlap.
- **Date ranges:** the data arrives in rounds: January 2025 (the existing files), then
  February–December 2025, then 2024, then 2026, then 2023. Every round's files go directly
  into the same `data/` folder (see below). If a file is enormous (T CrB and SS Cyg can be),
  that's fine: Step 2 thins dense CCD time series.

| # | Eruptive / cataclysmic | | # | Regular periodic |
|---|---|---|---|---|
| 1 | SS Cyg | | 23 | omi Cet (Mira) |
| 2 | U Gem | | 24 | chi Cyg |
| 3 | Z Cam | | 25 | R Leo |
| 4 | RU Peg | | 26 | R Hya |
| 5 | T CrB ¹ | | 27 | bet Per (Algol) |
| 6 | SU UMa | | 28 | RR Lyr |
| 7 | AM Her | | 29 | del Cep |
| 8 | YZ Cnc | | 30 | eta Aql |
| 9 | CH Cyg | | 31 | W UMa |
| 10 | R CrB | | 32 | R Cas |
| 11 | VW Hyi | | 33 | R Aql |
| 12 | AH Her | | 34 | U Ori |
| 13 | EX Dra | | 35 | T Cep |
| 14 | RX And | | 36 | R Cyg |
| 15 | SY Cnc | | 37 | R Ser |
| 16 | BZ UMa | | 38 | R UMa |
| 17 | V1504 Cyg | | 39 | U Her |
| 18 | T Leo | | 40 | SS Vir |
| 19 | AB Dra | | 41 | W Hya |
| 20 | TT Ari | | 42 | RX Boo |
| 21 | Z And | | 43 | g Her |
| 22 | AG Dra | | 44 | Z UMa |

¹ T CrB's VSX type is a recurrent nova (`NR`). `NR` is in the `anomalous` list of
`config/label_map.yaml`, so T CrB is labelled anomalous.

## Form settings (AAVSO website → Data → Data Download / "Search and download data")

For each star:

1. **Star name:** exactly as in the table (e.g. `SS Cyg`).
2. **Start / end date:** the dates of the current download round.
3. **Bands:** all bands. **Observation types:** all (Visual, CCD, DSLR, PEP…).
4. **Include fainter-than (non-detection) observations:** yes.
5. **Include discrepant observations:** no, if the form offers this option.
6. **Output format:** comma-separated (CSV) **with a header row**.

Any file names work (the pipeline identifies stars by `auid`, not by file name), and one file may hold several stars. Overlapping files are fine: duplicates are removed in Step 2.

**Check the header of the first file before downloading the rest.** It must be:

```
#,target,auid,jd,mag,uncertainty,fainterthan,band,type,observer,airmass,transformed,comp1_c,cmag,comp2_k,kmag,chart,commentcode,comments
```

The dataset audit and Step 1 check the header of **every** file. A file whose header differs is
**skipped** (never re-mapped by guesswork) and listed in `audit_skipped_files.csv` /
`skipped_files.csv`. Please don't edit the files by hand.

## Where to put the files in Google Drive

Every path is derived from one setting, **PROJECT_ROOT** = `/content/drive/MyDrive/cosmic_roja`
(`datasets.real.project_root` in `config/settings.yaml`; the notebook sets it). Only the `*.csv`
files **directly inside `data/`** are read; ZIPs and other files there are ignored. CSVs in
sub-folders of `data/`, or two files whose names differ only in upper/lower case, stop the run with
a clear error (they would be skipped, or overwrite each other in HDFS).

```
MyDrive/
└── cosmic_roja/                 <- PROJECT_ROOT: every path below is derived from it
    ├── data/                    <- ALL the AAVSO CSVs, directly in this folder (no sub-folders)
    ├── work/
    │   ├── labels/              <-   vsx_labels.csv (star types from VSX, fetched when missing)
    │   └── cache/               <-   Hadoop/Spark/Hive downloads (saves time after a reset)
    └── v3_5GB/                  <- V3_ROOT: created automatically; EVERYTHING the pipeline writes
        ├── ztf_raw/             <-   the 117 ZTF files + MANIFEST.csv (downloaded once, MD5-verified)
        ├── clean/               <-   copy of the cleaned ZTF table
        ├── work/                <-   window tensors, splits, models, predictions, data-flow records
        ├── outputs/             <-   every table, figures/, COMPARISON.md, DATA_FLOW.txt
        └── dashboard_data/      <-   dashboard tables + DATA_DICTIONARY.md
```

The old `MyDrive/cosmic` folder (renamed `cosmic_OLD_do_not_use`) is never read or written:
the notebook stops if any path points there.

## Labels (VSX)

Step 1 queries VizieR `B/vsx/vsx` with astroquery (main server, then its mirrors; NASA HEASARC's copy `aavsovsx` when
no VizieR server answers), **only for the stars found in your files**, one request per second. It saves `work/labels/vsx_labels.csv` (under PROJECT_ROOT) with columns `auid,name,ra,dec,type`.
If that fails (for example, VizieR is down), export the rows by hand from
<https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=B/vsx/vsx>: search each name and keep the
columns Name, RAJ2000, DEJ2000 and Type. Save them as `work/labels/vsx_labels.csv` with the header
`name,ra,dec,type` (RA/Dec in decimal degrees). The pipeline will then match on name.
