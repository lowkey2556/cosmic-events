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
- **Date range for every star:** **2021-01-01 to 2023-12-31** (JD 2459215.5 to 2460310.5).
  Three years keeps the files manageable, gives at least a few Mira cycles, and still yields
  hundreds to tens of thousands of rows per star. If a file is enormous (T CrB and SS Cyg can be),
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

¹ T CrB's VSX type is a recurrent nova (`NR`), which is **not** in your label map. It will be
unlabelled: kept for K-Means but excluded from training. If you want it labelled, add `NR` to
`anomalous` in `config/label_map.yaml` **before** running, and say so in the report.

## Form settings (AAVSO website → Data → Data Download / "Search and download data")

For each star:

1. **Star name:** exactly as in the table (e.g. `SS Cyg`).
2. **Start date:** `2021-01-01` · **End date:** `2023-12-31`.
3. **Bands:** all bands. **Observation types:** all (Visual, CCD, DSLR, PEP…).
4. **Include fainter-than (non-detection) observations:** yes.
5. **Include discrepant observations:** no, if the form offers this option.
6. **Output format:** comma-separated (CSV) **with a header row**.

Any file names work (the pipeline identifies stars by `auid`, not by file name), and one file may hold several stars. Overlapping files are fine: duplicates are removed in Step 2.

**Check the header of the first file before downloading the rest.** It must be:

```
#,target,auid,jd,mag,uncertainty,fainterthan,band,type,observer,airmass,transformed,comp1_c,cmag,comp2_k,kmag,chart,commentcode,comments
```

Step 1 checks this and stops with a clear error if it differs. If your export has different
columns, paste the header to me and I'll adapt the reader. Please don't edit the files by hand.

## Where to put the files in Google Drive

```
MyDrive/
└── cosmic/
    ├── raw/                 <- the 44 CSV files go here
    ├── labels/              <- leave empty: Step 1 fetches VSX rows and writes vsx_labels.csv here
    ├── cache/               <- created automatically (Hadoop/Spark/Hive tarballs)
    └── review2/             <- created automatically (all Review 2 outputs)
```

## Labels (VSX)

Step 1 queries VizieR `B/vsx/vsx` with astroquery, **only for the stars found in your files**,
one request per second. It saves `labels/vsx_labels.csv` with columns `auid,name,ra,dec,type`.
If that fails (for example, VizieR is down), export the rows by hand from
<https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=B/vsx/vsx>: search each name and keep the
columns Name, RAJ2000, DEJ2000 and Type. Save them as `labels/vsx_labels.csv` with the header
`name,ra,dec,type` (RA/Dec in decimal degrees). The pipeline will then match on name.
