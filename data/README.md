# Data

**Raw data is never committed.** `data/raw/` and `data/processed/` are ignored by git; only this README is tracked.

## Source

Daily station observations ("Tageswerte, Klima") of the German Weather Service (DWD), Climate Data Center:
https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/daily/kl/

The files are the ones shipped with the source project
[amk0098/heatrisk-deutschland](https://github.com/amk0098/heatrisk-deutschland), commit `9974c0e`, folder `data/raw/`.
Licence: DWD open data, use with source attribution ("Quelle: Deutscher Wetterdienst").

## Setup

Copy the five files into `data/raw/` of this repository:

```bash
git clone https://github.com/amk0098/heatrisk-deutschland ../heatrisk-deutschland
mkdir -p data/raw
cp ../heatrisk-deutschland/data/raw/*.txt data/raw/
```

Then `pytest` checks that the data matches the audited version (`tests/test_data.py`).

## Files

| File | City | DWD station ID | First day in file | Used from |
|---|---|---:|---|---|
| produkt_klima_tag_Berlin.txt | Berlin | 400 | 1951-01-01 | 1990-01-01 |
| produkt_klima_tag_Frankfurt.txt | Frankfurt | 1420 | 1935-07-01 | 1990-01-01 |
| produkt_klima_tag_Hamburg.txt | Hamburg | 1975 | 1936-01-01 | 1990-01-01 |
| produkt_klima_tag_koeln.txt | Köln | 2667 | 1957-09-01 | 1990-01-01 |
| produkt_klima_tag_Muenchen.txt | München | 1262 | 1992-05-17 | 1992-05-17 |

All files end on 2025-12-31.

## Schema

`;`-separated text, one row per day. Columns used: `MESS_DATUM` (YYYYMMDD), `TMK` mean temperature (°C),
`TXK` maximum temperature (°C), `TNK` minimum temperature (°C), `VPM` vapour pressure (hPa),
`UPM` relative humidity (%). Full column list: `DWD_HEADER_REFERENCE.txt` in the source project.

## Missing values

DWD marks missing values as `-999`; they are read as NaN (`src/data.py`).
Rows without a target (missing TXK tomorrow) or without lag features (first 3 days per city) are dropped.
Remaining NaNs in the features (at most 26 rows per split, mostly VPM/UPM) are imputed with training-period means
before training. Details: `docs/data_audit.md`.
