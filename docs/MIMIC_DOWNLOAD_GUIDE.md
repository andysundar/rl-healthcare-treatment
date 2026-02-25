# MIMIC-III Dataset Download Guide

## Prerequisites
- Active institutional email address
- Institutional affiliation (university, hospital, research organisation)
- Completion of the required CITI training course

---

## Tables Required by This Project

Place all CSV files in `data/raw/mimic-iii/` (uncompressed `.csv`, **not** `.csv.gz`).

### Base run (always required)

| File | Size (approx.) | Purpose |
|---|---|---|
| `PATIENTS.csv` | < 1 MB | Patient demographics, cohort selection |
| `ADMISSIONS.csv` | ~10 MB | Admission records, LOS |
| `DIAGNOSES_ICD.csv` | ~30 MB | ICD-9 codes — diabetes cohort filter |
| `LABEVENTS.csv` | ~1.7 GB | Glucose and lab measurements |
| `PRESCRIPTIONS.csv` | ~100 MB | Insulin orders; also used by `--use-med-history` |

### Optional — vital signs (`--use-vitals`)

| File | Size (approx.) | Purpose |
|---|---|---|
| `CHARTEVENTS.csv` | ~35 GB | HR, SBP, RR, SpO2 from bedside monitors |

> Without `CHARTEVENTS.csv`, omit the `--use-vitals` flag. The pipeline runs normally on the 5 base tables.

### Expected directory layout

```
data/
└── raw/
    └── mimic-iii/
        ├── PATIENTS.csv
        ├── ADMISSIONS.csv
        ├── DIAGNOSES_ICD.csv
        ├── LABEVENTS.csv
        ├── PRESCRIPTIONS.csv
        └── CHARTEVENTS.csv      ← optional, needed for --use-vitals
```

---

## Step-by-Step Access Process

### 1. Complete Required Training

Before accessing MIMIC-III, complete the CITI "Data or Specimens Only Research" course:

1. Go to <https://about.citiprogram.org/>
2. Register for an account
3. Complete the course: **"Data or Specimens Only Research"**
4. Download your completion certificate

### 2. Create a PhysioNet Account

1. Visit <https://physionet.org/>
2. Click **Register** in the top right
3. Fill out the form with your institutional email, professional affiliation, and research interests

### 3. Request Access to MIMIC-III

1. Go to <https://physionet.org/content/mimiciii/1.4/>
2. Click **Request Access**
3. You will need to:
   - Agree to the Data Use Agreement
   - Upload your CITI training certificate
   - Describe your research purpose
   - Obtain institutional approval (may require supervisor signature)

### 4. Wait for Approval

Approval typically takes 1–3 business days. You will receive an email notification when approved.

### 5. Download the Dataset

#### Option A: `wget` direct download

```bash
# Download only the tables needed for the base pipeline (faster)
MIMIC_URL="https://physionet.org/files/mimiciii/1.4"
OUT="data/raw/mimic-iii"
mkdir -p "$OUT"

for TABLE in PATIENTS ADMISSIONS DIAGNOSES_ICD LABEVENTS PRESCRIPTIONS; do
    wget -N -c --user YOUR_USERNAME --ask-password \
        "$MIMIC_URL/${TABLE}.csv.gz" -O "$OUT/${TABLE}.csv.gz"
    gzip -d "$OUT/${TABLE}.csv.gz"   # decompress to plain .csv
done

# Optional — only needed for --use-vitals (~35 GB)
wget -N -c --user YOUR_USERNAME --ask-password \
    "$MIMIC_URL/CHARTEVENTS.csv.gz" -O "$OUT/CHARTEVENTS.csv.gz"
gzip -d "$OUT/CHARTEVENTS.csv.gz"
```

> The pipeline loader reads plain `.csv` files. Always decompress with `gzip -d` after download.

#### Option B: Full mirror (all tables)

```bash
wget -r -N -c -np --user YOUR_USERNAME --ask-password \
    https://physionet.org/files/mimiciii/1.4/
```

Then move the required files into `data/raw/mimic-iii/` and decompress them.

#### Option C: Google BigQuery (no local download)

MIMIC-III is available on Google BigQuery at `physionet-data.mimiciii_clinical`.
This is useful for exploratory queries but **the project pipeline reads local CSV files** — export
the required tables to CSV before running the pipeline:

```sql
-- Example: export LABEVENTS for diabetic patients
SELECT * FROM `physionet-data.mimiciii_clinical.labevents`
WHERE subject_id IN (
    SELECT DISTINCT subject_id FROM `physionet-data.mimiciii_clinical.diagnoses_icd`
    WHERE icd9_code LIKE '250%'
)
```

Save each exported table as `TABLENAME.csv` in `data/raw/mimic-iii/`.

#### Option D: AWS S3

```bash
pip install awscli
aws configure   # enter your AWS credentials

# Download from S3 (requires credentialed PhysioNet access)
aws s3 sync s3://mimic-iii-physionet/ data/raw/mimic-iii/
```

After download, decompress any `.csv.gz` files: `gzip -d data/raw/mimic-iii/*.csv.gz`

---

## Dataset Size Summary

| Table | Compressed | Uncompressed |
|---|---|---|
| PATIENTS | < 1 MB | < 1 MB |
| ADMISSIONS | ~3 MB | ~10 MB |
| DIAGNOSES_ICD | ~9 MB | ~30 MB |
| LABEVENTS | ~1.7 GB | ~3.5 GB |
| PRESCRIPTIONS | ~30 MB | ~100 MB |
| CHARTEVENTS | ~3.5 GB | ~35 GB |
| Full dataset | ~7 GB | ~60+ GB |

---

## Privacy and Ethics

- Data is de-identified but remains protected health information
- Use only for the research purpose stated in your access application
- Do not attempt to re-identify patients
- Do not share data files with users who do not have independent PhysioNet access
- Comply with the MIMIC-III Data Use Agreement at all times

---

## Quick Start Without Full MIMIC-III

### MIMIC-III Demo (no credentials required)

PhysioNet provides a freely available demo subset of 100 patients:

- URL: <https://physionet.org/content/mimiciii-demo/1.4/>
- Contains the same table structure as the full dataset
- Suitable for code validation and pipeline testing

Download the same tables listed above from the demo URL and point `--mimic-dir` at the demo directory.

Note: the demo contains `NOTEEVENTS`, `CHARTEVENTS`, and other tables but some may be truncated.
Behaviour will match the full pipeline — use `--use-sample --sample-size 100` for realistic runtime.

### Synthetic data (zero setup)

All pipeline features work without any MIMIC-III data using the built-in synthetic generator:

```bash
./quick_start.sh --mode 4   # extended 16-dim state with synthetic vitals
./quick_start.sh --mode 5   # full pipeline, no real data needed
```

---

## Useful Resources

- MIMIC-III documentation: <https://mimic.mit.edu/docs/iii/>
- MIT-LCP code repository: <https://github.com/MIT-LCP/mimic-code>
- Tutorials and notebooks: <https://mimic.mit.edu/docs/iii/tutorials/>
- PhysioNet credentialing: <https://physionet.org/settings/credentialing/>
