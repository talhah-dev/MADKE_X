# MADKE-X paper: what is left

PDF: `overleaf/main.pdf` (55 pages). Last updated: 13 November 2026.

## 1. Needs the real Kaggle results (code version 1.2.0)

The old Kaggle results were lost when the session restarted, and they used the old code anyway.
Run again with the new `MADKE_X_upload.zip`, then download `results/live` as a zip.

| # | Page | Place in the paper | What to add |
|---|------|--------------------|-------------|
| 1 | 1 | Abstract, last sentence | The real live result |
| 2 | 3 | Introduction | One line with the live result |
| 3 | 43–44 | Section 8.2 Models and infrastructure | Model, number of questions, run time |
| 4 | 44 | **Table 15 (live results)** | All cells are "--" now (main gap) |
| 5 | 44 | Section 8.3 Results | Replace the "pending" text with real numbers |
| 6 | 45–46 | Section 8.5 Analyses | Analysis of the live records |
| 7 | 47 | Discussion | What the live runs showed |
| 8 | 49 | Conclusion, last sentence | The real result |

Table 15 fills itself: `python paper_assets.py`, then rebuild the PDF. Never type the numbers by hand.

## 2. Needs the authors

| # | Page | Item | Status |
|---|------|------|--------|
| 9 | 50 | Author roles (CRediT) | Red note: confirm who did what |
| 10 | 50 | AI declaration | Removed; write your own before submission (the journal requires it) |
| 11 | 50 | Funding statement | Not written yet |
| 12 | 1 | Author order | Now: Muhammad Ali, Muhammad Talha. Swap if needed |

## 3. Done

- GitHub link added on page 50: https://github.com/talhah-dev/MADKE_X
- Pages 1–42 and 51–55 (references) are finished, apart from the items above.

## 4. Optional

- Length: 55 pages; the teacher's target was about 40. Can be shortened if the teacher asks.
