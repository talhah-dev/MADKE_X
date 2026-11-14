# MADKE-X paper: what is left

PDF: `overleaf/main.pdf` (55 pages). Last updated: 14 November 2026.

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

## 5. Teacher's request (14 November 2026): present the paper as a new method, not a small improvement

The teacher asked that the title, abstract, contributions and conclusion look strong and new,
because the editor reads these four parts first.

### Done

| Part | Page | What was changed |
|------|------|------------------|
| Title | 1 | New title without "MADKE": "Governing the Evidence Path: A Traceable and Failure-Aware Multi-Agent Debate Framework for Knowledge-Intensive Question Answering" |
| Abstract | 1 | Opens with a general problem; MADKE-X introduced as "a governed debate framework"; 46.5% token saving added |
| Contributions | 3–4 | Now 6 items; the new framework comes first, the inspection of MADKE moved to fifth |
| Conclusion | 49 | Presents MADKE-X as a new framework; says it can be used around other debate systems |

Old versions are saved in `Extra_Files/sections_backup_before_reframe/`.

### Still left (ask the teacher)

| # | Item | Note |
|---|------|------|
| T1 | Word "groundbreaking" / "first ever" | Not used: reviewers dislike hype and a "first" claim cannot be proven. Strong factual wording was used instead. Ask the teacher if this is acceptable. |
| T2 | System name "MADKE-X" | Still used inside the paper; the name itself points to MADKE. Renaming means changing the paper, figures and code. Ask the teacher if a new name is wanted, and which. |
| T3 | MADKE named as the base system | Stays: the paper cites and compares against MADKE, so it cannot be hidden. |
| T4 | Teacher's review of the four rewritten parts | Send the new PDF (pages 1, 3–4, 49) to the teacher for approval. |
| T5 | Free GPU / compressed model question | Message sent to the teacher: keep it stated in the paper (current), or pay for a server and run the full model? Waiting for the answer. |
| T6 | GPT-3.5 / GPT-4 Turbo shut down (23 Oct 2026) | Optional: add one sentence in Section 8.1 (page 43) saying these models can no longer be run. |
