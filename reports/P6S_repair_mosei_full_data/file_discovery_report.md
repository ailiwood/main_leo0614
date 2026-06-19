# P6S-Repair MOSEI Full Data Discovery

## Result: FULL MOSEI FOUND

| Item | Status | Detail |
|------|:------:|--------|
| Official labels | FOUND | `data/CMU-MOSEI/CMU-MOSEI-20230514T151450Z-001/CMU-MOSEI/Labels/` |
| Train split | 16,282 | official |
| Val split | 1,862 | official |
| Test split | 4,655 | official |
| Total | 22,788 | matches CMU-MOSEI (~22,856) |
| Raw text | YES | in CSV, usable for RoBERTa |
| CSD files | 7 files, 29.5GB | COVAREP/OpenFace2/WordVectors |
| label.csv generated | `data/mosei/label.csv` | 22,788 segments, 3.1MB |

## Previous 3293 Subset Clarified

The 3293-segment CSD subset was the aligned computational sequence — a SMALL SUBSET of full MOSEI. The COMPLETE dataset exists at `data/CMU-MOSEI/`.
