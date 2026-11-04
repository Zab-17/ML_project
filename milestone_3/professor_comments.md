# Milestone 3: what we did about the professor's comments

Both earlier submissions lost marks. Before running a single model, we rebuilt the dataset from the raw file to answer every comment. The table pairs each comment with what changed and the number behind the change. The report carries the same table in Section 2, "Response to the Phase 1 and Phase 2 Feedback".

| Comment (phase) | What we changed | Evidence |
|---|---|---|
| Dataset too small; limits of the data not discussed (Phase 1) | The project moved to the Zillow file in Phase 2. The report states the limits of its label in the Limitations section. | 2,985,217 parcels in the raw file; 2,859,302 homes kept. |
| What is FIPS? (Phase 2) | FIPS is the Federal Information Processing Standard county code. It is now read as a county name (6037 Los Angeles, 6059 Orange, 6111 Ventura). Its copy, `regionidcounty`, was removed with a measured reason. | Cramér's V of 1.000 between the two columns. The county itself was later dropped at 0.998 with the county land-use code. |
| Raw latitude and longitude used as inputs (Phase 2) | The coordinates became miles north and miles east of the median training home, because a degree of longitude here is 57 miles and a degree of latitude is 69. Both were then dropped by the pair rule, because ZIP code carries the same information. | 0.983 and 0.988 with ZIP code. ZIP code is 0.490 with the label, against 0.119 and 0.013. |
| City, county and region dropped (Phase 2) | All five location columns were brought back and measured. ZIP code, neighbourhood and census tract are kept as one-hot groups. City and county were dropped by the measured pair rule, not by assumption. | 325, 98 and 16 model columns. City is 0.921 with ZIP code and weaker with the label (0.391 against 0.490). |
| No distribution analysis; log taken with no analysis (Phase 2) | A figure shows the label and the living area before and after the log, with the skew of each. | The label's skew of 18.9 falls to −0.12. The living area's skew of 262.1 falls to 0.39. |
| A decision with no number, for every feature (Phase 2) | Each of the 58 raw columns has a written decision with its blank share, its number of values, its strength with the label and its strongest partner. | 17 features kept, 6 dropped by the pair rule and 3 dropped by name as leaks. |
| Text feature dropped as hard to encode (the professor's rule on text) | Zoning is one-hot encoded. Codes under 0.1% of training homes are grouped together. | Raw zoning is 0.411 with the label and only 0.505 with ZIP code; 120 model columns. |
| AI use not declared (Phases 1 and 2) | The report has a "Use of AI" section, and every code file opens with the same statement. | Every script carries the statement. |
| References not cited (Phases 1 and 2) | Every method, library and data source is cited where it is used. | 28 references in `report/references.bib`. |

## Still open

- **The licence.** The Zillow data may be used only during a competition that closed on 10 January 2018. The professor has not been told.
- **The label.** It is a 2016 assessed value, not a sale price. Under California's Proposition 13, an assessed value is reset only when the home sells. The report states this limit; no fix can remove it.

## Where the detail is

- All 12 fixes in full: [`changes/after_milestone_2.md`](../changes/after_milestone_2.md). Appendix A of the report summarizes them.
- The column-by-column decisions: `zillow/zillow_columns.py`, with the measured numbers in `zillow/out/checks/column_check.csv`.
- The report: [`report/main.tex`](report/main.tex).
