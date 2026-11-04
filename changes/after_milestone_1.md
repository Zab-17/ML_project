# After Milestone 1: from Greater Cairo to the Zillow file

## What we submitted

Phase 1, "Fair Price in a Moving Market: Predicting Residential Prices in Greater Cairo". It proposed predicting residential asking prices in Greater Cairo from Egyptian listing datasets on Kaggle. After keeping only sale listings in Cairo and Giza and trimming the extreme 1%, about 15,600 listings were left.

## What the professor said

- On the data: *"your chosen dataset's size is very small … You need to find a larger dataset or change the regional scoping"*.
- On the report: it was *"not discussing the datasets in depth or the limitations of each, but rather iterating the phases that I already shared on the course outlines"*.
- He then set a minimum dataset size, with real named columns rather than bare identifiers. Our notes from the time record 300,000 rows. The later King County notes cite 500,000 rows and 50 columns, so check the course page before quoting either.
- The Milestone 3 report also lists two comments that applied to both Phase 1 and Phase 2: AI use was not declared, and references were not cited.

## What we changed

1. **The dataset.** We first checked whether Egypt could meet the minimum. No Egyptian dataset has more than 64,106 usable rows, and the whole live inventory of Egypt's largest property portal is about 279,500 listings. Egypt's national open-data portal does not resolve, and there is no public register of transactions. So the project moved to the Zillow Prize 2017 properties file: 2,985,217 parcels and 58 columns, covering Los Angeles, Orange and Ventura counties in California. The label is `taxvaluedollarcnt`, the county's assessed value of the parcel.
2. **The limits of the new data**, which every later report must state:
   - The label is an assessed value, not a sale price. The file has no sale prices at all.
   - The licence allows use only during the Zillow Prize competition, which closed on 10 January 2018. The professor has not been told yet.
3. **Phase 2 was built on the Zillow file.** It kept 2,950,951 rows, and 12 raw features became 47 encoded columns.
4. **Not fixed until Milestone 3:** the AI statement and the citations. Both were marked down again in Phase 2.

## Where the Milestone 1 work is now

On the `greater-cairo` branch: the Phase 1 report (`report/phase1_report/`) and the Egyptian developer tables (`archive/egypt-phase1/`). That work covers 225 developers, 1,090 projects and a compound lookup. None of it is used by the Zillow project.
