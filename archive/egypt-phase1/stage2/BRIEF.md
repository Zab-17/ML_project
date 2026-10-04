# Stage 2 brief: developer profiles (Egypt, Greater Cairo residential)

You profile every developer in your slice file `stage2/slice_K.csv` (K = your slice number).
The rows are in priority order: the first rows are the biggest developers, so give them the most depth.
Base folder: /Users/zeyadkhaled/Desktop/machine learning/data/developers/

## Accuracy rule (the most important rule)
- Every value you write must come from a source you actually fetched. List the source in `source_ids`.
- If you cannot source a value, leave it blank. Never fill a value from your own memory. Never estimate.
- ENGLISH ONLY. No Arabic characters anywhere. Check with a regex before finishing.
- Use Egyptian pounds (EGP) for prices. Use acres for land; convert feddans with 1 feddan = 1.038 acres, and state in `notes` that you converted.

## Allowed sources
- The developer's official website and brochure PDFs
- Investor-relations pages and presentations, and Egyptian Exchange disclosures
- Reputable press: Daily News Egypt, Enterprise, Ahram Online, Zawya, Arab Finance, Egypt Independent, Forbes Middle East
- The government platform blogs.realestate.gov.eg
- Wikipedia, but only as a secondary source

## Forbidden sources
- Forbidden because their terms prohibit automated access. Never fetch them: nawy.com, propertyfinder.eg, bayut, dubizzle/olx, aqarmap, and emaarmisr.com. For Emaar Misr, use press coverage and investor-relations PDFs instead.
- Broker and SEO sites. They may be cited only as `weak`, and never as the only source for a number: gprproperty, egyprop, realestate-capsule, deedgate, sandsofwealth, realting, compoundgate, nileestate, dlleni, flatandvilla, and any broker site clone that has the developer's name in its domain.

## Politeness rules
- At most about 1 request per second per site.
- Do not log in, submit forms or download anything behind a registration wall.
- If you have Chrome browser tools, use them ONLY for pages that need JavaScript to render. Do not click anything that submits data.

## Outputs
Write these files in `stage2/`. Save them after every 5 developers, so a restart cannot lose work.

### 1. `developers_K.csv`
Columns:
`developer_id, name_en, legal_name_en, founded_year, hq_city, website, egx_ticker, parent_group, n_projects_total, n_projects_greater_cairo, units_delivered, units_delivered_year, land_bank_acres, annual_sales_egp_bn, annual_sales_year, sales_rank_note, delivery_track_record, reputation_notes, verified, last_checked, source_ids`

- `delivery_track_record`: short factual text from the sources. Examples: "delivered X units on time per 2025 IR", or "press reports delays at Y in 2024".
- `sales_rank_note`: for example, a position in The Board Consulting rankings from Daily News Egypt, with the period.
- `last_checked`: today's date, YYYY-MM-DD.

### 2. `projects_K.csv`
One row per project. Include ALL of the developer's projects, including coastal ones. The `city` column lets us filter later.

Columns:
`project_id, developer_id, project_name_en, city, district, project_category, land_area_acres, launch_year, first_delivery_year, delivery_status, unit_types, finishing_levels, facilities, avg_price_per_m2_egp, price_type, price_date, starting_price_egp, payment_plan, notes, source_ids`

- `project_id`: `DEVxxx-P01`, `DEVxxx-P02`, and so on.
- `project_category`: one of `residential`, `mixed-use`, `coastal/resort`, `commercial`, `administrative`.
- `unit_types`: pipe-separated, from this list: `apartment`, `duplex`, `penthouse`, `studio`, `townhouse`, `twin house`, `standalone villa`, `chalet`, `serviced apartment`.
- `finishing_levels`: pipe-separated, from this list: `core and shell`, `semi-finished`, `fully finished`, `furnished`.
- `facilities`: pipe-separated, lower case. Examples: `clubhouse`, `swimming pools`, `lagoon`, `sports courts`, `kids area`, `commercial area`, `underground parking`, `schools`, `hospital`, `mosque`, `gym`, `dog park`, `bbq area`, `landscape`, `golf course`, `security gates`.
- `delivery_status`: one of `delivered`, `partially delivered`, `under construction`, `launched`, `unknown`.
- `price_type`: one of these three:
  - `ir_average`: an average price per square metre published by the developer in investor relations.
  - `press_reported`: a price reported by reputable press.
  - `advertised_starting`: a "starting from" price on the developer's own marketing.
- Always fill `price_date`, the date the price was published or seen.

### 3. `sources_K.csv`
Columns:
`source_id, url, publisher, title, accessed_date, quality, notes`

- `source_id`: `SK-001`, `SK-002`, and so on, with your slice number as K.
- `quality`: one of `primary`, `secondary`, `weak`.

## Final report
Keep it short. Give:
- developers done and projects found
- how many projects have a price, split by price type
- developers you could not profile, and why
- rejected content farms
- tool failures, meaning 403 errors, JavaScript-only pages and dead sites
