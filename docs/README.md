# docs/

Put the sample documents here. This folder is input only: the pipeline reads these
files and never changes them.

## What to add

- Banking documents of the kinds the chatbot should answer from: policies, circulars,
  annual reports, process manuals.
- English-language documents only, for now. Where a document is published in two
  languages, take the English file, and avoid files that mix both in one PDF.
- PDF is the main format. Include some documents with tables, charts, flowcharts and
  scanned pages, because those are what this prototype is meant to handle.
- Start with 2 or 3 documents. Parsing and embedding are slow on CPU, so the full set
  of 10 to 20 comes later.

## What must not go here

Only public or sanitised documents. Nothing with real customer data: account numbers,
PAN, Aadhaar, or names alongside financial details. If a file looks like it holds real
customer data, ingestion stops and asks rather than processing it.

## Information needed about each document

Every answer cites a document title, and conflicting documents are shown with their
effective dates. That needs, for each file:

- title (as it should appear in citations)
- version
- effective date
- access group (default `all`)

These come from `manifest.csv` in the project folder (one level up from here), one row
per file: `file,doc_title,version,effective_date,access_group`. `file` is the path
below this folder. A document that is not listed gets its file name as its title, no
version, no date and the group `all`. For the RBI documents the manifest was built
from `RBI_Documents_Index.xlsx` with `python scripts/build_manifest.py`.

## Good to know

- A document is identified by where its file sits in this folder, and its content by
  its SHA-256 hash. An unchanged file is skipped on the next run, and an edited file
  replaces what was indexed from it before.
- Keep file names stable. Rename or move a file only if you mean it to be treated as
  new. Run `python -m docqa.ingest docs --prune` afterwards to remove the old entry.
- Only PDFs are ingested. Other files in this folder are ignored.
