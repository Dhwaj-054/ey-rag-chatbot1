# eval/

Test questions and evaluation output. Success for this prototype means accurate, cited
answers on the questions in `questions.csv`.

## questions.csv

One row per question. The header row is already there; add rows below it.

| Column | What to put |
|---|---|
| `question` | The question as a user would ask it. |
| `expected_answer` | The correct answer, written out. Quote numbers exactly, with units and reporting period. |
| `source_doc` | File name in `docs/` that holds the answer. Leave empty if the answer is in no document. |
| `page` | Page where the answer is, counted from 1 as a PDF viewer shows it (not the number printed on the page). Leave empty if the answer is in no document. |
| `type` | What kind of content holds the answer. See below. |

Suggested values for `type` (a proposal; change them if another split suits you
better, but keep them consistent, because results are reported per type):

- `text`: the answer is in running text
- `table`: the answer is in a table
- `chart`: the answer has to be read off a chart
- `flowchart`: the answer is a step or decision in a process diagram
- `scanned`: the answer is on a scanned page
- `not_in_docs`: no document contains the answer; the correct reply is
  "I could not find this in the documents."

Tips:

- Write questions and expected answers in English (the prototype is English-only for
  now).
- Include several `not_in_docs` questions. They are the only way to measure whether the
  chatbot refuses when it should.
- If a question has commas or quotes in it, wrap the field in double quotes, as usual
  for CSV. Saving from Excel as "CSV UTF-8" does this for you.
- 5 to 10 questions per type is enough to start.

## The figure questions (written on 2026-11-24)

`questions.csv` holds 37 questions: 27 of type `flowchart`, 6 of type `chart` and 4 of
type `not_in_docs`. There are none yet about running text or tables. Every expected
answer was read off the figure itself, on the page given. The project owner has still
to confirm them, and none of the questions has been run.

They need these 11 documents, ingested with figure descriptions switched on:

| File name (`source_doc`) | Page | Figure | Published at |
|---|---|---|---|
| `CERSAI Central KYC Registry Operating Guidelines 2016.pdf` | 7 | Swim-lane flowchart "FI Registration" | https://www.pfrda.org.in/documents/33652/145099/10.pdf |
| `SEBI Consultation Paper Blocking of Funds for Trading in Secondary Market.pdf` | 9 | "Transaction Flow", seven numbered steps | https://www.sebi.gov.in/sebi_data/attachdocs/jan-2023/1673940594580.pdf |
| `IBBI Insolvency and Bankruptcy News Jan-Mar 2025.pdf` | 5, 12 | Line, bar and pie charts | https://www.ibbi.gov.in/uploads/whatsnew/912e97d4d9f96651386541fb7059203b.pdf |
| `IDBI Bank Complaint Process Flow.pdf` | 1 | Flowchart with escalation by day | https://www.idbi.bank.in/pdf/Complaint-Flow-Chart.pdf |
| `India Post Payments Bank Complaint Flowchart.pdf` | 1 | Flowchart with two decisions | https://ippbonline.bank.in/documents/d/ippb/flow-chart-of-csrm-final |
| `Jammu and Kashmir Bank Grievance Redressal Flowchart.pdf` | 1 | Flowchart with four levels | https://jkb.bank.in/sites/default/files/Grievance_Redressal_Flowchart.pdf |
| `DCB Bank Customer Grievance Redressal Flowchart.pdf` | 1, 2 | One flowchart over two pages | https://www.dcb.bank.in/api/dcb/assets/2026-03/Complaint-Flowchart-March-2026.pdf |
| `SEBI Circular SCORES 20 Sep 2023.pdf` | 12 | Schedule III, process chart | https://www.sebi.gov.in/sebi_data/attachdocs/sep-2023/1695456916964.pdf |
| `IRDAI Insurance Repository Handbook.pdf` | 8, 10 | Process flows | https://policyholder.gov.in/documents/37343/625507/Handbook_Final+version.pdf |
| `Reserve Bank of India (Non-Banking Financial Companies Peer to Peer Lending Platform).pdf` | 24 | "Funds Transfer Mechanism" | in the RBI set, under `Master Directions/` |
| `Master Circular Lead Bank Scheme.pdf` | 68 | Flow chart of Annex III | in the RBI set, under `Master Circulars/` |

The first nine are not part of the RBI set. Save them under exactly these names, in a
folder such as `Figure tests` inside the documents folder, or add them with "+ Add" on
the chat screen. PDFs are not kept in git.

What the questions test, and what to look for in the results:

- **The wording is inside a picture** in the CERSAI guidelines, the SEBI consultation
  paper, the peer to peer directions, and the pie charts and the line chart of the IBBI
  newsletter. Their answers can only come from the description the vision model wrote.
- **The wording is selectable text** in the other flowcharts and in the bar chart of
  the IBBI newsletter (page 5). The words of the boxes can be found as plain text; the
  arrows and the yes/no branches exist only in the drawing. A right answer does not
  prove that the description was used: look at the kind of the passages found.
- The procedure of the Lead Bank Scheme flow chart is also written out on pages 66 and
  67. An answer that cites one of those pages is right as well, although "page in top
  5" will count it as a miss.
- The four `not_in_docs` questions are about matters no document here deals with, so
  the search should find nothing relevant. Change them if such a document is added.
- The personal-data check counts two numbers in the IBBI newsletter (pages 24 and 27).
  They are figures in tables; the limit of three is not reached.

With an answer taking about two and a half minutes, all 37 questions take about an
hour and a half. `--retrieval-only` first is much quicker.

## Running the questions

Stop the chat screen first. Then, from the project folder:

| Command | What it does |
|---|---|
| `python -m docqa.evaluate --suggest-threshold` | Runs the search only and proposes `retrieval.rerank_score_threshold`. It needs both answerable and `not_in_docs` questions |
| `python -m docqa.evaluate --retrieval-only` | Judges the search alone. The answer model is not used |
| `python -m docqa.evaluate` | Search and answer, for every question |
| `python -m docqa.evaluate --type table --limit 5` | Only some questions |

`source_doc` may be a file name or a path below `docs/`. The document has to be
ingested: a question about a document that is not gets a note saying so.

## results/

Each run writes a new folder here, named after the time it started:

- `results.csv`: one row per question, with the answer, the best rerank score, the top
  five passages and the measures below. The last two columns, `manual_correct` and
  `manual_notes`, are empty: they are for your own judgement of whether the answer is
  right.
- `summary.md`: the measures per question type, and the settings of the run.

The measures, per question type:

- **page in top 5**: the page that holds the answer is among the five best passages.
- **cites expected page**: the answer cites that document and page.
- **citations valid**: every citation points at a passage the model was given.
- **refusal correct**: `not_in_docs` questions are refused, and the others are not.
- **latency**: seconds per question.
