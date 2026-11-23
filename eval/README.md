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
