# Faster answers without a time limit: what was looked at

Written on 2026-11-24, on the laptop that has no models. Nothing here was measured
with a real model on the project's documents. Figures marked "estimate" are worked out
from one measurement (release 2026-11-20 on the laptop that runs the models: 81 s of
search and 248 s of answer model for one question) and from model sizes. Figures from
papers and model pages are given with their source; the model pages themselves could
not be opened from this laptop, so licences come from search results and have to be
checked on the page before a model is used.

## Why the time limits went

Release 2026-11-23b gave an answer 120 seconds and ranking 20 seconds. A limit can only
be kept by giving a model less to read: one or two passages where four were found, or
9 candidates ranked where 15 were found. That costs answers, so both limits were taken
out on 2026-11-24. An answer is again written from every passage that scores well
enough, and every candidate is ranked.

What stayed from that release, because it costs nothing in quality: the search models
run without the library's wasted trial run, the answer model is given its instructions
when the screen starts, the progress tracker, and adding documents from the screen. One
thing stayed that can change a ranking slightly: the reranker's 8-bit weights
(`reranker.quantize`), which make ranking about twice as fast.

## Where the time of an answer goes now

Estimates for the laptop that runs the models, with 15 candidates, 4 passages kept and
up to 8,000 characters of evidence:

| Step | Time | What decides it |
|---|---|---|
| Question and database | under a second | |
| Ranking 15 passages | 15 to 25 s (estimate; 14 s on the other laptop with a stand-in model) | the reranker: 568 million numbers, on the processor |
| The answer model reads | about 2 minutes (estimate) | about 2,000 tokens of evidence at about 15 tokens a second |
| The answer model writes | 20 to 50 s (estimate) | 100 to 200 tokens at 4 to 5 tokens a second |

So about three minutes, two of them the 7B model reading. Anything that is to make an
answer much faster without giving the model less to read has to make that reading
faster.

## What can be done, in the order it is recommended

### 1. A smaller answer model of the same family: Qwen3 4B Instruct

- **What:** `Qwen3-4B-Instruct-2507`, 4.0 billion parameters where `qwen2.5:7b` has
  7.6. It is the variant that does not "think" before answering. In Ollama:
  `qwen3:4b-instruct-2507-q4_K_M`, 2.5 GB.
- **How much faster:** reading and writing time go roughly with the size of a model, so
  about half (estimate): an answer of about three minutes becomes about a minute and
  a half to two minutes, from all the evidence.
- **Quality:** a 2026 study ran 17 small models as the answering model of a RAG system
  on a processor, without a graphics card, and scored the answers. This model came out
  as the best trade: correctness 0.71, faithfulness to the evidence 0.80, 71 s per
  answer. The 8B model of the same family scored 0.72 and 0.83 at 339 s. The study's
  documents were Russian and its questions were its own, so this says the model is a
  serious candidate, not that it answers these documents as well as the 7B does. That
  has to be seen on real questions here.
- **Licence:** Apache 2.0 according to the copies of its model page that a search
  returned. Check the page itself.
- **What it needs:** the project owner's decision, because `CLAUDE.md` lists the
  models that may be used, and a 2.5 GB download on the laptop that runs the models.
  No change to the code: the model is a setting.

Others of that size, should Qwen3 4B disappoint: `Phi-4-mini` (3.8B, MIT licence) and
`SmolLM3-3B` (Apache 2.0). IBM's Granite 4.0 Micro is another of that size; its
licence was not confirmed. Gemma models come under Google's own terms, which is the
kind of licence `CLAUDE.md` keeps out.

### 2. How many processor threads Ollama uses

Free, and it changes nothing about the answers. The processor has 2 fast and 8 slow
cores. Ollama's documentation says it chooses the number of threads by itself and
recommends the number of physical cores; reports for such processors go both ways (one
found fast cores only much better, another found all cores best). `HANDOVER.md`,
"Making it faster", has the ten-minute experiment.

### 3. A smaller reranker

- **What:** `Alibaba-NLP/gte-reranker-modernbert-base`: 149 million parameters, about
  a quarter of the present one, reads up to 8,192 tokens, Apache 2.0, and its own page
  gives it a score of 56.19 on the BEIR benchmark. No comparison with
  `bge-reranker-v2-m3` on the same test was found.
- **How much faster:** ranking in roughly a quarter of the time (estimate), so 5 s
  where it is 15 to 25 s. That is a small part of an answer, so this comes after 1.
- **Cost:** the reranker decides when the chatbot says it found nothing. Another
  reranker has another scale of scores, so the threshold has to be chosen again, and
  the search judged again, on real questions.
- **What it needs:** the owner's decision and a download. It is a setting
  (`reranker.model`); the loading goes through the same library.

### 4. Smaller passages

Passages are 400 to 800 tokens. With passages half that size the model reads half as
much for the same number of passages. A 2025 study over several sets of documents
found small passages (64 to 128 tokens) best for short factual answers and large ones
(512 to 1,024) best for the rest; another found that very small pieces lose the
context they need. This changes what is found as well as how long it takes, every
document has to be ingested again, and `CLAUDE.md` fixes the sizes. Worth trying only
if 1 to 3 are not enough, and only with test questions to judge it by.

### 5. The same model in a faster server program

OpenVINO Model Server is Intel's server program, made for Intel processors and their
built-in graphics chips. It runs on Windows without Docker, offers the same kind of
address as Ollama, and a ready-made 4-bit version of Qwen2.5 7B exists for it. No measurement for
this kind of laptop was found, so the gain is unknown. It would be a second model
server next to Ollama, which is a change to the stack.

## Looked at and not recommended

- **Ollama on the built-in graphics chip (Vulkan).** Experimental. One report has it
  writing nonsense on an Intel built-in chip with models of 3B and more, and the
  speeds one article quotes for such chips are close to what the processor does.
- **Cutting the evidence down to its relevant sentences before the model reads it.**
  The best-known model for this (Provence) reports 50 to 80 percent less text with
  little loss, and is licensed for non-commercial use only. Methods that delete single
  words are a risk where figures have to be quoted exactly.
- **Leaving the reranker out.** The search without it takes under a second, but the
  rule for refusing ("I could not find this in the documents") rests on its score.
- **A different database.** The database takes 6 ms of an answer (`machine.md`).
- **A hosted endpoint or a graphics card.** The fastest of all, and outside the rule
  that nothing leaves the machine and the hardware there is. `HANDOVER.md` has both.

## How to try another answer model

On the laptop that runs the models, after the owner has agreed to the model:

    ollama pull qwen3:4b-instruct-2507-q4_K_M

To compare it with the present model on the same questions, without changing a
setting (needs questions in `eval/questions.csv`):

    python -m docqa.evaluate
    python -m docqa.evaluate --answer-model qwen3:4b-instruct-2507-q4_K_M

Each run writes its own folder under `eval/results/` with every answer and its time,
and `summary.md` names the model. To use it in the chat screen, put it in
`config.local.yaml`:

    llm:
      answer_model: qwen3:4b-instruct-2507-q4_K_M

After a few questions, `python -m docqa.stats` shows the two models side by side under
"By answer model": how long their answers took, and how fast each reads and writes.

For a model that does think before it answers (the Qwen3 names without "instruct"),
set `llm.reasoning_effort: none` as well, or the thinking is written out first and
takes longer than the answer.

## Sources

- Little Brains, Big Feats: Exploring Compact Language Models (arXiv 2606.30062):
  https://arxiv.org/pdf/2606.30062
- Qwen3-4B-Instruct-2507, model page: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- The same model in Ollama's library: https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M
- gte-reranker-modernbert-base, model page: https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base
- Rethinking Chunk Size for Long-Document Retrieval (arXiv 2505.21700): https://arxiv.org/html/2505.21700v2
- Provence: context pruning for retrieval-augmented generation (arXiv 2501.16214):
  https://arxiv.org/pdf/2501.16214 and its model page
  https://huggingface.co/naver/provence-reranker-debertav3-v1
- OpenVINO Model Server: https://docs.openvino.ai/2026/model-server/ovms_what_is_openvino_model_server.html
  and the 4-bit Qwen2.5 7B for it: https://huggingface.co/OpenVINO/Qwen2.5-7B-Instruct-int4-ov
- Ollama, Modelfile reference (threads): https://docs.ollama.com/modelfile
- Ollama and Vulkan on an Intel built-in chip, the report of wrong output:
  https://github.com/ggml-org/llama.cpp/issues/19327
- How Ollama uses Intel's fast and slow cores, one test:
  https://www.glukhov.org/llm-performance/ollama/ollama-cpu-cores-usage/
- Switching off thinking through Ollama's OpenAI-compatible address, two reports:
  https://github.com/NousResearch/hermes-agent/issues/25758 and
  https://github.com/NousResearch/hermes-agent/issues/46131
