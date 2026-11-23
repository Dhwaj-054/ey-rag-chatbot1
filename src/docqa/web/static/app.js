// The chat screen. Plain JavaScript, no libraries, nothing fetched from outside this machine.
//
// It talks to the server in src/docqa/web/server.py:
//   GET  /api/status, /api/stats, /api/documents   for the side panel
//   POST /api/ask                                   one question, answered as a stream of events
//
// Text that comes from documents or from the model is never put into the page as HTML
// without being escaped first. The one exception is a table, which is rebuilt from a
// short list of allowed tags (see cleanTable).

"use strict";

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const view = { busy: false, controller: null, canAsk: false, documents: [], statusTimer: null };

// -- small helpers ---------------------------------------------------------------------

function esc(text) {
  return String(text).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function escapeRegExp(text) {
  return text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function seconds(value) {
  if (value === null || value === undefined) return "no data yet";
  if (value < 0.1) return Math.round(value * 1000) + " ms";
  if (value < 10) return value.toFixed(1) + " s";
  if (value < 120) return Math.round(value) + " s";
  return Math.floor(value / 60) + " min " + Math.round(value % 60) + " s";
}

function count(value) {
  return value === null || value === undefined ? "0" : Number(value).toLocaleString();
}

function plural(value, one, many) {
  return count(value) + " " + (Number(value) === 1 ? one : many);
}

async function getJSON(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(url + " answered " + response.status);
  return response.json();
}

// A hover card. Any element with a _tip property ({ title, lines }) gets one.
const tooltip = $("tooltip");

function showTip(target) {
  const tip = target._tip;
  if (!tip) return;
  tooltip.replaceChildren(el("div", "tip-title", tip.title), ...tip.lines.filter(Boolean).map((line) => el("div", "tip-line", line)));
  tooltip.hidden = false;
  const box = target.getBoundingClientRect();
  const size = tooltip.getBoundingClientRect();
  let left = tip.side === "right" ? box.right + 10 : box.left;
  let top = tip.side === "right" ? box.top : box.bottom + 8;
  left = Math.max(8, Math.min(left, window.innerWidth - size.width - 8));
  if (top + size.height > window.innerHeight - 8) top = Math.max(8, box.top - size.height - 8);
  tooltip.style.left = left + "px";
  tooltip.style.top = top + "px";
}

function hideTip() {
  tooltip.hidden = true;
}

function tipTarget(event) {
  let node = event.target;
  while (node && node !== document.body) {
    if (node._tip) return node;
    node = node.parentNode;
  }
  return null;
}

document.addEventListener("mouseover", (event) => {
  const target = tipTarget(event);
  if (target) showTip(target);
  else hideTip();
});
document.addEventListener("focusin", (event) => {
  const target = tipTarget(event);
  if (target) showTip(target);
});
document.addEventListener("focusout", hideTip);
document.addEventListener("scroll", hideTip, true);

// -- side panel ------------------------------------------------------------------------

function setStatus(kind, text) {
  $("status").className = "status " + kind;
  $("status-text").textContent = text;
}

function showProblems(problems) {
  const banner = $("banner");
  if (!problems.length) {
    banner.hidden = true;
    return;
  }
  const list = el("ul");
  for (const problem of problems) list.append(el("li", null, problem));
  banner.replaceChildren(el("strong", null, "The chatbot is not ready to answer."), list);
  banner.hidden = false;
}

async function refreshStatus() {
  clearTimeout(view.statusTimer);
  let status;
  try {
    status = await getJSON("/api/status");
  } catch (error) {
    setStatus("problem", "The server does not answer");
    view.canAsk = false;
    updateComposer();
    view.statusTimer = setTimeout(refreshStatus, 5000);
    return;
  }

  const settings = status.settings;
  const elsewhere = settings.answer_model_host;
  $("models").replaceChildren(
    chip("Answers", settings.answer_model + (elsewhere ? " on " + elsewhere : ""), elsewhere
      ? {
          title: "The answer model is not on this computer",
          lines: ["Every question, and the passages found for it, are sent to " + elsewhere + "."],
        }
      : { title: "The answer model runs on this computer", lines: ["Questions and passages are sent nowhere else."] }),
    chip("Search", settings.embedding_model.split("/").pop()),
    chip("Ranking", settings.reranker_model.split("/").pop()),
  );

  const loading = status.models.state === "loading";
  const blocking = status.problems.filter((problem) => !loading || !problem.startsWith("The search models"));
  view.canAsk = status.index.passages > 0 && status.threshold !== null && status.models.state !== "failed";
  if (loading) {
    setStatus("loading", "Loading the search models");
  } else if (status.ready) {
    const took = Object.values(status.models.seconds || {}).reduce((sum, value) => sum + value, 0);
    setStatus("ready", took ? "Ready. Models loaded in " + seconds(took) : "Ready");
  } else {
    setStatus("problem", "Not ready");
  }
  showProblems(blocking);
  updateComposer();
  view.statusTimer = setTimeout(refreshStatus, loading ? 2500 : 30000);
}

function chip(label, value, tip) {
  const node = el("span", "chip");
  node.append(label + " ", el("b", null, value));
  if (tip) node._tip = tip;
  return node;
}

function tile(value, label, tip) {
  const node = el("div", "tile");
  node.append(el("div", "value", value), el("div", "label", label));
  if (tip) node._tip = tip;
  return node;
}

function fact(label, value, tip) {
  const row = el("div", "fact");
  row.append(el("span", "fact-label", label), el("span", "fact-value", value));
  if (tip) row._tip = tip;
  return row;
}

async function refreshStats() {
  let stats;
  try {
    stats = await getJSON("/api/stats");
  } catch (error) {
    return;
  }
  const index = stats.index;
  const queries = stats.queries;
  const none = "no data yet";
  const kinds = index.text_passages === null
    ? []
    : [count(index.text_passages) + " text, " + plural(index.tables, "table", "tables") + ", " + plural(index.figures, "figure", "figures")];
  $("tiles").replaceChildren(
    tile(count(index.documents), "documents"),
    tile(count(index.pages), "pages"),
    tile(count(index.passages), "passages", {
      title: "A passage is a piece of text, a table or a figure description that a search can find.",
      lines: kinds,
    }),
  );
  $("facts").replaceChildren(
    fact("Questions asked", count(queries.questions), {
      title: queries.answered + " answered, " + queries.refused + " refused, " + queries.failed + " failed",
      lines: [],
    }),
    fact("Typical answer time", queries.median_latency_s === null ? none : seconds(queries.median_latency_s), {
      title: "From question to finished answer",
      lines: ["Half of the answered questions took less than this."],
    }),
    fact("Typical search time", queries.median_search_s === null ? none : seconds(queries.median_search_s), {
      title: "Finding and ranking the passages",
      lines: ["Ranking takes most of it. Fewer candidates make it shorter."],
    }),
    fact("of which database", queries.median_database_s === null || queries.median_database_s === undefined ? none : seconds(queries.median_database_s), {
      title: "Looking the passages up in the vector store",
      lines: ["The rest of the search time is the two search models, which run on the processor."],
    }),
    fact("Wait for the first word", queries.median_first_token_s === null ? none : seconds(queries.median_first_token_s), {
      title: "How long the answer model reads before it writes",
      lines: ["Fewer or shorter passages make it shorter."],
    }),
    fact("Writing speed", queries.median_tokens_per_s === null ? none : queries.median_tokens_per_s.toFixed(1) + " tokens/s", {
      title: "How fast the answer model writes once it has started",
      lines: ["A token is about three quarters of a word."],
    }),
  );
}

async function refreshDocuments() {
  try {
    view.documents = (await getJSON("/api/documents")).documents;
  } catch (error) {
    view.documents = [];
  }
  renderDocuments();
}

function renderDocuments() {
  const wanted = $("doc-filter").value.trim().toLowerCase();
  const shown = view.documents.filter(
    (doc) => !wanted || doc.title.toLowerCase().includes(wanted) || doc.source_path.toLowerCase().includes(wanted),
  );
  const list = $("doc-list");
  list.replaceChildren();
  for (const doc of shown) {
    const item = el("li", "doc");
    item.tabIndex = 0;
    item.setAttribute("aria-label", doc.title);
    const facts = [
      doc.effective_date || "no date",
      doc.pages === null ? null : doc.pages + (doc.pages === 1 ? " page" : " pages"),
      doc.passages + (doc.passages === 1 ? " passage" : " passages"),
    ].filter(Boolean);
    item.append(el("span", "doc-name", doc.short_title), el("span", "doc-meta", facts.join(" · ")));
    item._tip = {
      title: doc.title,
      side: "right",
      lines: [
        "File: " + doc.source_path,
        doc.version ? "Version: " + doc.version : null,
        "Effective date: " + (doc.effective_date || "not recorded"),
        facts.slice(1).join(", ") + ", access group " + doc.access_group,
        "Ingested: " + doc.ingested_at.replace("T", " ").replace("+00:00", " UTC"),
      ],
    };
    list.append(item);
  }
  $("doc-count").textContent = wanted ? shown.length + " of " + view.documents.length : String(view.documents.length);
  const empty = $("doc-empty");
  empty.hidden = shown.length > 0;
  empty.textContent = view.documents.length ? "No document matches that." : "Nothing has been indexed yet.";
}

// -- text ------------------------------------------------------------------------------

// Turns plain text into paragraphs and lists. `chips` replace matched pieces (citations)
// with ready-made HTML; everything else is escaped.
function formatText(text, chips) {
  const holders = [];
  let work = text;
  for (const item of chips) {
    work = work.replace(item.pattern, (...match) => {
      holders.push(item.html(match));
      return "\u0000" + (holders.length - 1) + "\u0000";
    });
  }
  let html = esc(work).replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  const bullet = /^\s*[-*•]\s+/;
  const numbered = /^\s*\d+[.)]\s+/;
  html = html
    .split(/\n{2,}/)
    .filter((block) => block.trim())
    .map((block) => {
      const lines = block.split("\n").filter((line) => line.trim());
      if (lines.length && lines.every((line) => bullet.test(line))) {
        return "<ul>" + lines.map((line) => "<li>" + line.replace(bullet, "") + "</li>").join("") + "</ul>";
      }
      if (lines.length && lines.every((line) => numbered.test(line))) {
        return "<ol>" + lines.map((line) => "<li>" + line.replace(numbered, "") + "</li>").join("") + "</ol>";
      }
      return "<p>" + lines.join("<br>") + "</p>";
    })
    .join("");
  return html.replace(/\u0000(\d+)\u0000/g, (_, index) => holders[Number(index)]);
}

// While the model is writing, its references read [E1, p. 12]. They are shown as grey
// chips until the server has checked them.
const PENDING = [
  {
    pattern: /\[\s*E\s?(\d{1,2})\s*(?:,\s*pp?\.?\s*(\d+)(?:\s*[-–]\s*(\d+))?)?\s*\]/gi,
    html: (match) => '<span class="cite pending">passage ' + esc(match[1]) + (match[2] ? " · p. " + esc(match[2]) + (match[3] ? "-" + esc(match[3]) : "") : "") + "</span>",
  },
];

function citationChips(citations) {
  const titles = [...new Set(citations.map((citation) => citation.doc_title))];
  return titles.map((title) => ({
    pattern: new RegExp("\\[" + escapeRegExp(title) + ",\\s*pp?\\.\\s*(\\d+)(?:-(\\d+))?\\]", "g"),
    html: (match) => {
      const page = Number(match[1]);
      const citation = citations.find((item) => item.doc_title === title && item.page === page) || citations.find((item) => item.doc_title === title);
      const pages = "p" + (match[2] ? "p. " + match[1] + "-" + match[2] : ". " + match[1]);
      return (
        '<button type="button" class="cite" data-chunk="' + esc(citation.chunk_id) + '" data-title="' + esc(title) + '" data-pages="' + esc(pages) + '">' +
        '<span class="cite-name">' + esc(citation.short_title) + '</span><span class="cite-page">' + esc(pages) + "</span></button>"
      );
    },
  }));
}

// A table from a document, rebuilt from a short list of allowed tags.
function cleanTable(html) {
  const allowed = new Set(["TABLE", "CAPTION", "THEAD", "TBODY", "TFOOT", "TR", "TH", "TD", "BR", "DIV"]);
  const source = new DOMParser().parseFromString(html, "text/html").querySelector("table");
  if (!source) return null;
  const copy = (node) => {
    if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.textContent);
    if (node.nodeType !== Node.ELEMENT_NODE) return null;
    if (!allowed.has(node.tagName)) return document.createTextNode(node.textContent);
    const made = document.createElement(node.tagName === "DIV" ? "span" : node.tagName.toLowerCase());
    for (const name of ["colspan", "rowspan"]) {
      const value = node.getAttribute(name);
      if (value && /^\d{1,3}$/.test(value)) made.setAttribute(name, value);
    }
    for (const child of node.childNodes) {
      const made_child = copy(child);
      if (made_child) made.append(made_child);
    }
    return made;
  };
  return copy(source);
}

// -- one question ----------------------------------------------------------------------

function nearBottom() {
  return thread.scrollHeight - thread.scrollTop - thread.clientHeight < 140;
}

function scrollDown(force) {
  if (force || nearBottom()) thread.scrollTop = thread.scrollHeight;
}

function addQuestion(question) {
  $("welcome")?.remove();
  const row = el("div", "msg user");
  row.append(el("div", "bubble", question));
  thread.append(row);
}

function createAnswer() {
  const card = el("article", "msg card");
  const steps = el("div", "steps");
  const answer = el("div", "answer");
  const notes = el("div", "notes");
  const meta = el("div", "meta");
  const sources = el("details", "sources");
  sources.hidden = true;
  card.append(steps, answer, notes, meta, sources);
  thread.append(card);

  let raw = "";
  let step = null;
  let retrieval = null;
  const sourceNodes = new Map();

  const self = {
    startStep(label) {
      if (step && step.text.textContent === label) return; // already showing this step
      self.endStep();
      const node = el("div", "step active");
      const text = el("span", null, label);
      const clock = el("span", "clock", "0 s");
      node.append(text, clock);
      steps.append(node);
      const began = performance.now();
      const timer = setInterval(() => {
        clock.textContent = seconds((performance.now() - began) / 1000);
      }, 500);
      step = { node, text, clock, began, timer };
      scrollDown();
    },

    endStep(label, failed, took) {
      if (!step) return;
      clearInterval(step.timer);
      step.node.className = "step " + (failed ? "failed" : "done");
      if (label) step.text.textContent = label;
      step.clock.textContent = seconds(took === undefined ? (performance.now() - step.began) / 1000 : took);
      step = null;
    },

    showRetrieval(data) {
      retrieval = data;
      const given = data.sources.filter((source) => source.given_to_model).length;
      if (data.found) {
        self.endStep("Found " + data.sources.length + (data.sources.length === 1 ? " passage" : " passages"), false, data.elapsed_s);
        self.startStep("Reading " + given + (given === 1 ? " passage" : " passages") + " and writing the answer");
      } else {
        const best = data.best_score === null ? "" : " The best passage scored " + data.best_score.toFixed(2) + "; the threshold is " + data.threshold.toFixed(2) + ".";
        self.endStep("Nothing relevant enough was found." + best, false, data.elapsed_s);
      }
      if (!data.sources.length) return;

      const summary = el("summary");
      summary.textContent = data.found
        ? "Passages found (" + data.sources.length + "). " + given + " given to the answer model."
        : "Closest passages (" + data.sources.length + "). None was used.";
      const list = el("div", "source-list");
      for (const source of data.sources) {
        const node = sourceCard(source, data.found);
        sourceNodes.set(source.chunk_id, node);
        list.append(node);
      }
      sources.replaceChildren(summary, list);
      sources.hidden = false;
      sources.open = true;
      scrollDown();
    },

    addText(piece) {
      raw += piece;
      answer.innerHTML = formatText(raw, PENDING);
      answer.classList.add("cursor");
      scrollDown();
    },

    showAnswer(data) {
      self.endStep(data.refused ? null : "Answer written");
      answer.classList.remove("cursor");
      answer.classList.toggle("refused", data.refused);
      answer.innerHTML = formatText(data.text, citationChips(data.citations));
      for (const button of answer.querySelectorAll("button.cite")) {
        button._tip = { title: button.dataset.title, lines: [button.dataset.pages + ". Click to see the passage."] };
        button.addEventListener("click", () => self.reveal(button.dataset.chunk));
      }
      notes.replaceChildren(...data.notes.map((note) => el("div", "note", note)));

      const cited = new Set(data.citations.map((citation) => citation.chunk_id));
      for (const [chunkId, node] of sourceNodes) {
        if (!cited.has(chunkId)) continue;
        node.classList.add("cited");
        const badge = node.querySelector(".badge.unused");
        if (badge) badge.remove();
        node.querySelector("summary .score").before(el("span", "badge cited", "cited"));
      }

      const timings = data.timings;
      const search = timings.retrieval;
      meta.append(
        chip("Question read", seconds(search.embed_s), {
          title: "Turning the question into numbers",
          lines: ["Done by the embedding model, on this computer's processor."],
        }),
        chip("Database", seconds(search.search_s), {
          title: "Looking the passages up in the vector store",
          lines: ["Two searches, by meaning and by words, merged into one list."],
        }),
        chip("Ranking", seconds(search.rerank_s), {
          title: "Reading " + plural(search.candidates, "passage", "passages") + " against the question",
          lines: ["Done by the reranker, on this computer's processor.", "Fewer candidates make it shorter."],
        }),
      );
      if (data.answer_model) {
        if (timings.first_token_s !== null) {
          meta.append(
            chip("First word after", seconds(timings.first_token_s), {
              title: "Time the answer model spent reading before it wrote anything",
              lines: [
                count(timings.evidence_chars) + " characters of evidence" + (timings.prompt_tokens ? ", " + count(timings.prompt_tokens) + " tokens in all" : ""),
                "Fewer or shorter passages make this shorter.",
              ],
            }),
          );
        }
        const speed = timings.tokens_per_s ? " at " + timings.tokens_per_s.toFixed(1) + " tokens/s" : "";
        const wrote = timings.first_token_s === null ? timings.generation_s : timings.generation_s - timings.first_token_s;
        meta.append(chip("Writing", seconds(Math.max(wrote, 0)) + speed), chip("Model", data.answer_model));
      } else {
        meta.append(chip("Answer model", "not called"));
      }
      meta.append(chip("Total", seconds(data.latency_s)));
      scrollDown();
    },

    reveal(chunkId) {
      const node = sourceNodes.get(chunkId);
      if (!node) return;
      sources.open = true;
      node.open = true;
      node.scrollIntoView({ block: "center" });
      node.classList.remove("flash");
      void node.offsetWidth; // restarts the animation
      node.classList.add("flash");
    },

    fail(message) {
      self.endStep(null, true);
      answer.classList.remove("cursor");
      notes.append(el("div", "note error", message));
      scrollDown();
    },

    stopped() {
      self.endStep("Stopped", true);
      answer.classList.remove("cursor");
      notes.append(el("div", "note", "Stopped. The answer above is incomplete and was not checked."));
    },

    finish() {
      self.endStep();
      answer.classList.remove("cursor");
      if (!answer.textContent && !notes.childElementCount && retrieval === null) {
        notes.append(el("div", "note error", "The server closed the connection without an answer."));
      }
    },
  };
  return self;
}

function sourceCard(source, found) {
  const card = el("details", "source" + (source.given_to_model ? " given" : ""));
  const summary = el("summary");
  const title = el("span", "src-title", source.short_title);
  title._tip = { title: source.doc_title, lines: [source.pages + (source.heading_path.length ? " · " + source.heading_path.join(" > ") : "")] };
  summary.append(el("span", "rank", String(source.rank)), title, el("span", "src-page", source.pages), el("span", "badge", source.type));
  if (found && !source.given_to_model) {
    const unused = el("span", "badge unused", "not given to the model");
    unused._tip = { title: "Left out to keep the answer fast", lines: ["Its score is well below the best passage's, or the evidence limit was reached."] };
    summary.append(unused);
  }
  const score = el("span", "score");
  const track = el("span", "track");
  const bar = el("span", "bar");
  const value = source.score === null ? 0 : Math.max(0, Math.min(1, source.score));
  bar.style.width = Math.round(value * 100) + "%";
  track.append(bar);
  score.append(track, el("span", "num", source.score === null ? "" : source.score.toFixed(2)));
  score._tip = { title: "Relevance score from the reranker", lines: ["From 0 (unrelated) to 1 (answers the question)."] };
  summary.append(score);
  card.append(summary);

  // The passage itself is built only when it is first opened.
  card.addEventListener(
    "toggle",
    () => {
      if (!card.open || card.querySelector(".src-body")) return;
      const body = el("div", "src-body");
      const facts = [];
      if (source.heading_path.length) facts.push("Section: " + source.heading_path.join(" > "));
      facts.push("Effective date: " + (source.effective_date || "not recorded"));
      if (source.version) facts.push("Version: " + source.version);
      body.append(el("div", "src-facts", facts.join(" · ")));

      const table = source.type === "table" && source.html ? cleanTable(source.html) : null;
      if (table) {
        const wrap = el("div", "table-wrap");
        wrap.append(table);
        body.append(wrap);
      } else if (source.type === "figure") {
        const figure = el("div", "figure");
        if (source.image_url) {
          const image = el("img");
          image.src = source.image_url;
          image.alt = "Figure from " + source.doc_title + ", " + source.pages;
          image.loading = "lazy";
          figure.append(image);
        }
        if (source.model_generated) {
          figure.append(el("div", "note", "This description was written by the vision model. It may contain mistakes: check it against the figure."));
        }
        figure.append(el("pre", "passage", source.text));
        body.append(figure);
      } else {
        body.append(el("pre", "passage", source.text));
      }
      card.append(body);
    },
  );
  return card;
}

async function readEvents(response, onEvent) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}

async function ask(question) {
  view.busy = true;
  updateComposer();
  addQuestion(question);
  const answer = createAnswer();
  answer.startStep("Searching the documents");
  scrollDown(true);
  const controller = new AbortController();
  view.controller = controller;
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.error || "The server answered " + response.status + ".");
    }
    await readEvents(response, (event, data) => {
      if (event === "status" && data.stage === "waiting") answer.startStep("Waiting for the previous question to finish");
      else if (event === "status" && data.stage === "searching") answer.startStep("Searching the documents");
      else if (event === "retrieval") answer.showRetrieval(data);
      else if (event === "token") answer.addText(data.text);
      else if (event === "answer") answer.showAnswer(data);
      else if (event === "error") answer.fail(data.message);
    });
  } catch (error) {
    if (error.name === "AbortError") answer.stopped();
    else answer.fail(error.message);
  } finally {
    answer.finish();
    view.busy = false;
    view.controller = null;
    updateComposer();
    refreshStats();
    $("question").focus();
  }
}

// -- composer --------------------------------------------------------------------------

function updateComposer() {
  const question = $("question");
  $("send").hidden = view.busy;
  $("stop").hidden = !view.busy;
  $("send").disabled = !view.canAsk || !question.value.trim();
  question.disabled = view.busy;
}

function resizeQuestion() {
  const question = $("question");
  question.style.height = "auto";
  question.style.height = Math.min(question.scrollHeight, 180) + "px";
  question.style.overflowY = question.scrollHeight > 180 ? "auto" : "hidden";
}

$("composer").addEventListener("submit", (event) => {
  event.preventDefault();
  const question = $("question");
  const text = question.value.trim();
  if (!text || view.busy || !view.canAsk) return;
  question.value = "";
  resizeQuestion();
  ask(text);
});

$("question").addEventListener("input", () => {
  resizeQuestion();
  updateComposer();
});

$("question").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    $("composer").requestSubmit();
  }
});

$("stop").addEventListener("click", () => view.controller?.abort());

$("doc-filter").addEventListener("input", renderDocuments);

$("toggle-sidebar").addEventListener("click", () => {
  const hidden = $("app").classList.toggle("sidebar-hidden");
  $("toggle-sidebar").setAttribute("aria-expanded", String(!hidden));
});

if (window.matchMedia("(max-width: 860px)").matches) {
  $("app").classList.add("sidebar-hidden");
  $("toggle-sidebar").setAttribute("aria-expanded", "false");
}

refreshStatus();
refreshStats();
refreshDocuments();
updateComposer();
