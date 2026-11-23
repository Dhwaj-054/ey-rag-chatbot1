// The chat screen. Plain JavaScript, no libraries, nothing fetched from outside this machine.
//
// It talks to the server in src/docqa/web/server.py:
//   GET  /api/status, /api/stats, /api/documents   for the side panel
//   POST /api/ask                                   one question, answered as a stream of events
//   GET  /api/ingest, POST /api/upload, /api/ingest, /api/ingest/stop   adding documents
//
// The answer to a question arrives as a stream. It is read with the browser's Streams
// API (response.body.getReader(), in readEvents below), so every step the server takes
// shows on the screen the moment it happens: that is what the progress tracker at the
// top of each answer is drawn from.
//
// Text that comes from documents or from the model is never put into the page as HTML
// without being escaped first. The one exception is a table, which is rebuilt from a
// short list of allowed tags (see cleanTable).

"use strict";

const $ = (id) => document.getElementById(id);
const thread = $("thread");
const view = { busy: false, controller: null, canAsk: false, ingesting: false, documents: [], statusTimer: null };

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

// A stopwatch reading: 1:07.
function clock(value) {
  const whole = Math.max(0, Math.floor(value));
  return Math.floor(whole / 60) + ":" + String(whole % 60).padStart(2, "0");
}

function count(value) {
  return value === null || value === undefined ? "0" : Number(value).toLocaleString();
}

function plural(value, one, many) {
  return count(value) + " " + (Number(value) === 1 ? one : many);
}

function fileSize(bytes) {
  if (bytes < 1024 * 1024) return Math.max(1, Math.round(bytes / 1024)) + " KB";
  return (bytes / 1024 / 1024).toFixed(1) + " MB";
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

// A dialog that is open as a modal sits in the browser's top layer, above everything in
// the page. The hover card is moved into it while it is open, or it would be hidden.
function tipHome() {
  const dialog = $("ingest-dialog");
  const home = dialog.open ? dialog : document.body;
  if (tooltip.parentNode !== home) home.append(tooltip);
}

document.addEventListener("mouseover", (event) => {
  const target = tipTarget(event);
  if (target) {
    tipHome();
    showTip(target);
  } else hideTip();
});
document.addEventListener("focusin", (event) => {
  const target = tipTarget(event);
  if (target) {
    tipHome();
    showTip(target);
  }
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
  const chips = [
    chip("Answers", settings.answer_model + (elsewhere ? " on " + elsewhere : ""), elsewhere
      ? {
          title: "The answer model is not on this computer",
          lines: ["Every question, and the passages found for it, are sent to " + elsewhere + "."],
        }
      : { title: "The answer model runs on this computer", lines: ["Questions and passages are sent nowhere else."] }),
    chip("Search", settings.embedding_model.split("/").pop()),
    chip("Ranking", settings.reranker_model.split("/").pop() + (settings.reranker_8_bit ? " · 8-bit" : ""), {
      title: "The model that reads each candidate passage against the question",
      lines: [
        settings.reranker_8_bit ? "It runs with 8-bit weights, which is about twice as fast on a processor." : "It runs at full precision.",
        "Up to " + plural(settings.candidates, "candidate", "candidates") + " are ranked" +
          (settings.rerank_time_limit_s ? ", for at most " + seconds(settings.rerank_time_limit_s) + "." : "."),
      ],
    }),
  ];
  if (settings.time_limit_s) {
    chips.push(chip("Time limit", seconds(settings.time_limit_s), {
      title: "The longest an answer should take",
      lines: [
        "The evidence is cut to what the answer model can read in that time.",
        "When the time is up, the answer stops at the end of its last complete statement.",
      ],
    }));
  }
  $("models").replaceChildren(...chips);

  const loading = status.models.state === "loading";
  const blocking = status.problems.filter((problem) => !loading || !problem.startsWith("The search models"));
  view.canAsk = status.index.passages > 0 && status.threshold !== null && status.models.state !== "failed";
  if (status.ingesting) {
    setStatus("loading", "Ingesting documents");
  } else if (loading) {
    setStatus("loading", "Loading the search models");
  } else if (status.ready) {
    const took = Object.values(status.models.seconds || {}).reduce((sum, value) => sum + value, 0);
    setStatus("ready", took ? "Ready. Models loaded in " + seconds(took) : "Ready");
  } else {
    setStatus("problem", "Not ready");
  }
  showProblems(blocking);
  setIngesting(status.ingesting);
  if (status.ingesting && !ingest.pollTimer) pollJob();
  view.statusTimer = setTimeout(refreshStatus, loading || status.ingesting ? 2500 : 30000);
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
  const given = (value) => value !== null && value !== undefined;
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
      lines: [queries.stopped_at_limit ? plural(queries.stopped_at_limit, "answer was", "answers were") + " stopped at the time limit." : null],
    }),
    fact("Typical answer time", given(queries.median_latency_s) ? seconds(queries.median_latency_s) : none, {
      title: "From question to finished answer",
      lines: ["Half of the answered questions took less than this."],
    }),
    fact("Typical search time", given(queries.median_search_s) ? seconds(queries.median_search_s) : none, {
      title: "Finding and ranking the passages",
      lines: ["Ranking takes most of it. Fewer candidates make it shorter."],
    }),
    fact("of which database", given(queries.median_database_s) ? seconds(queries.median_database_s) : none, {
      title: "Looking the passages up in the vector store",
      lines: ["The rest of the search time is the two search models, which run on the processor."],
    }),
    fact("Wait for the first word", given(queries.median_first_token_s) ? seconds(queries.median_first_token_s) : none, {
      title: "How long the answer model reads before it writes",
      lines: ["Fewer or shorter passages make it shorter."],
    }),
    fact("Reading speed", given(queries.median_read_chars_per_s) ? count(Math.round(queries.median_read_chars_per_s)) + " characters/s" : none, {
      title: "How fast the answer model reads the passages it is given",
      lines: ["Measured on this computer. With a time limit, it decides how much evidence an answer gets."],
    }),
    fact("Writing speed", given(queries.median_tokens_per_s) ? queries.median_tokens_per_s.toFixed(1) + " tokens/s" : none, {
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
  empty.textContent = view.documents.length ? "No document matches that." : "Nothing has been indexed yet. Use Add to ingest PDFs.";
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

// -- one question: the progress tracker ------------------------------------------------

// The five steps of an answer, in order. The server names the one it is on.
const STEPS = [
  {
    name: "question",
    label: "Question",
    doing: "Reading the question",
    tip: { title: "Turning the question into numbers", lines: ["Done by the embedding model, on this computer's processor."] },
  },
  {
    name: "database",
    label: "Search",
    doing: "Searching the index",
    tip: { title: "Looking the passages up in the vector store", lines: ["Two searches, by meaning and by words, merged into one list of candidates."] },
  },
  {
    name: "ranking",
    label: "Ranking",
    doing: "Ranking the candidates",
    tip: { title: "Reading each candidate passage against the question", lines: ["Done by the reranker, on this computer's processor.", "The candidates the search placed highest are read first."] },
  },
  {
    name: "reading",
    label: "Reading",
    doing: "The answer model is reading",
    tip: { title: "The answer model reads the passages before it writes", lines: ["On a processor this is the longest part of an answer.", "Fewer or shorter passages make it shorter."] },
  },
  {
    name: "writing",
    label: "Writing",
    doing: "Writing the answer",
    tip: { title: "The answer model writes, word by word", lines: ["The words appear below as they are written."] },
  },
];

function createTracker(host) {
  const root = el("div", "tracker");
  const list = el("ol", "t-steps");
  const nodes = {};
  for (const step of STEPS) {
    const item = el("li", "t-step pending");
    const info = el("span", "t-info", "");
    item.append(el("span", "t-mark"), el("span", "t-name", step.label), info);
    item._tip = { title: step.tip.title, lines: [...step.tip.lines] };
    list.append(item);
    nodes[step.name] = { item, info, step, began: null, detail: "" };
  }
  const bar = el("div", "t-bar");
  bar.setAttribute("role", "progressbar");
  bar.setAttribute("aria-label", "Time used of the limit for an answer");
  bar.setAttribute("aria-valuemin", "0");
  const fill = el("span", "t-fill");
  bar.append(fill);
  bar.hidden = true;
  const foot = el("div", "t-foot");
  const now = el("span", "t-now", "Sending the question");
  now.setAttribute("aria-live", "polite");
  const time = el("span", "t-clock", "");
  foot.append(now, time);
  root.append(list, bar, foot);
  host.append(root);

  let limit = null;
  let began = null;
  let timer = null;
  let active = null;
  let words = 0;

  function mark(name, state) {
    nodes[name].item.className = "t-step " + state;
  }

  function elapsed() {
    return began === null ? 0 : (performance.now() - began) / 1000;
  }

  function showClock(value) {
    if (limit) {
      const share = value / limit;
      fill.style.width = Math.min(share, 1) * 100 + "%";
      bar.className = "t-bar" + (share > 1 ? " over" : share > 0.85 ? " near" : "");
      bar.setAttribute("aria-valuenow", String(Math.round(Math.min(value, limit))));
      time.textContent = clock(value) + " of " + clock(limit);
    } else {
      time.textContent = clock(value);
    }
  }

  function tick() {
    if (began === null) return;
    showClock(elapsed());
    if (!active) return;
    const node = nodes[active];
    const spent = (performance.now() - node.began) / 1000;
    const detail = active === "writing" && words ? plural(words, "word", "words") : node.detail;
    node.info.textContent = (detail ? detail + " · " : "") + seconds(spent);
  }

  function stopClock() {
    clearInterval(timer);
    timer = null;
  }

  const self = {
    // The server has started on this question: the clock runs from here.
    begin(timeLimit) {
      limit = timeLimit || null;
      began = performance.now();
      bar.hidden = !limit;
      if (limit) bar.setAttribute("aria-valuemax", String(Math.round(limit)));
      now.textContent = "Starting";
      stopClock();
      timer = setInterval(tick, 250);
      tick();
    },

    step(data) {
      if (data.name === "waiting") {
        if (data.state === "start") {
          now.textContent = data.reason === "ingestion"
            ? "Waiting: documents are being ingested. The answer starts when that is done."
            : "Waiting for the previous question to finish";
        }
        return;
      }
      const node = nodes[data.name];
      if (!node) return;
      if (data.state === "start") {
        active = data.name;
        node.began = performance.now();
        node.detail = "";
        mark(data.name, "active");
        now.textContent = node.step.doing;
        if (data.name === "ranking") {
          node.detail = "0 of " + data.of;
          now.textContent = "Ranking " + plural(data.of, "candidate passage", "candidate passages") + " against the question";
        } else if (data.name === "reading") {
          node.detail = plural(data.passages, "passage", "passages");
          now.textContent = "The answer model is reading " + plural(data.passages, "passage", "passages") +
            " (" + count(data.chars) + " characters)." +
            (data.expected_s ? " About " + seconds(data.expected_s) + " on this computer." : "");
        }
      } else if (data.state === "progress") {
        node.detail = data.done + " of " + data.of;
      } else if (data.state === "end") {
        mark(data.name, "done");
        let text = seconds(data.took_s);
        if (data.name === "ranking") {
          const cut = data.done < data.of;
          text = (cut ? data.done + " of " + data.of : String(data.done)) + " · " + text;
          if (cut) {
            node.item.classList.add("cut");
            node.item._tip.lines.push("Stopped at its time limit: " + data.done + " of the " + data.of + " candidates were read.");
          }
        } else if (data.name === "writing") {
          if (words) text = plural(words, "word", "words") + " · " + text;
          if (data.stopped) node.item.classList.add("cut");
        } else if (data.name === "reading" && node.detail) {
          text = node.detail + " · " + text;
        }
        node.info.textContent = text;
        if (active === data.name) active = null;
      }
      tick();
    },

    wrote(text) {
      words = text.trim() ? text.trim().split(/\s+/).length : 0;
    },

    // Nothing was found: the answer model is not asked, so its two steps do not happen.
    skipModel() {
      for (const name of ["reading", "writing"]) {
        mark(name, "skipped");
        nodes[name].info.textContent = "not needed";
      }
    },

    finish(message) {
      stopClock();
      for (const step of STEPS) {
        if (nodes[step.name].item.classList.contains("active")) mark(step.name, "done");
      }
      active = null;
      if (began !== null) showClock(elapsed());
      if (message) now.textContent = message;
    },

    fail(message) {
      stopClock();
      for (const step of STEPS) {
        if (nodes[step.name].item.classList.contains("active")) mark(step.name, "failed");
      }
      active = null;
      if (began !== null) showClock(elapsed());
      now.textContent = message;
    },

    running() {
      return timer !== null || began === null;
    },
  };
  return self;
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
  const tracker = createTracker(card);
  const answer = el("div", "answer");
  const notes = el("div", "notes");
  const meta = el("div", "meta");
  const sources = el("details", "sources");
  sources.hidden = true;
  card.append(answer, notes, meta, sources);
  thread.append(card);

  let raw = "";
  let retrieval = null;
  let ended = false;
  const sourceNodes = new Map();

  const self = {
    tracker,

    showRetrieval(data) {
      retrieval = data;
      const given = data.sources.filter((source) => source.given_to_model).length;
      if (!data.found) {
        tracker.skipModel();
        const best = data.best_score === null ? "" : " The best passage scored " + data.best_score.toFixed(2) + "; the threshold is " + data.threshold.toFixed(2) + ".";
        tracker.finish("Nothing relevant enough was found." + best);
        ended = true;
      }
      if (!data.sources.length) return;

      const summary = el("summary");
      summary.textContent = data.found
        ? "Passages found (" + data.sources.length + "). " + given + " given to the answer model."
        : "Closest passages (" + data.sources.length + "). None was used.";
      const list = el("div", "source-list");
      const limited = Boolean(data.plan && data.plan.shortened);
      for (const source of data.sources) {
        const node = sourceCard(source, data.found, limited);
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
      tracker.wrote(raw);
      answer.innerHTML = formatText(raw, PENDING);
      answer.classList.add("cursor");
      scrollDown();
    },

    showAnswer(data) {
      const timings = data.timings;
      if (!ended) {
        tracker.finish(
          data.refused
            ? "The answer model replied that the passages do not answer the question."
            : (timings.stopped_at_limit ? "Stopped at the time limit after " : "Answered in ") + seconds(data.latency_s),
        );
      }
      ended = true;
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

      if (data.answer_model) {
        meta.append(
          chip("Model", data.answer_model),
          chip("Evidence", count(timings.evidence_chars) + " characters", {
            title: "How much the answer model was given to read",
            lines: [
              timings.prompt_tokens ? count(timings.prompt_tokens) + " tokens in all, with its instructions and the question" : null,
              timings.first_token_s ? "It wrote its first word after " + seconds(timings.first_token_s) + "." : null,
            ],
          }),
        );
        if (timings.tokens_per_s) meta.append(chip("Writing speed", timings.tokens_per_s.toFixed(1) + " tokens/s"));
      } else {
        meta.append(chip("Answer model", "not called"));
      }
      meta.append(chip("Total", seconds(data.latency_s), {
        title: "From the start of the search to the last word",
        lines: [timings.time_limit_s ? "The limit for an answer is " + seconds(timings.time_limit_s) + "." : null],
      }));
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
      ended = true;
      tracker.fail("Something went wrong");
      answer.classList.remove("cursor");
      notes.append(el("div", "note error", message));
      scrollDown();
    },

    stopped() {
      ended = true;
      tracker.fail("Stopped");
      answer.classList.remove("cursor");
      notes.append(el("div", "note", "Stopped. The answer above is incomplete and was not checked."));
    },

    finish() {
      answer.classList.remove("cursor");
      if (!ended) {
        tracker.fail("The connection closed");
        if (!answer.textContent && !notes.childElementCount) {
          notes.append(el("div", "note error", "The server closed the connection without an answer."));
        }
      }
    },
  };
  return self;
}

function sourceCard(source, found, limited) {
  const card = el("details", "source" + (source.given_to_model ? " given" : ""));
  const summary = el("summary");
  const title = el("span", "src-title", source.short_title);
  title._tip = { title: source.doc_title, lines: [source.pages + (source.heading_path.length ? " · " + source.heading_path.join(" > ") : "")] };
  summary.append(el("span", "rank", String(source.rank)), title, el("span", "src-page", source.pages), el("span", "badge", source.type));
  if (found && !source.given_to_model) {
    const unused = el("span", "badge unused", "not given to the model");
    unused._tip = {
      title: "Left out to keep the answer fast",
      lines: [limited
        ? "There was no time left to read it within the limit for an answer, or its score is well below the best passage's."
        : "Its score is well below the best passage's, or the evidence limit was reached."],
    };
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

// Reads the server's stream of events as it arrives (the Streams API). Each event is a
// block of lines ending in an empty line: "event: step" and "data: {...}".
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
      if (event === "start") answer.tracker.begin(data.time_limit_s);
      else if (event === "step") answer.tracker.step(data);
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

// -- adding documents -------------------------------------------------------------------

const ingest = {
  waiting: [],          // PDFs that are in the folders and were never ingested
  selected: new Set(),  // their source paths, as ticked
  uploads: [],          // what happened to each file sent from this computer
  job: null,            // the ingestion run, as the server last described it
  lines: 0,             // how many of its log lines the page has
  pollTimer: null,
  watching: false,      // true while this page shows a run that it has not put away
  confirmAbove: 25,
  maxUploadMb: 200,
  describeSet: false,
};

const JOB_STATES = {
  waiting: "Waiting for the answer in progress to finish",
  running: "Ingesting",
  stopping: "Stopping after the step in hand",
  done: "Finished",
  failed: "Failed",
  stopped: "Stopped",
};
const FILE_STATES = {
  waiting: "waiting",
  reading: "reading",
  between: "read, waiting for the next step",
  figures: "describing figures",
  indexing: "embedding and storing",
  done: "ingested",
  failed: "not ingested",
  left: "left for a later run",
};
const STEP_WORDS = { read: "Reading", figures: "Describing the figures of", index: "Embedding and storing" };

function jobActive(job) {
  return Boolean(job) && ["waiting", "running", "stopping"].includes(job.state);
}

function setIngesting(value) {
  if (view.ingesting === value) return;
  view.ingesting = value;
  updateComposer();
}

async function loadIngest() {
  let data;
  try {
    data = await getJSON("/api/ingest?since=" + ingest.lines);
  } catch (error) {
    showIngestError("The list of files could not be loaded: " + error.message);
    return;
  }
  ingest.waiting = data.waiting;
  ingest.confirmAbove = data.confirm_above;
  ingest.maxUploadMb = data.max_upload_mb;
  if (!ingest.describeSet) {
    $("describe-figures").checked = data.describe_figures;
    ingest.describeSet = true;
  }
  const names = new Set(ingest.waiting.map((item) => item.source_path));
  for (const name of [...ingest.selected]) if (!names.has(name)) ingest.selected.delete(name);
  $("waiting-folder")._tip = {
    title: "Where the PDFs are looked for",
    lines: ["Documents folder: " + data.folders.docs, "Uploads folder: " + data.folders.uploads, "The documents folder is paths.docs_dir in config.yaml (or config.local.yaml)."],
  };
  showIngestError(data.problem);
  takeJob(data.job);
  renderWaiting();
  showIngestView();
  if (jobActive(ingest.job) && !ingest.pollTimer) pollJob();
}

function takeJob(job) {
  const before = ingest.job;
  if (job && before && job.number === before.number) {
    job.lines = [...(before.lines || []), ...job.lines];
  }
  ingest.job = job;
  if (job) ingest.lines = job.line_count;
  if (jobActive(job)) ingest.watching = true;
  renderJobStrip();
}

function showIngestView() {
  const job = ingest.job;
  const progress = Boolean(job) && (jobActive(job) || ingest.watching);
  $("ingest-choose").hidden = progress;
  $("ingest-progress").hidden = !progress;
  $("ingest-start").hidden = progress;
  $("refresh-docs").hidden = progress;
  $("ingest-stop").hidden = !jobActive(job);
  $("ingest-stop").disabled = Boolean(job) && job.state === "stopping";
  $("ingest-again").hidden = !progress || jobActive(job);
  if (progress) renderJob();
  updateIngestButton();
}

function showIngestError(message) {
  const box = $("ingest-error");
  box.hidden = !message;
  box.textContent = message || "";
}

function renderWaiting() {
  const wanted = $("waiting-filter").value.trim().toLowerCase();
  const shown = ingest.waiting.filter(
    (item) => !wanted || item.title.toLowerCase().includes(wanted) || item.source_path.toLowerCase().includes(wanted),
  );
  const list = $("waiting-list");
  list.replaceChildren();
  const most = 300;
  for (const item of shown.slice(0, most)) {
    const row = el("li", "waiting-item");
    const label = el("label");
    const box = el("input");
    box.type = "checkbox";
    box.checked = ingest.selected.has(item.source_path);
    box.addEventListener("change", () => {
      if (box.checked) ingest.selected.add(item.source_path);
      else ingest.selected.delete(item.source_path);
      updateIngestButton();
    });
    const text = el("span", "w-text");
    text.append(
      el("span", "w-name", item.short_title),
      el("span", "w-meta", (item.root === "uploads" ? "added here · " : "") + item.name + " · " + fileSize(item.size)),
    );
    label.append(box, text);
    row.append(label);
    row._tip = { title: item.title, lines: ["File: " + item.source_path, fileSize(item.size)] };
    list.append(row);
  }
  if (shown.length > most) {
    list.append(el("li", "waiting-more", (shown.length - most) + " more. Use the filter to find them, or 'Select all shown'."));
  }
  ingest.shown = shown;
  $("waiting-count").textContent = wanted ? shown.length + " of " + ingest.waiting.length : String(ingest.waiting.length);
  const empty = $("waiting-empty");
  empty.hidden = shown.length > 0;
  empty.textContent = ingest.waiting.length
    ? "No file matches that."
    : "No PDF is waiting. Every PDF in the documents folder and the uploads folder has been ingested.";
  updateIngestButton();
}

function updateIngestButton() {
  const button = $("ingest-start");
  const many = ingest.selected.size;
  button.disabled = many === 0;
  button.textContent = many ? "Ingest " + plural(many, "document", "documents") : "Ingest";
}

async function uploadFiles(files) {
  showIngestError(null);
  for (const file of files) {
    const entry = { name: file.name, state: "copying", detail: fileSize(file.size) };
    ingest.uploads.unshift(entry);
    renderUploads();
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      entry.state = "refused";
      entry.detail = "Only PDF files can be added.";
    } else if (file.size > ingest.maxUploadMb * 1024 * 1024) {
      entry.state = "refused";
      entry.detail = "Larger than " + ingest.maxUploadMb + " MB.";
    } else {
      try {
        const response = await fetch("/api/upload?name=" + encodeURIComponent(file.name), {
          method: "POST",
          headers: { "Content-Type": "application/pdf" },
          body: file,
        });
        const body = await response.json().catch(() => ({}));
        if (!response.ok) {
          entry.state = "refused";
          entry.detail = body.error || "The server answered " + response.status + ".";
        } else if (body.status === "duplicate") {
          entry.state = "known";
          entry.detail = "Already ingested as: " + body.title;
        } else {
          entry.state = "added";
          entry.detail = fileSize(body.size) + " · ticked below";
          ingest.selected.add(body.source_path);
        }
      } catch (error) {
        entry.state = "refused";
        entry.detail = error.message;
      }
    }
    renderUploads();
  }
  await loadIngest();
}

function renderUploads() {
  const words = { copying: "copying", added: "added", known: "not needed", refused: "not added" };
  const list = $("upload-list");
  list.replaceChildren();
  for (const entry of ingest.uploads.slice(0, 30)) {
    const row = el("li", "upload " + entry.state);
    row.append(el("span", "u-name", entry.name), el("span", "u-state", words[entry.state]), el("span", "u-detail", entry.detail));
    list.append(row);
  }
}

async function startIngest(body) {
  showIngestError(null);
  let response;
  let reply;
  try {
    response = await fetch("/api/ingest", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    reply = await response.json().catch(() => ({}));
  } catch (error) {
    showIngestError(error.message);
    return;
  }
  if (response.status === 409 && reply.needs_confirmation) {
    if (window.confirm(reply.error)) await startIngest({ ...body, confirmed: true });
    return;
  }
  if (!response.ok) {
    showIngestError(reply.error || "The server answered " + response.status + ".");
    return;
  }
  ingest.lines = 0;
  ingest.job = null;
  ingest.selected.clear();
  ingest.uploads = [];
  renderUploads();
  takeJob(reply.job);
  setIngesting(true);
  showIngestView();
  pollJob();
}

async function pollJob() {
  clearTimeout(ingest.pollTimer);
  ingest.pollTimer = null;
  let data;
  try {
    data = await getJSON("/api/ingest?files=0&since=" + ingest.lines);
  } catch (error) {
    ingest.pollTimer = setTimeout(pollJob, 3000);
    return;
  }
  const wasActive = jobActive(ingest.job);
  takeJob(data.job);
  showIngestView();
  if (jobActive(ingest.job)) {
    ingest.pollTimer = setTimeout(pollJob, 1000);
    return;
  }
  setIngesting(false);
  if (wasActive || ingest.watching) {
    refreshDocuments();
    refreshStats();
    refreshStatus();
  }
}

function renderJobStrip() {
  const strip = $("job-strip");
  const job = ingest.job;
  if (!job || !(jobActive(job) || ingest.watching)) {
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  strip.className = "job-strip " + job.state;
  const finished = job.done + job.failed;
  let text = JOB_STATES[job.state];
  if (jobActive(job) && job.total) text += " " + Math.min(finished + 1, job.total) + " of " + job.total;
  else if (!jobActive(job)) text += ": " + job.done + " ingested" + (job.failed ? ", " + job.failed + " failed" : "");
  $("job-strip-text").textContent = text;
  $("job-strip-fill").style.width = (job.total ? (finished / job.total) * 100 : jobActive(job) ? 5 : 100) + "%";
}

function renderJob() {
  const job = ingest.job;
  if (!job) return;
  const finished = job.done + job.failed;
  $("job-title").textContent = JOB_STATES[job.state] + (job.total ? ": " + finished + " of " + plural(job.total, "document", "documents") : "");
  $("job-clock").textContent = clock(job.elapsed_s);
  $("job-fill").style.width = (job.total ? (finished / job.total) * 100 : jobActive(job) ? 4 : 100) + "%";
  $("job-bar").className = "bar " + job.state;

  const current = job.current;
  let now = "";
  if (job.state === "waiting") now = "It starts when the answer that is being written is finished.";
  else if (current && current.step === "models") now = "Loading the search models again, so that questions can be asked.";
  else if (current && current.name) {
    now = STEP_WORDS[current.step] + " " + current.name + " (" + current.position + " of " + current.of + ")";
    if (current.step === "figures" && current.figure) {
      now += ": figure " + current.figure.position + " of " + current.figure.of + ", page " + current.figure.page;
    }
  } else if (jobActive(job)) now = "Looking at the files";
  $("job-now").textContent = now;

  const list = $("job-files");
  list.replaceChildren();
  const most = 200;
  for (const file of job.files.slice(0, most)) {
    const row = el("li", "job-file " + file.state);
    row.append(el("span", "f-mark"), el("span", "f-name", file.name), el("span", "f-state", FILE_STATES[file.state] + (file.detail && file.state !== "failed" ? " · " + file.detail : "")));
    if (file.state === "failed") row.append(el("span", "f-problem", file.detail));
    for (const note of file.notes || []) row.append(el("span", "f-note", note));
    row._tip = { title: file.source_path, lines: [] };
    list.append(row);
  }
  if (job.files.length > most) list.append(el("li", "waiting-more", (job.files.length - most) + " more"));

  const lines = $("job-lines");
  lines.textContent = (job.lines || []).join("\n");
  if ($("job-log").open) lines.scrollTop = lines.scrollHeight;

  const result = $("job-result");
  result.replaceChildren();
  if (!jobActive(job)) {
    const report = job.report || {};
    const facts = [];
    if (job.error) result.append(el("div", "note error", job.error));
    facts.push(plural(report.files_ingested || 0, "document", "documents") + " ingested, " + plural(report.chunks_indexed || 0, "passage", "passages"));
    if (job.what === "refresh") facts.push(plural(report.files_skipped || 0, "document was", "documents were") + " already up to date");
    if (report.figures_described) facts.push(plural(report.figures_described, "figure", "figures") + " described by the vision model");
    facts.push("reading " + seconds(report.read_s || 0) + ", figures " + seconds(report.figures_s || 0) + ", embedding and storing " + seconds(report.index_s || 0));
    result.append(el("p", "job-summary", facts.join(" · ")));
    if (report.stopped) result.append(el("div", "note", "The run was stopped. The documents that were not finished are still in the list and can be ingested later; what was read of them is kept."));
    // What went wrong with a file, and what to know about one, is shown with that file above.
  }
}

function openIngest() {
  const dialog = $("ingest-dialog");
  if (!dialog.open) dialog.showModal();
  loadIngest();
}

$("add-docs").addEventListener("click", openIngest);
$("job-strip").addEventListener("click", openIngest);
$("ingest-close").addEventListener("click", () => $("ingest-dialog").close());
$("ingest-dialog").addEventListener("close", () => {
  hideTip();
  // Closing the dialog on a run that has ended puts its result away.
  if (ingest.watching && !jobActive(ingest.job)) {
    ingest.watching = false;
    renderJobStrip();
  }
});
$("pick-files").addEventListener("click", () => $("file-input").click());
$("file-input").addEventListener("change", (event) => {
  const files = [...event.target.files];
  event.target.value = "";
  if (files.length) uploadFiles(files);
});
const dropZone = $("drop-zone");
for (const name of ["dragenter", "dragover"]) {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("over");
  });
}
for (const name of ["dragleave", "drop"]) {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("over");
  });
}
dropZone.addEventListener("drop", (event) => {
  const files = [...event.dataTransfer.files];
  if (files.length) uploadFiles(files);
});
$("waiting-filter").addEventListener("input", renderWaiting);
$("select-shown").addEventListener("click", () => {
  for (const item of ingest.shown || []) ingest.selected.add(item.source_path);
  renderWaiting();
});
$("select-none").addEventListener("click", () => {
  ingest.selected.clear();
  renderWaiting();
});
$("ingest-start").addEventListener("click", () => {
  if (!ingest.selected.size) return;
  startIngest({ files: [...ingest.selected], describe_figures: $("describe-figures").checked });
});
$("refresh-docs").addEventListener("click", () => {
  const sure = window.confirm(
    "Every document that is already ingested is checked. Those whose file or settings have changed are ingested again. New files are left alone. Go ahead?",
  );
  if (sure) startIngest({ refresh: true, describe_figures: $("describe-figures").checked });
});
$("ingest-stop").addEventListener("click", async () => {
  $("ingest-stop").disabled = true;
  try {
    await fetch("/api/ingest/stop", { method: "POST" });
  } catch (error) {
    // the next poll shows where the run stands
  }
  pollJob();
});
$("ingest-again").addEventListener("click", () => {
  ingest.watching = false;
  renderJobStrip();
  loadIngest();
});

// -- composer --------------------------------------------------------------------------

function updateComposer() {
  const question = $("question");
  $("send").hidden = view.busy;
  $("stop").hidden = !view.busy;
  $("send").disabled = !view.canAsk || view.ingesting || !question.value.trim();
  question.disabled = view.busy;
  question.placeholder = view.ingesting
    ? "Documents are being ingested. Questions can be asked when that is done."
    : "Ask a question about the documents";
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
  if (!text || view.busy || !view.canAsk || view.ingesting) return;
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
