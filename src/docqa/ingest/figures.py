"""Describing a figure with the vision model, once.

A picture cannot be searched. During ingestion the vision model writes a description of
each figure, and that description is searched like any other text. For a flowchart the
description lists every box and every arrow, because a flowchart's meaning is in its
arrows.

Descriptions are cached in SQLite by the hash of the cropped image, the model name and
``PROMPT_VERSION``. The vision model is the slowest step of the whole pipeline on a
CPU, so an unchanged figure is never described twice. Changing the instruction below
means raising ``PROMPT_VERSION``; every figure is then described again on the next run.

The description is model-generated and can be wrong. It is stored as such
(``Chunk.model_generated``), and an answer that uses it must say so.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

from docqa.config import Config
from docqa.db import Database
from docqa.llm import LLMClient
from docqa.schema import Element, ElementType, FigureDescription, FlowchartEdge

log = logging.getLogger(__name__)

PROMPT_VERSION = "1"

SYSTEM_PROMPT = (
    "You describe figures taken from banking documents, so that they can be found by a "
    "text search. You report only what is printed in the figure. You never estimate, "
    "interpret or add information."
)

_INSTRUCTION = """\
Describe the figure in the image. Reply with one JSON object and nothing else, with \
exactly these keys:

"figure_type": one of "bar chart", "line chart", "pie chart", "other chart", \
"flowchart", "organisation chart", "diagram", "table image", "form or screenshot", \
"map", "photograph", "logo", "signature or stamp", "other".
"title": the title printed in the figure, or null if it has none.
"axes_and_legend": for a chart, what the horizontal axis shows, what the vertical axis \
shows with its unit, and the legend entries. Otherwise null.
"readable_values": a list of the values printed in the figure that you can read \
clearly, each with its label, for example "FY 2024-25 net profit: 1,250 crore". Copy \
numbers exactly as printed, with their units. Do not estimate a value from the height \
of a bar or the position of a point. If no value is printed clearly, use [].
"nodes": for a flowchart, organisation chart or process diagram, the text of every \
box, in order. Otherwise [].
"edges": for a flowchart, organisation chart or process diagram, one entry per arrow: \
{"source": "text of the box the arrow leaves", "target": "text of the box it points \
to", "condition": "the label on the arrow, such as Yes or No, or null"}. Otherwise [].

Rules:
- Copy text exactly as it is printed. Do not translate, correct or complete it.
- If you cannot read something, leave it out. Do not guess.
- Do not say what the figure means and do not draw conclusions from it."""

# Figure types that carry nothing worth searching for unless something readable is in them.
DECORATIVE_TYPES = frozenset({"photograph", "logo", "signature or stamp"})

_MAX_FIELD = 2000
_EDGE_TEXT = re.compile(r"^(?P<source>.+?)\s*(?:->|→|=>)\s*(?P<target>.+?)(?:\s*\[(?P<condition>.+)\])?$")


def build_prompt(caption: str = "") -> str:
    """The instruction sent with the image. The printed caption, if any, is given as context."""
    if not caption.strip():
        return _INSTRUCTION
    return (
        f"{_INSTRUCTION}\n\nThe caption printed with this figure in the document is: "
        f"{json.dumps(caption.strip())}. Use it only to understand what the figure is."
    )


def parse_reply(reply: str, image_sha256: str, model: str) -> FigureDescription:
    """Read the model's reply as a figure description.

    A 7B model does not always return clean JSON. Text around the JSON object is
    ignored, and values of the wrong type are dropped. If no JSON object can be found
    at all, the reply is kept as ``raw_text`` so that the work is not thrown away.
    """
    data = _find_json_object(reply)
    if data is None:
        return FigureDescription(
            image_sha256=image_sha256,
            model=model,
            figure_type="unknown",
            title=None,
            axes_and_legend=None,
            raw_text=reply.strip()[: _MAX_FIELD * 2] or None,
        )
    return FigureDescription(
        image_sha256=image_sha256,
        model=model,
        figure_type=(_text(data.get("figure_type")) or "other").lower(),
        title=_text(data.get("title")),
        axes_and_legend=_text(data.get("axes_and_legend")),
        readable_values=_texts(data.get("readable_values")),
        nodes=_texts(data.get("nodes")),
        edges=_edges(data.get("edges")),
    )


def is_informative(description: FigureDescription) -> bool:
    """False for a logo, photograph, signature or stamp with nothing readable in it."""
    if description.raw_text:
        return True
    if description.figure_type in DECORATIVE_TYPES:
        return bool(description.readable_values or description.nodes or description.edges)
    return True


def description_text(description: FigureDescription, caption: str = "") -> str:
    """A figure description as the text that is embedded, searched and shown as evidence."""
    if description.raw_text:
        lines = ["Figure description (the model's reply could not be structured):"]
        if caption.strip():
            lines.append(f"Caption: {caption.strip()}")
        lines.append(description.raw_text)
        return "\n".join(lines)

    first = f"Figure ({description.figure_type})"
    lines = [f"{first}: {description.title}" if description.title else first]
    if caption.strip():
        lines.append(f"Caption: {caption.strip()}")
    if description.axes_and_legend:
        lines.append(f"Axes and legend: {description.axes_and_legend}")
    if description.readable_values:
        lines.append("Values printed in the figure:")
        lines.extend(f"- {value}" for value in description.readable_values)
    if description.nodes:
        lines.append("Boxes: " + "; ".join(description.nodes))
    if description.edges:
        lines.append("Arrows:")
        for edge in description.edges:
            condition = f" [{edge.condition}]" if edge.condition else ""
            lines.append(f"- {edge.source} -> {edge.target}{condition}")
    return "\n".join(lines)


def describe_figure(
    element: Element, llm: LLMClient, config: Config, *, db: Database | None = None
) -> FigureDescription:
    """Have the vision model describe one figure.

    The description gives the figure type, title, axes and legend, and any values that
    are clearly readable. For a flowchart it also lists the nodes and the edges
    (``A -> B``, with the condition if there is one).

    Descriptions are cached by the hash of the cropped image: the vision model is
    never run again on an unchanged figure.
    """
    return describe_figure_cached(element, llm, config, db)[0]


def describe_figure_cached(
    element: Element, llm: LLMClient, config: Config, db: Database | None = None
) -> tuple[FigureDescription, bool]:
    """As ``describe_figure``, and also says whether the description came from the cache."""
    if element.element_type is not ElementType.FIGURE or element.image_path is None:
        raise ValueError(f"Element {element.element_id} is not a figure with a saved image.")
    from docqa import resources

    db = db or resources.get_database(config)
    image = element.image_path.read_bytes()
    image_sha256 = hashlib.sha256(image).hexdigest()
    model = config.llm.vision_model

    cached = db.get_figure_description(image_sha256, model, PROMPT_VERSION)
    if cached is not None:
        return cached, True

    result = llm.describe_image(
        build_prompt(element.text), image, mime_type="image/png", system=SYSTEM_PROMPT, model=model
    )
    if result.finish_reason == "length":
        log.warning(
            "The description of %s was cut off at llm.max_tokens; it may be incomplete.",
            element.image_path.name,
        )
    description = parse_reply(result.text, image_sha256, model)
    db.save_figure_description(description, PROMPT_VERSION, result.text, result.latency_s)
    return description, False


def _find_json_object(reply: str) -> dict[str, Any] | None:
    """The first JSON object in ``reply``, or ``None``. Text around it is ignored."""
    decoder = json.JSONDecoder()
    position = reply.find("{")
    while position != -1:
        try:
            value, _end = decoder.raw_decode(reply, position)
        except ValueError:
            position = reply.find("{", position + 1)
            continue
        if isinstance(value, dict):
            return value
        position = reply.find("{", position + 1)
    return None


def _text(value: Any) -> str | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)
    if not isinstance(value, str):
        return None
    value = " ".join(value.split())
    if not value or value.lower() in ("null", "none", "n/a"):
        return None
    return value[:_MAX_FIELD]


def _texts(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return ()
    return tuple(text for text in (_text(item) for item in value) if text)


def _edges(value: Any) -> tuple[FlowchartEdge, ...]:
    if not isinstance(value, list):
        return ()
    edges = []
    for item in value:
        if isinstance(item, dict):
            source, target = _text(item.get("source")), _text(item.get("target"))
            condition = _text(item.get("condition"))
        elif isinstance(item, str) and (match := _EDGE_TEXT.match(item.strip())):
            source, target = _text(match["source"]), _text(match["target"])
            condition = _text(match["condition"])
        else:
            continue
        if source and target:
            edges.append(FlowchartEdge(source, target, condition))
    return tuple(edges)
