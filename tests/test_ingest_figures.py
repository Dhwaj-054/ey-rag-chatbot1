"""Figure descriptions: reading the vision model's reply, and the cache.

The vision model is a real ``LLMClient`` talking to a faked server (tests/fakes.py).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from PIL import Image

from docqa.db import Database, description_from_json, description_to_json
from docqa.ingest.figures import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_prompt,
    describe_figure,
    describe_figure_cached,
    description_text,
    is_informative,
    parse_reply,
)
from docqa.schema import Element, ElementType, FigureDescription, FlowchartEdge
from tests.fakes import ScriptedLLM

FLOWCHART_REPLY = json.dumps(
    {
        "figure_type": "Flowchart",
        "title": "Loan application process",
        "axes_and_legend": None,
        "readable_values": [],
        "nodes": ["Application received", "Credit score check", "Manual review", "Approve"],
        "edges": [
            {"source": "Application received", "target": "Credit score check", "condition": None},
            {"source": "Credit score check", "target": "Approve", "condition": "score at or above the cut-off"},
            {"source": "Credit score check", "target": "Manual review", "condition": "score below the cut-off"},
        ],
    }
)


# -- reading the reply ----------------------------------------------------------------


def test_a_clean_json_reply_is_read_field_by_field():
    description = parse_reply(FLOWCHART_REPLY, "a" * 64, "qwen2.5vl:7b")

    assert description.image_sha256 == "a" * 64
    assert description.model == "qwen2.5vl:7b"
    assert description.figure_type == "flowchart"
    assert description.title == "Loan application process"
    assert description.axes_and_legend is None
    assert description.nodes[0] == "Application received"
    assert description.edges[1] == FlowchartEdge(
        "Credit score check", "Approve", "score at or above the cut-off"
    )
    assert description.edges[0].condition is None
    assert description.raw_text is None


def test_text_and_code_fences_around_the_json_are_ignored():
    reply = f"Here is the description:\n```json\n{FLOWCHART_REPLY}\n```\nI hope this helps."

    assert parse_reply(reply, "a" * 64, "m") == parse_reply(FLOWCHART_REPLY, "a" * 64, "m")


def test_values_of_the_wrong_type_are_dropped_or_tidied():
    reply = json.dumps(
        {
            "figure_type": "bar chart",
            "title": "null",
            "axes_and_legend": "  x: year;\n y: crore  ",
            "readable_values": ["FY24: 1,250 crore", 42, None, "", {"not": "text"}],
            "nodes": "a single box",
            "edges": [
                "A -> B [if approved]",
                "C → D",
                {"source": "E"},
                17,
            ],
        }
    )

    description = parse_reply(reply, "a" * 64, "m")

    assert description.title is None
    assert description.axes_and_legend == "x: year; y: crore"
    assert description.readable_values == ("FY24: 1,250 crore", "42")
    assert description.nodes == ("a single box",)
    assert description.edges == (FlowchartEdge("A", "B", "if approved"), FlowchartEdge("C", "D"))


def test_a_reply_that_is_not_json_is_kept_as_written():
    description = parse_reply("The figure shows a pie chart of deposits.", "a" * 64, "m")

    assert description.figure_type == "unknown"
    assert description.raw_text == "The figure shows a pie chart of deposits."
    assert description.nodes == () and description.readable_values == ()
    assert description_text(description, "Figure 4").splitlines() == [
        "Figure description (the model's reply could not be structured):",
        "Caption: Figure 4",
        "The figure shows a pie chart of deposits.",
    ]


def test_description_text_for_a_chart():
    description = FigureDescription(
        image_sha256="a" * 64,
        model="m",
        figure_type="bar chart",
        title="Net profit",
        axes_and_legend="x: financial year; y: net profit in crore",
        readable_values=("FY 2023-24: 1,100 crore", "FY 2024-25: 1,250 crore"),
    )

    assert description_text(description).splitlines() == [
        "Figure (bar chart): Net profit",
        "Axes and legend: x: financial year; y: net profit in crore",
        "Values printed in the figure:",
        "- FY 2023-24: 1,100 crore",
        "- FY 2024-25: 1,250 crore",
    ]


def test_logos_and_photographs_are_informative_only_if_something_is_readable():
    def of_type(figure_type: str, **more) -> FigureDescription:
        return FigureDescription("a" * 64, "m", figure_type, None, None, **more)

    assert not is_informative(of_type("logo"))
    assert not is_informative(of_type("signature or stamp"))
    assert is_informative(of_type("photograph", readable_values=("Branch code: 0421",)))
    assert is_informative(of_type("diagram"))
    assert is_informative(of_type("unknown", raw_text="something"))


def test_the_caption_is_given_to_the_model_as_context():
    assert "caption" not in build_prompt("").lower()
    assert '"Figure 1: Process"' in build_prompt("Figure 1: Process")


def test_descriptions_survive_the_trip_through_the_database():
    description = parse_reply(FLOWCHART_REPLY, "a" * 64, "qwen2.5vl:7b")
    assert description_from_json(description_to_json(description)) == description

    unparsed = parse_reply("free text", "b" * 64, "m")
    assert description_from_json(description_to_json(unparsed)) == unparsed


# -- calling the model, and the cache -------------------------------------------------


@pytest.fixture
def figure(tmp_path) -> Element:
    path = tmp_path / "crop.png"
    Image.new("RGB", (120, 80), "white").save(path)
    return Element("d", "d:e00003", 2, None, ElementType.FIGURE, "Figure 1: Process", image_path=path)


def test_the_vision_model_gets_the_image_the_instruction_and_the_system_prompt(tmp_cfg, figure):
    llm = ScriptedLLM(tmp_cfg, FLOWCHART_REPLY)
    database = Database(tmp_cfg.paths.sqlite_path)

    description = describe_figure(figure, llm.client, tmp_cfg, db=database)

    assert description.figure_type == "flowchart"
    request = llm.requests[0]
    assert request["model"] == "qwen2.5vl:7b"
    assert request["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    text_part, image_part = request["messages"][1]["content"]
    assert text_part["text"] == build_prompt("Figure 1: Process")
    sent = base64.b64decode(image_part["image_url"]["url"].removeprefix("data:image/png;base64,"))
    assert sent == figure.image_path.read_bytes()


def test_an_unchanged_figure_is_never_described_twice(tmp_cfg, figure, tmp_path):
    llm = ScriptedLLM(tmp_cfg, FLOWCHART_REPLY)
    database = Database(tmp_cfg.paths.sqlite_path)

    first, first_cached = describe_figure_cached(figure, llm.client, tmp_cfg, database)
    second, second_cached = describe_figure_cached(figure, llm.client, tmp_cfg, database)
    # The same picture saved under another name, as after re-ingesting the document.
    copy = tmp_path / "copy.png"
    copy.write_bytes(figure.image_path.read_bytes())
    moved = Element("d", "d:e00009", 2, None, ElementType.FIGURE, "", image_path=copy)
    third, third_cached = describe_figure_cached(moved, llm.client, tmp_cfg, database)

    assert (first_cached, second_cached, third_cached) == (False, True, True)
    assert first == second == third
    assert len(llm.requests) == 1
    assert database.count_figure_descriptions() == 1


def test_a_different_picture_or_a_new_instruction_version_is_described_again(tmp_cfg, figure, tmp_path):
    llm = ScriptedLLM(tmp_cfg, FLOWCHART_REPLY)
    database = Database(tmp_cfg.paths.sqlite_path)
    describe_figure(figure, llm.client, tmp_cfg, db=database)

    other_path = tmp_path / "other.png"
    Image.new("RGB", (120, 80), "black").save(other_path)
    other = Element("d", "d:e00004", 3, None, ElementType.FIGURE, "", image_path=other_path)
    describe_figure(other, llm.client, tmp_cfg, db=database)

    assert len(llm.requests) == 2
    image_hash = describe_figure(figure, llm.client, tmp_cfg, db=database).image_sha256
    assert database.get_figure_description(image_hash, "qwen2.5vl:7b", PROMPT_VERSION) is not None
    assert database.get_figure_description(image_hash, "qwen2.5vl:7b", "another-version") is None
    assert database.get_figure_description(image_hash, "another-model", PROMPT_VERSION) is None


def test_only_figures_with_a_saved_image_can_be_described(tmp_cfg):
    llm = ScriptedLLM(tmp_cfg, FLOWCHART_REPLY)
    no_image = Element("d", "d:e1", 1, None, ElementType.FIGURE, "caption", image_path=None)
    not_a_figure = Element("d", "d:e2", 1, None, ElementType.TEXT, "text", image_path=Path("x.png"))

    for element in (no_image, not_a_figure):
        with pytest.raises(ValueError, match="not a figure"):
            describe_figure(element, llm.client, tmp_cfg, db=Database(tmp_cfg.paths.sqlite_path))
    assert llm.requests == []
