"""Small PDFs and pictures made on the spot, for the tests that run the real models.

They are built without any PDF library, so that the tests need nothing beyond what the
project already installs. tests/test_pdf_samples.py checks that the files are valid.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


def text_pdf(path: Path, lines: list[str]) -> Path:
    """Write a one-page PDF with a real text layer.

    The first line is set larger, as a heading. Only plain ASCII without brackets or
    backslashes is supported, which is all the tests need.
    """
    for line in lines:
        if not line.isascii() or any(character in line for character in "()\\"):
            raise ValueError(f"Not supported in this simple PDF writer: {line!r}")
    content = ["BT", "/F1 18 Tf", "72 760 Td", f"({lines[0]}) Tj", "/F1 11 Tf"]
    for line in lines[1:]:
        content += ["0 -22 Td", f"({line}) Tj"]
    content.append("ET")
    stream = "\n".join(content).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
    ]
    body = b"%PDF-1.4\n"
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body += str(number).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(body)
    body += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n0000000000 65535 f \n"
    for offset in offsets:
        body += f"{offset:010d} 00000 n \n".encode()
    body += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    path.write_bytes(body)
    return path


def scanned_pdf(path: Path, lines: list[str]) -> Path:
    """Write a one-page PDF that is only a picture of text, like a scan. It has no text layer."""
    page = Image.new("RGB", (1240, 1754), "white")  # A4 at 150 dots per inch
    draw = ImageDraw.Draw(page)
    for number, line in enumerate(lines):
        draw.text((120, 160 + number * 70), line, fill="black", font_size=38)
    page.save(path, "PDF", resolution=150.0)
    return path


def flowchart_png(path: Path) -> Path:
    """Draw a flowchart of three boxes, top to bottom, with one labelled arrow."""
    image = Image.new("RGB", (900, 700), "white")
    draw = ImageDraw.Draw(image)
    boxes = {
        "Application received": (300, 40, 600, 130),
        "Credit score check": (300, 250, 600, 340),
        "Manual review": (300, 500, 600, 590),
    }
    for label, box in boxes.items():
        draw.rectangle(box, outline="black", width=4)
        draw.text((box[0] + 25, box[1] + 28), label, fill="black", font_size=26)
    for top, bottom in ((130, 250), (340, 500)):
        draw.line((450, top, 450, bottom), fill="black", width=4)
        draw.polygon([(440, bottom - 18), (460, bottom - 18), (450, bottom)], fill="black")
    draw.text((470, 400), "score below the cut-off", fill="black", font_size=22)
    image.save(path)
    return path
