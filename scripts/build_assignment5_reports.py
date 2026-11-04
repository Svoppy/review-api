"""Build styled DOCX/PDF submission reports from docs/assignment5 Markdown."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "assignment5"
OUTPUT = ROOT / "output" / "assignment5"


def add_inline(paragraph, text: str) -> None:
    pattern = re.compile(r"(\*\*.+?\*\*|\*.+?\*|\[[^]]+\]\(https?://[^)]+\))")
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() > cursor:
            paragraph.add_run(text[cursor : match.start()])
        token = match.group(0)
        if token.startswith("**"):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
        elif token.startswith("*"):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
        else:
            label, url = re.match(r"\[([^]]+)\]\(([^)]+)\)", token).groups()
            run = paragraph.add_run(f"{label} ({url})")
            run.font.color.rgb = RGBColor(31, 78, 121)
        cursor = match.end()
    if cursor < len(text):
        paragraph.add_run(text[cursor:])


def make_docx(markdown: Path, destination: Path, practical: bool = False) -> None:
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.65)
    section.left_margin = Inches(0.8)
    section.right_margin = Inches(0.8)

    normal = doc.styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(12)
    normal.font.color.rgb = RGBColor(38, 50, 56)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(5)
    for stylename, size, color in (("Heading 1", 17, "17365D"), ("Heading 2", 13, "1F4E79")):
        style = doc.styles[stylename]
        style.font.name = "Aptos Display"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.keep_with_next = True
        style.paragraph_format.space_before = Pt(10 if stylename == "Heading 1" else 7)
        style.paragraph_format.space_after = Pt(3)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header.add_run("REVIEWGUARD  /  ASSIGNMENT 5").font.color.rgb = RGBColor(110, 125, 138)
    header.runs[0].font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fr = footer.add_run("DevOps and CI/CD in Research Projects  •  ")
    fr.font.size = Pt(8)
    # Keep the footer understated while leaving the body at the requested 12pt.

    raw = markdown.read_text(encoding="utf-8").splitlines()
    first_title = True
    in_references = False
    for line in raw:
        line = line.strip()
        if not line:
            continue
        if line == "<!-- PAGE_BREAK -->":
            doc.add_page_break()
            continue
        if line == "<!-- JENKINS_SCREENSHOT -->":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run("Figure 1. Successful Jenkins pipeline evidence is supplied alongside this report.")
            r.italic = True
            r.font.size = Pt(10)
            continue
        if line.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(line[2:])
            run.font.name = "Aptos Display"
            run.font.size = Pt(21 if first_title else 17)
            run.bold = True
            run.font.color.rgb = RGBColor(23, 54, 93)
            first_title = False
        elif line.startswith("## "):
            heading = line[3:]
            p = doc.add_paragraph(heading, style="Heading 1")
            in_references = heading.lower() == "references"
        elif line.startswith("### "):
            doc.add_paragraph(line[4:], style="Heading 2")
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            add_inline(p, line[2:])
        elif re.match(r"\d+\. ", line):
            p = doc.add_paragraph(style="List Number")
            add_inline(p, re.sub(r"^\d+\. ", "", line))
        else:
            p = doc.add_paragraph()
            if in_references:
                p.paragraph_format.left_indent = Inches(0.18)
                p.paragraph_format.first_line_indent = Inches(-0.18)
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(3)
                add_inline(p, line)
                for run in p.runs:
                    run.font.size = Pt(10)
            else:
                add_inline(p, line)
    doc.core_properties.title = markdown.stem.replace("_", " ").title()
    doc.core_properties.subject = "Assignment 5: DevOps and CI/CD for ReviewGuard"
    doc.core_properties.author = "ReviewGuard research project"
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.save(destination)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    theory = [
        line
        for line in (SOURCE / "theoretical_report.md").read_text(encoding="utf-8").splitlines()
        if line.strip() != "**Author: Dias Kazikhanov**"
    ]
    practical = [
        line
        for line in (SOURCE / "practical_report.md").read_text(encoding="utf-8").splitlines()
        if line.strip() != "**Author: Dias Kazikhanov**"
    ]
    combined = [
        "# Assignment 5: DevOps and CI/CD for ReviewGuard",
        "**Theoretical Research and Practical Implementation Report**",
        "**Dias Kazikhanov**",
        "",
        "## Part 1. Theoretical report",
        *theory[1:],
        "",
        "<!-- PAGE_BREAK -->",
        "",
        "## Part 2. Practical report",
        *practical[1:],
    ]
    combined_path = SOURCE / "assignment5_report.md"
    combined_path.write_text("\n".join(combined) + "\n", encoding="utf-8")
    make_docx(combined_path, OUTPUT / "assignment5_report.docx")


if __name__ == "__main__":
    main()
