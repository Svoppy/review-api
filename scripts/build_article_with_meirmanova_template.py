from __future__ import annotations

import re
import shutil
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(
    "/Users/diaskazikhanov/Library/Containers/net.whatsapp.WhatsApp/Data/tmp/documents/"
    "9ECC3AC2-9CE2-4648-9A4B-B26E9ADE2429/Мейрманова_статья.docx"
)
SOURCE = ROOT / "docs/article_improved_en.docx"
OUTPUT = ROOT / "docs/article_improved_en_meirmanova_reviewed.docx"

FONT = "Times New Roman"
BLUE = "2F5597"
LIGHT_BLUE = "F2F5FA"
GRID = "B7C3D0"
BLACK = "000000"


def set_run_font(run, *, size=11.5, bold=False, italic=False, color=BLACK):
    run.font.name = FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    for key in ("ascii", "hAnsi", "cs", "eastAsia"):
        rfonts.set(qn(f"w:{key}"), FONT)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_style_font(style, *, size=11.5, bold=False, italic=False):
    style.font.name = FONT
    style._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), FONT)
    style._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), FONT)
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = italic
    style.font.color.rgb = RGBColor.from_string(BLACK)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_borders(cell, color=GRID, size="5"):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        node = borders.find(qn("w:" + edge))
        if node is None:
            node = OxmlElement("w:" + edge)
            borders.append(node)
        node.set(qn("w:val"), "single")
        node.set(qn("w:sz"), size)
        node.set(qn("w:space"), "0")
        node.set(qn("w:color"), color)


def set_cell_margins(cell, top=70, start=85, bottom=70, end=85):
    tc_pr = cell._tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn("w:" + name))
        if node is None:
            node = OxmlElement("w:" + name)
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    node = OxmlElement("w:tblHeader")
    node.set(qn("w:val"), "true")
    tr_pr.append(node)


def set_no_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    tr_pr.append(OxmlElement("w:cantSplit"))


def set_table_geometry(table, widths):
    """Write explicit DXA table width/grid values for Word and geometry audits."""
    dxa_widths = [int(round(width * 1440)) for width in widths]
    total = sum(dxa_widths)
    tbl = table._tbl
    tbl_pr = tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.insert(0, tbl_w)
    tbl_w.set(qn("w:w"), str(total))
    tbl_w.set(qn("w:type"), "dxa")
    tbl_ind = tbl_pr.find(qn("w:tblInd"))
    if tbl_ind is None:
        tbl_ind = OxmlElement("w:tblInd")
        tbl_pr.append(tbl_ind)
    tbl_ind.set(qn("w:w"), "85")
    tbl_ind.set(qn("w:type"), "dxa")

    grid = tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for dxa_width in dxa_widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(dxa_width))
        grid.append(col)


def set_keep_next(paragraph):
    ppr = paragraph._p.get_or_add_pPr()
    if ppr.find(qn("w:keepNext")) is None:
        ppr.append(OxmlElement("w:keepNext"))


def set_cell_text(cell, text, *, size=8.5, bold=False, color=BLACK, alignment=WD_ALIGN_PARAGRAPH.LEFT):
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = alignment
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.line_spacing = 1.0
    run = p.add_run(str(text))
    set_run_font(run, size=size, bold=bold, color=color)


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.27)
    section.page_height = Inches(11.69)
    for attr in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
        setattr(section, attr, Inches(0.79))
    section.header_distance = Inches(0.35)
    section.footer_distance = Inches(0.35)

    for style_name, size, bold, italic in (
        ("Normal", 11.5, False, False),
        ("Body Text", 11.5, False, False),
        ("Heading 1", 12, True, False),
        ("Heading 2", 12, False, True),
        ("Title", 12.5, True, False),
        ("Subtitle", 11.5, False, True),
        ("List Paragraph", 11, False, False),
    ):
        if style_name in doc.styles:
            set_style_font(doc.styles[style_name], size=size, bold=bold, italic=italic)

    normal = doc.styles["Normal"]
    normal.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    normal.paragraph_format.first_line_indent = Inches(0.5)
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.line_spacing = 1.0


def clear_body(doc):
    body = doc.element.body
    for child in list(body):
        if child.tag != qn("w:sectPr"):
            body.remove(child)


def clear_header_footer(section):
    for part in (section.header, section.footer):
        element = part._element
        for child in list(element):
            if child.tag != qn("w:p"):
                continue
            # Keep one empty paragraph so Word retains a valid header/footer part.
            for pchild in list(child):
                child.remove(pchild)


def paragraph_text(paragraph):
    # Paragraph.text omits hyperlink runs in some python-docx versions. Reading all
    # text nodes keeps DOI/URL text from the source article.
    return "".join(node.text or "" for node in paragraph._p.iter(qn("w:t")))


def iter_source_blocks(doc):
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            p = Paragraph(child, doc)
            yield "p", p, paragraph_text(p)
        elif child.tag == qn("w:tbl"):
            yield "tbl", Table(child, doc), None


def set_para(paragraph, *, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0.5, hanging=None, left=0, before=0, after=0, line=1.0):
    pf = paragraph.paragraph_format
    pf.alignment = alignment
    pf.left_indent = Inches(left) if left else None
    pf.first_line_indent = Inches(indent) if indent else None
    if hanging is not None:
        pf.first_line_indent = Inches(-hanging)
    pf.space_before = Pt(before)
    pf.space_after = Pt(after)
    pf.line_spacing = line


def add_plain_paragraph(doc, text, *, italic=False, bold=False, size=11.5, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0.5, left=0, hanging=None, before=0, after=0, keep=False):
    p = doc.add_paragraph()
    set_para(p, alignment=alignment, indent=indent, left=left, hanging=hanging, before=before, after=after)
    run = p.add_run(text)
    set_run_font(run, size=size, bold=bold, italic=italic)
    if keep:
        set_keep_next(p)
    return p


def add_author_line(doc, name, role):
    p = doc.add_paragraph()
    set_para(p, alignment=WD_ALIGN_PARAGRAPH.RIGHT, indent=0, after=0)
    r1 = p.add_run(name)
    set_run_font(r1, size=11.5, bold=True, italic=True)
    r2 = p.add_run(role)
    set_run_font(r2, size=11.5, italic=True)
    return p


def add_section_heading(doc, text, *, sub=False):
    p = doc.add_paragraph()
    set_para(p, alignment=WD_ALIGN_PARAGRAPH.LEFT, indent=0, before=5 if not sub else 3, after=2)
    run = p.add_run(text)
    set_run_font(run, size=12, bold=not sub, italic=sub)
    set_keep_next(p)
    return p


def add_caption(doc, text):
    text = re.sub(r"^(Table|Figure)\s+(\d+)\s*[.:]\s*", r"\1 \2 – ", text.strip())
    p = doc.add_paragraph()
    set_para(p, alignment=WD_ALIGN_PARAGRAPH.CENTER, indent=0, before=4, after=2)
    run = p.add_run(text)
    set_run_font(run, size=10.5, italic=True)
    set_keep_next(p)
    return p


def add_bullet(doc, text):
    p = doc.add_paragraph()
    set_para(p, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0, left=0.25, hanging=0.16, after=1)
    run = p.add_run("• " + text)
    set_run_font(run, size=11.2)
    return p


def add_table_from_source(doc, src_table, index):
    rows = [[cell.text.replace("\n", " / ").strip() for cell in row.cells] for row in src_table.rows]
    if not rows:
        return
    ncols = len(rows[0])
    # The reference template distributes table columns evenly. This gives
    # short metric headers enough room and prevents values such as 7,000 or
    # -0.0013 from being split into individual characters.
    widths = [6.68 / ncols] * ncols
    table = doc.add_table(rows=1, cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_geometry(table, widths)
    for col, value in enumerate(rows[0]):
        cell = table.rows[0].cells[col]
        cell.width = Inches(widths[col])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, BLUE)
        set_cell_borders(cell)
        set_cell_margins(cell)
        set_cell_text(cell, value, size=8.5, bold=True, color="FFFFFF", alignment=WD_ALIGN_PARAGRAPH.CENTER)
    set_repeat_header(table.rows[0])
    for row_i, values in enumerate(rows[1:]):
        row = table.add_row()
        set_no_split(row)
        for col, value in enumerate(values):
            cell = row.cells[col]
            cell.width = Inches(widths[col])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_i % 2 == 1:
                set_cell_shading(cell, LIGHT_BLUE)
            set_cell_borders(cell)
            set_cell_margins(cell)
            align = WD_ALIGN_PARAGRAPH.CENTER if (col == 1 or (ncols == 5 and col in (2, 4))) else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cell, value, size=8.35, alignment=align)

    # Table 2 in the source draft did not include trivial class-prior
    # baselines. Add them explicitly so every learned system is compared with
    # a reproducible lower bound on the fixed test split.
    if index == 2:
        for row_i, values in enumerate(
            [
                ["Majority baseline", "Sentiment", "0.2737", "0.6965"],
                ["Majority baseline", "Provenance", "0.3333", "0.5000"],
            ],
            start=len(rows) - 1,
        ):
            row = table.add_row()
            set_no_split(row)
            for col, value in enumerate(values):
                cell = row.cells[col]
                cell.width = Inches(widths[col])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                if row_i % 2 == 1:
                    set_cell_shading(cell, LIGHT_BLUE)
                set_cell_borders(cell)
                set_cell_margins(cell)
                align = WD_ALIGN_PARAGRAPH.CENTER if col == 1 else WD_ALIGN_PARAGRAPH.LEFT
                set_cell_text(cell, value, size=8.35, alignment=align)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_manual_table(doc, caption, headers, rows, size=8.25):
    add_caption(doc, caption)
    ncols = len(headers)
    widths = [6.68 / ncols] * ncols
    table = doc.add_table(rows=1, cols=ncols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_geometry(table, widths)
    for col, value in enumerate(headers):
        cell = table.rows[0].cells[col]
        cell.width = Inches(widths[col])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, BLUE)
        set_cell_borders(cell)
        set_cell_margins(cell)
        set_cell_text(cell, value, size=size, bold=True, color="FFFFFF", alignment=WD_ALIGN_PARAGRAPH.CENTER)
    set_repeat_header(table.rows[0])
    for row_i, values in enumerate(rows):
        row = table.add_row()
        set_no_split(row)
        for col, value in enumerate(values):
            cell = row.cells[col]
            cell.width = Inches(widths[col])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if row_i % 2 == 1:
                set_cell_shading(cell, LIGHT_BLUE)
            set_cell_borders(cell)
            set_cell_margins(cell)
            alignment = WD_ALIGN_PARAGRAPH.CENTER if col >= 2 else WD_ALIGN_PARAGRAPH.LEFT
            set_cell_text(cell, value, size=size, alignment=alignment)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)


def add_review_metrics(doc):
    add_manual_table(
        doc,
        "Table 4 – Additional macro-averaged test metrics. Values are mean +/- standard deviation over five seeds.",
        ["Model", "Task", "Macro-P", "Macro-R", "Weighted F1"],
        [
            ["TF-IDF + logistic regression", "Sentiment", "0.6919", "0.7688", "0.8446"],
            ["Single-task XLM-R", "Sentiment", "0.7740 +/- 0.0028", "0.7737 +/- 0.0173", "0.8964 +/- 0.0035"],
            ["Multitask XLM-R", "Sentiment", "0.7687 +/- 0.0104", "0.7762 +/- 0.0180", "0.8962 +/- 0.0043"],
            ["TF-IDF + logistic regression", "Provenance", "0.9034", "0.9000", "0.8998"],
            ["Single-task XLM-R", "Provenance", "0.9577 +/- 0.0083", "0.9571 +/- 0.0087", "0.9571 +/- 0.0088"],
            ["Multitask XLM-R", "Provenance", "0.9531 +/- 0.0090", "0.9526 +/- 0.0094", "0.9526 +/- 0.0094"],
        ],
    )
    add_manual_table(
        doc,
        "Table 5 – Per-class test metrics for the multitask model. Values are mean +/- standard deviation over five seeds.",
        ["Task", "Class", "Precision", "Recall", "F1"],
        [
            ["Sentiment", "Negative", "0.8700 +/- 0.0156", "0.8566 +/- 0.0192", "0.8630 +/- 0.0083"],
            ["Sentiment", "Neutral", "0.4887 +/- 0.0415", "0.5284 +/- 0.0663", "0.5041 +/- 0.0205"],
            ["Sentiment", "Positive", "0.9475 +/- 0.0063", "0.9437 +/- 0.0047", "0.9456 +/- 0.0023"],
            ["Provenance", "Authentic", "0.9654 +/- 0.0073", "0.9389 +/- 0.0208", "0.9518 +/- 0.0100"],
            ["Provenance", "AI-generated", "0.9408 +/- 0.0190", "0.9663 +/- 0.0077", "0.9533 +/- 0.0088"],
        ],
    )
    add_plain_paragraph(
        doc,
        "The additional metrics expose an important asymmetry that Macro-F1 alone hides: the neutral sentiment class is substantially weaker (F1 0.5041 +/- 0.0205) than negative and positive sentiment, while the two provenance classes remain balanced. The 95 percent t-based confidence intervals across seeds are [0.7600, 0.7817] for multitask sentiment Macro-F1 and [0.9409, 0.9642] for multitask provenance Macro-F1. Paired Cohen's dz values are -0.1383 and -0.2724 for multitask versus single-task sentiment and provenance, respectively, and 6.2744 and 5.6106 versus the classical baseline; because only five seeds are available, these effect sizes are descriptive. AUROC, PR-AUC, calibration error, and Brier score are not reported because the release artifacts contain aggregate confusion-matrix metrics but no per-example probability outputs; these metrics should be added in the next evaluation run.",
        size=11.5,
        indent=0.5,
        after=2,
    )
    add_plain_paragraph(
        doc,
        "The majority baselines in Table 2 are 0.2737 Macro-F1 and 0.6965 accuracy for sentiment, and 0.3333 Macro-F1 and 0.5000 accuracy for provenance. For the four Macro-F1 comparisons in Table 3, Holm-adjusted p-values are 0.0020 for both multitask-versus-baseline comparisons, 0.3658 for provenance versus single-task, and 0.8016 for sentiment versus single-task. These inferences remain conditional on the fixed test split and are exploratory because only five training seeds are available.",
        size=11.5,
        indent=0.5,
        after=2,
    )
    add_plain_paragraph(
        doc,
        "A robustness audit across language, source, and domain slices also limits the generalization claim. Across the five seeds, the worst-language Macro-F1 averaged 0.5688 +/- 0.0220 for sentiment and 0.8905 +/- 0.0267 for provenance; the worst-source and worst-domain sentiment Macro-F1 values averaged 0.6210 +/- 0.0039 and 0.6223 +/- 0.0030. Several slices contain no neutral sentiment examples, so these slice scores are diagnostic and should not be interpreted as fully comparable three-class estimates.",
        size=11.5,
        indent=0.5,
        after=2,
    )


def add_reference(doc, text):
    p = doc.add_paragraph()
    set_para(p, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0, left=0.30, hanging=0.30, after=2, line=1.0)
    run = p.add_run(text)
    set_run_font(run, size=9.5)


def build():
    if not REFERENCE.exists():
        raise FileNotFoundError(REFERENCE)
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REFERENCE, OUTPUT)
    doc = Document(OUTPUT)
    configure_document(doc)
    clear_body(doc)
    clear_header_footer(doc.sections[0])

    doc.core_properties.title = "Development of a Multitask Transformer-Based Model and Web System for Joint Sentiment Analysis and AI-Generated Review Provenance Classification: A Controlled Mixed-Domain Study"
    doc.core_properties.subject = "Joint sentiment analysis and review provenance classification"
    doc.core_properties.author = "Dias Kazikhanov"
    doc.core_properties.keywords = "multitask learning, XLM-RoBERTa, sentiment analysis, review authenticity"

    add_plain_paragraph(doc, "UDC 004.021", alignment=WD_ALIGN_PARAGRAPH.LEFT, indent=0, size=11.5, after=3)
    add_author_line(doc, "Dias Kazikhanov, ", "Master’s student, School of Software Engineering, Astana IT University, Astana, Kazakhstan")
    add_author_line(doc, "Aigul Meirmanova, ", "PhD, Postdoctoral Researcher, Assistant Professor, School of Software Engineering, Astana IT University, Astana, Kazakhstan")
    add_plain_paragraph(
        doc,
        "DEVELOPMENT OF A MULTITASK TRANSFORMER-BASED MODEL AND WEB SYSTEM FOR JOINT SENTIMENT ANALYSIS AND AI-GENERATED REVIEW PROVENANCE CLASSIFICATION: A CONTROLLED MIXED-DOMAIN STUDY",
        alignment=WD_ALIGN_PARAGRAPH.CENTER,
        indent=0,
        size=12.5,
        bold=True,
        before=8,
        after=8,
        keep=True,
    )

    source = Document(SOURCE)
    started = False
    abstract_done = False
    table_counter = 0
    for kind, item, text in iter_source_blocks(source):
        if kind == "tbl":
            if started:
                table_counter += 1
                add_table_from_source(doc, item, index=table_counter)
                if table_counter == 3:
                    add_review_metrics(doc)
            continue

        clean = text.strip()
        if not started:
            if clean == "Abstract":
                add_plain_paragraph(doc, "Abstract", alignment=WD_ALIGN_PARAGRAPH.CENTER, indent=0, size=12, bold=True, italic=True, before=2, after=3, keep=True)
                started = True
            continue

        if not abstract_done:
            if clean.startswith("Keywords:"):
                p = doc.add_paragraph()
                set_para(p, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, indent=0.5, after=5)
                label, rest = clean.split(":", 1)
                rest = rest.replace("review authenticity", "AI-generated review provenance")
                r = p.add_run(label + ":")
                set_run_font(r, size=11.5, bold=True, italic=True)
                r = p.add_run(rest)
                set_run_font(r, size=11.5, italic=True)
                abstract_done = True
            elif clean:
                add_plain_paragraph(doc, clean, italic=True, size=11.5, indent=0.5, after=3)
            continue

        if not clean:
            continue

        # Correct two citation-index slips in the source draft while keeping
        # the underlying argument unchanged: Caruana is reference [1],
        # XLM-RoBERTa is [2], and MAiDE-up is [3].
        clean = clean.replace("tasks are related [3]", "tasks are related [1]")
        clean = clean.replace("pretrained across 100 languages [1, 2]", "pretrained across 100 languages [2]")
        clean = clean.replace("coordinated manipulation [4]", "coordinated manipulation [3]")
        clean = clean.replace("returns sentiment and authenticity labels with confidence scores", "returns sentiment and provenance labels with confidence scores")
        clean = clean.replace(
            "The source roles and label semantics are shown in Table 1.",
            "The source roles and label semantics are shown in Table 1 [3,4,6,7].",
        )
        clean = clean.replace(
            "These intervals describe uncertainty within the chosen split; they do not quantify uncertainty from new sources, domains, or languages.",
            "These intervals describe uncertainty within the chosen split; they do not quantify uncertainty from new sources, domains, or languages. We also computed worst-slice Macro-F1 by language, source, and domain with a minimum slice support of 25.",
        )
        clean = clean.replace(
            "No external funding or conflict-of-interest information was supplied for this manuscript. Replace this statement with the official declaration required by the target venue before submission.",
            "No external funding or conflict-of-interest declaration was available in the project materials at the time of writing; the authors should confirm the official statement required by the target venue before submission.",
        )
        clean = clean.replace(
            "The project repository contains the data-normalization code, training CLI, configuration snapshot, exported model format, API service, and aggregated reports. The principal reproducibility artifacts include data/processed/joint_reviews.article30k.jsonl, configs/model.article30k.mac_m2_16gb.yaml, the multiseed report and statistics directories, and the split/leakage audit. The corpus digest is given in Section 4. Redistribution of source records follows the original dataset terms; exact reproduction should use the cited dataset cards and recorded configuration rather than silently substituting newer versions.",
            "The project repository contains the data-normalization code, training CLI, configuration snapshot, exported model format, API service, and aggregated reports. The principal reproducibility artifacts include data/processed/joint_reviews.article30k.jsonl, configs/model.article30k.mac_m2_16gb.yaml, the multiseed report and statistics directories, and the split/leakage audit. The complete-run manifest records commit 64cf0a580d4a28434454aba763bec51ff052c689, Python 3.12.12, macOS 26.2 arm64, the MPS backend, split seed 42, and training seeds 11, 21, 42, 84, and 126. The corpus digest is given in Section 4. Redistribution of source records follows the original dataset terms; exact reproduction should use the cited dataset cards and recorded configuration rather than silently substituting newer versions. The repository was dirty during the run, so the modified paths must be preserved with any supplementary artifact.",
        )
        clean = clean.replace(
            "No participants were recruited and no new personal data were collected. The analysis uses textual records obtained from open research sources and does not reproduce raw reviews, author names, or direct user identifiers. Provenance predictions should be treated as research signals and should not be used as standalone evidence of author misconduct.",
            "No participants were recruited and no new personal data were collected. The study uses textual records from open research sources. The release package does not reproduce raw review text or direct identifiers; the locally processed corpus may retain source-specific identifiers for audit purposes and must not be redistributed without checking the original licenses and applying an explicit privacy/PII review. Provenance predictions are research signals, not standalone evidence of author misconduct.",
        )
        clean = clean.replace(
            "Deployment validation: probabilities were not calibrated, a moderation threshold was not selected, and the web system is decision support rather than an autonomous enforcement service.",
            "Deployment validation: probabilities were not calibrated, a moderation threshold was not selected, and the web system is decision support rather than an autonomous enforcement service.",
        )
        clean = clean.replace(
            "One fixed split: five training seeds measure initialization sensitivity but do not replace source-held-out, language-held-out, or external evaluation.",
            "One fixed split: five training seeds measure initialization sensitivity but do not replace source-held-out, language-held-out, or external evaluation. Before redistribution, source-specific provenance subtype metadata must also be audited against authenticity_label; the reported metrics use the authenticity_label field directly.",
        )
        clean = clean.replace("Caruana, R..", "Caruana, R.")
        clean = clean.replace("Smetanin, S..", "Smetanin, S.")
        clean = clean.replace("et al..", "et al.")

        # Captions remain manual paragraphs in the reference template.
        if re.match(r"^(Table|Figure)\s+\d+\s*[.:]", clean, re.I):
            add_caption(doc, clean)
            continue

        # Preserve the source article's section hierarchy while adopting the
        # template's manual bold/italic heading treatment.
        if item.style.name in {"Heading 1", "Heading 2"} or re.match(r"^\d+(?:\.\d+)?\s+", clean):
            add_section_heading(doc, clean, sub=(item.style.name == "Heading 2" or bool(re.match(r"^\d+\.\d+\s+", clean))))
            continue

        if clean.startswith("[") and re.match(r"^\[\d+\]", clean):
            clean = clean.replace("..", ".")
            add_reference(doc, clean)
            continue

        if item.style.name.startswith("List") or clean.startswith("RQ1.") or clean.startswith("RQ2.") or clean.startswith("RQ3.") or clean.startswith("Provenance validity:"):
            add_bullet(doc, clean)
            continue

        add_plain_paragraph(doc, clean, size=11.5, indent=0.5, after=2)

    add_section_heading(doc, "Information about authors")
    add_plain_paragraph(
        doc,
        "Dias Kazikhanov – Master’s student, School of Software Engineering, Astana IT University, Astana, Kazakhstan.",
        italic=True,
        size=11,
        indent=0.5,
    )
    add_plain_paragraph(
        doc,
        "Aigul Meirmanova – PhD, Postdoctoral Researcher, Assistant Professor, School of Software Engineering, Astana IT University, Astana, Kazakhstan.",
        italic=True,
        size=11,
        indent=0.5,
    )

    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build()
