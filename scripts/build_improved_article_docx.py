from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


OUT = Path("docs/article_improved_en.docx")


def set_run_font(run, name="Arial", size=10.5, bold=False, italic=False, color="000000"):
    run.font.name = name
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), name)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_borders(cell, color="D9D9D9", size="6"):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = "w:" + edge
        element = borders.find(qn(tag))
        if element is None:
            element = OxmlElement(tag)
            borders.append(element)
        element.set(qn("w:val"), "single")
        element.set(qn("w:sz"), size)
        element.set(qn("w:space"), "0")
        element.set(qn("w:color"), color)


def set_cell_margins(cell, top=90, start=100, bottom=90, end=100):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    margins = tc_pr.first_child_found_in("w:tcMar")
    if margins is None:
        margins = OxmlElement("w:tcMar")
        tc_pr.append(margins)
    for m, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = margins.find(qn("w:" + m))
        if node is None:
            node = OxmlElement("w:" + m)
            margins.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def set_keep_with_next(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    keep = OxmlElement("w:keepNext")
    p_pr.append(keep)


def remove_paragraph_borders(paragraph):
    p_pr = paragraph._p.get_or_add_pPr()
    p_bdr = p_pr.find(qn("w:pBdr"))
    if p_bdr is None:
        p_bdr = OxmlElement("w:pBdr")
        p_pr.append(p_bdr)
    for edge in ("top", "left", "bottom", "right", "between"):
        node = p_bdr.find(qn("w:" + edge))
        if node is None:
            node = OxmlElement("w:" + edge)
            p_bdr.append(node)
        node.set(qn("w:val"), "nil")


def set_row_cant_split(row):
    tr_pr = row._tr.get_or_add_trPr()
    cant_split = OxmlElement("w:cantSplit")
    tr_pr.append(cant_split)


def add_page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run()
    set_run_font(run, size=9, color="666666")
    fld_char1 = OxmlElement("w:fldChar")
    fld_char1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_char2 = OxmlElement("w:fldChar")
    fld_char2.set(qn("w:fldCharType"), "end")
    run._r.append(fld_char1)
    run._r.append(instr)
    run._r.append(fld_char2)


def add_hyperlink(paragraph, text, url):
    part = paragraph.part
    r_id = part.relate_to(url, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink", is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    r_pr = OxmlElement("w:rPr")
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "1F4E79")
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    r_pr.append(color)
    r_pr.append(underline)
    new_run.append(r_pr)
    text_node = OxmlElement("w:t")
    text_node.text = text
    new_run.append(text_node)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def add_body(doc, text, style="Normal"):
    p = doc.add_paragraph(style=style)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.line_spacing = 1.08
    run = p.add_run(text)
    set_run_font(run)
    return p


def add_bullet(doc, text, number=False):
    p = doc.add_paragraph(style="List Number" if number else "List Bullet")
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.05
    run = p.add_run(text)
    set_run_font(run)
    return p


def add_heading(doc, text, level=1):
    p = doc.add_paragraph(style=f"Heading {level}")
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, size=13 if level == 1 else 11, bold=True)
    return p


def add_caption(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(7)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    set_run_font(run, size=9, italic=True, color="444444")
    return p


def add_table(doc, headers, rows, widths):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    for idx, header in enumerate(headers):
        cell = table.rows[0].cells[idx]
        cell.width = Inches(widths[idx])
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_shading(cell, "1F4E79")
        set_cell_borders(cell)
        set_cell_margins(cell)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        run = p.add_run(header)
        set_run_font(run, size=9, bold=True, color="FFFFFF")
    set_repeat_table_header(table.rows[0])
    for row_i, row_data in enumerate(rows):
        row = table.add_row()
        set_row_cant_split(row)
        for idx, value in enumerate(row_data):
            cell = row.cells[idx]
            cell.width = Inches(widths[idx])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_borders(cell)
            set_cell_margins(cell)
            if row_i % 2 == 1:
                set_cell_shading(cell, "F2F6FA")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.0
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if idx == 1 or (len(row_data) == 4 and idx in (2, 3)) else WD_ALIGN_PARAGRAPH.LEFT
            run = p.add_run(str(value))
            set_run_font(run, size=8.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(1)
    return table


def add_reference(doc, number, author, title, venue, year, url):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.0
    r = p.add_run(f"[{number}] {author}. {title}. {venue}, {year}. ")
    set_run_font(r, size=9)
    add_hyperlink(p, url, url)


def configure_document(doc):
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.72)
    section.bottom_margin = Inches(0.68)
    section.left_margin = Inches(0.82)
    section.right_margin = Inches(0.82)
    section.header_distance = Inches(0.3)
    section.footer_distance = Inches(0.3)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Arial"
    normal._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    normal._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08
    for style_name, size in (("Heading 1", 13), ("Heading 2", 11)):
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(10 if style_name == "Heading 1" else 7)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True
    title_style = styles["Title"]
    title_style.font.name = "Arial"
    title_style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
    title_style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
    title_style.font.size = Pt(17)
    title_style.font.bold = True
    title_style.font.color.rgb = RGBColor(0, 0, 0)
    title_style.paragraph_format.space_after = Pt(5)
    title_style_ppr = title_style._element.get_or_add_pPr()
    title_style_bdr = title_style_ppr.find(qn("w:pBdr"))
    if title_style_bdr is None:
        title_style_bdr = OxmlElement("w:pBdr")
        title_style_ppr.append(title_style_bdr)
    for edge in ("top", "left", "bottom", "right", "between"):
        node = title_style_bdr.find(qn("w:" + edge))
        if node is None:
            node = OxmlElement("w:" + edge)
            title_style_bdr.append(node)
        node.set(qn("w:val"), "nil")
    for style_name in ("List Bullet", "List Number"):
        style = styles[style_name]
        style.font.name = "Arial"
        style._element.rPr.rFonts.set(qn("w:ascii"), "Arial")
        style._element.rPr.rFonts.set(qn("w:hAnsi"), "Arial")
        style.font.size = Pt(10.2)

    header = section.header.paragraphs[0]
    header.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    header_run = header.add_run("ReviewGuard | Dissertation article")
    set_run_font(header_run, size=8, color="666666")
    footer = section.footer.paragraphs[0]
    add_page_number(footer)


def build():
    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(5)
    remove_paragraph_borders(title)
    title_run = title.add_run("Development of a Multitask Transformer-Based Model and Web System for Joint Sentiment Analysis and Review Authenticity Detection in E-Commerce")
    set_run_font(title_run, size=17, bold=True)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.paragraph_format.space_after = Pt(8)
    subtitle_run = subtitle.add_run("A reproducible evaluation of a shared XLM-RoBERTa encoder on a partially labelled corpus of 30,000 reviews")
    set_run_font(subtitle_run, size=11, italic=True, color="444444")

    author = doc.add_paragraph()
    author.alignment = WD_ALIGN_PARAGRAPH.CENTER
    author.paragraph_format.space_after = Pt(1)
    r = author.add_run("Dias Kazikhanov")
    set_run_font(r, size=11.5, bold=True)
    aff = doc.add_paragraph()
    aff.alignment = WD_ALIGN_PARAGRAPH.CENTER
    aff.paragraph_format.space_after = Pt(1)
    r = aff.add_run("Astana IT University, Master of Computer Science and Engineering")
    set_run_font(r, size=10)
    date = doc.add_paragraph()
    date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date.paragraph_format.space_after = Pt(12)
    r = date.add_run("September 2026")
    set_run_font(r, size=9.5, color="555555")

    add_heading(doc, "Abstract", 1)
    add_body(doc, "Online reviews combine an evaluative signal with information about how a text was produced. This article presents the completed experimental stage of a dissertation project that develops a multitask Transformer-based model and the ReviewGuard web system for joint review analysis. The model predicts three-way sentiment and a provenance label that is operationalized as human-authored versus AI-generated text in the MAiDE-up resource; this definition is narrower than universal fake-review detection. A corpus of 30,000 records was assembled from four open datasets. TF-IDF with logistic regression, two single-task XLM-RoBERTa models, and one shared-encoder multitask model were compared under one leakage-controlled split and five training seeds. The multitask model reached test Macro-F1 of 0.7709 +/- 0.0087 for sentiment and 0.9526 +/- 0.0094 for provenance, compared with 0.7161 and 0.8998 for the classical baseline. Differences from the single-task Transformer were inconclusive: -0.0013 for sentiment and -0.0046 for provenance. The result supports a compact shared service, but not a claim of positive transfer or universal authenticity detection. The web layer exposes predictions, confidence, truncation status, and research-context warnings for human-in-the-loop use.")
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(11)
    r = p.add_run("Keywords: ")
    set_run_font(r, size=10, bold=True)
    r = p.add_run("sentiment analysis; review authenticity; provenance classification; multitask learning; XLM-RoBERTa; e-commerce; reproducibility; web system.")
    set_run_font(r, size=10)

    add_heading(doc, "1 Introduction", 1)
    add_body(doc, "Online reviews influence product ranking, merchant reputation, customer trust, and purchase decisions. An e-commerce analytics service therefore needs to answer two different questions: what evaluation a reviewer expresses, and whether the text belongs to the provenance category represented in the training data. These questions are related because they use the same review text, but they are not interchangeable. A negative review is not automatically deceptive, and a classifier score is not independent evidence of misconduct.")
    add_body(doc, "The dissertation topic is to develop a multitask Transformer-based model and web system for joint sentiment analysis and review authenticity detection in e-commerce. In the executed study, authenticity is defined narrowly and reproducibly through the binary MAiDE-up label: human-authored hotel review versus AI-generated hotel review. The paper therefore uses provenance classification when describing the empirical target and reserves broader authenticity language for the application goal.")
    add_body(doc, "Multi-task learning provides a direct way to test whether one encoder can support both tasks through shared representations. XLM-RoBERTa is appropriate for the multilingual corpus because it was pretrained across 100 languages [1, 2]. However, shared parameters do not guarantee positive transfer. The comparison with single-task models is essential, and the training protocol must be documented well enough to separate an architectural effect from differences in sampling or optimization.")
    add_body(doc, "The contribution is fourfold. First, the study reports a completed 30,000-record, five-seed evaluation. Second, it makes label provenance and the split audit explicit. Third, it separates the scientific comparison from the supporting web-system implementation. Fourth, it constrains the interpretation to the tested domain, label definition, and fixed evaluation protocol.")

    add_heading(doc, "2 Related Work", 1)
    add_body(doc, "The multi-task learning formulation treats a shared representation as an inductive bias that can improve generalization when tasks are related [3]. In modern NLP, pretrained Transformer encoders provide the shared representation, while task-specific heads allow different label spaces. The benefit is empirical rather than automatic: task imbalance, label sparsity, and gradient conflict can reduce or reverse transfer.")
    add_body(doc, "XLM-RoBERTa extends this setting to multilingual text through cross-lingual pretraining [2]. Its use in this study is motivated by the corpus composition, not by an assumption that all language slices are equally represented. The Russian portion dominates the merged corpus, while the provenance labels come from a multilingual hospitality dataset. This makes source- and language-aware evaluation important.")
    add_body(doc, "Review authenticity has several non-equivalent meanings. Classical deceptive-opinion studies distinguish truthful and deceptive writing, whereas newer datasets focus on AI-generated reviews. MAiDE-up is valuable for the latter setting, but its human-versus-AI labels should not be generalized to all spam, incentive abuse, account takeover, or coordinated manipulation [4]. The present study adopts that narrower semantics.")
    add_body(doc, "Finally, a fixed test set and repeated seeds require careful interpretation. Statistical testing guidance for NLP recommends matching the test to the experimental unit and reporting uncertainty rather than treating one p-value as a complete argument [5]. This article therefore reports seed variation, paired confidence intervals, and approximate randomization comparisons.")

    add_heading(doc, "3 Research Questions and Scope", 1)
    add_body(doc, "The study investigates three questions:")
    add_bullet(doc, "RQ1. Does multitask XLM-RoBERTa improve Macro-F1 over TF-IDF with logistic regression on sentiment or provenance under the fixed test protocol?")
    add_bullet(doc, "RQ2. Does multitask XLM-RoBERTa differ from single-task XLM-RoBERTa trained for the same task?")
    add_bullet(doc, "RQ3. Which data, label, and training properties limit transfer beyond this protocol and the ReviewGuard demonstration system?")
    add_body(doc, "The principal hypotheses are deliberately modest. H1 predicts that multitask training will not materially underperform single-task training on sentiment. H2 predicts that both Transformer variants will exceed the classical baseline on provenance. H3 treats the web interface as an engineering deliverable, not as evidence of model interpretability or a substitute for external validation.")

    add_heading(doc, "4 Data and Label Provenance", 1)
    add_body(doc, "The processed input contains 30,000 normalized records and is identified by SHA-256 digest a296cf7bb8cb0dbb3eebd8a0cc9c478422e9f5959836bc4d20915f9f81c9c5fc. Four open sources were merged with explicit source identifiers. The source roles and label semantics are shown in Table 1.")
    doc.add_page_break()
    add_caption(doc, "Table 1. Corpus composition and label provenance.")
    add_table(doc, ["Source", "n", "Sentiment", "Provenance"], [
        ("MAiDE-up", "7,000", "Source label; negative/positive", "3,500 human-authored and 3,500 AI-generated"),
        ("Perekrestok", "10,000", "Rating mapping: 1-2 negative, 3 neutral, 4-5 positive", "No label"),
        ("RuReviews", "3,000", "Native three-class dataset label", "No label"),
        ("Wildberries", "10,000", "Rating mapping: 1-2 negative, 3 neutral, 4-5 positive", "No label"),
    ], [1.1, 0.55, 2.35, 2.85])
    add_body(doc, "All records have a sentiment label. There are 23,734 Russian records and 6,266 records in Turkish, French, Korean, English, Chinese, Spanish, Italian, German, or Romanian. The domain mixture is 23,000 e-commerce records and 7,000 hospitality records. Provenance labels are available only for the 7,000 MAiDE-up records, so e-commerce rows contribute to sentiment loss but not to provenance loss.")
    add_body(doc, "The rating-derived labels are a practical volume compromise, not independent manual annotation. The sentiment task therefore measures agreement with the project's three-class rating mapping. It should not be interpreted as a universal linguistic definition of polarity. Similarly, the provenance task measures the MAiDE-up human-versus-AI distinction and should not be reported as a detector for every kind of fake review.")

    add_heading(doc, "5 Methodology", 1)
    add_heading(doc, "5.1 Leakage-Controlled Split", 2)
    add_body(doc, "Using split seed 42, the corpus was divided into 23,975 training, 3,013 validation, and 3,012 test records. Texts were case-folded and whitespace-normalized before splitting, then grouped by normalized text and stratified by source and available-label pair. The audit found 521 exact and 564 normalized duplicate rows in the merged corpus, but zero cross-partition overlap by record identifier, exact text, or normalized text for every train, validation, and test pair. The test sentiment distribution is 710 negative, 204 neutral, and 2,098 positive; the provenance test is balanced at 350 authentic and 350 AI-generated examples.")

    add_heading(doc, "5.2 Model Architecture", 2)
    add_body(doc, "The classical reference uses TF-IDF unigrams and bigrams with up to 50,000 features and a class-balanced logistic-regression classifier for each task. The Transformer systems use FacebookAI/xlm-roberta-base. The shared encoder produces a contextual representation from the first token. One MLP head predicts three sentiment classes and a second predicts two provenance classes; each head uses a GELU hidden layer and dropout of 0.1. For a record with both labels, the loss is the weighted sum of the two cross-entropies. When one label is absent, only the available task contributes to the loss. Single-task models use the same encoder and head design but optimize one label only.")

    add_heading(doc, "5.3 Training Protocol", 2)
    add_body(doc, "The local Apple Silicon configuration uses maximum sequence length 192, batch size 2, four-step gradient accumulation, gradient checkpointing, AdamW with learning rate 2e-5, weight decay 0.01, 10 percent linear warmup, a five-epoch maximum, and early stopping with patience 2. Multi-task training uses a source-balanced sampler and class-weighted cross-entropy; the single-task loop does not use the source-balanced sampler. Consequently, the multitask-versus-single-task comparison evaluates two configured training systems rather than one isolated architectural switch.")
    add_body(doc, "Training was run on a Mac mini M2 Pro with 16 GB unified memory through the MPS backend. The recorded environment was Python 3.12.12 on macOS 26.2 arm64. Transformer systems used seeds 11, 21, 42, 84, and 126. Checkpoints were selected using validation Macro-F1 and the test set was reserved for final evaluation. Macro-F1 is the primary metric because class balance differs across tasks; accuracy is reported as a secondary metric.")

    add_heading(doc, "5.4 ReviewGuard Web System", 2)
    add_body(doc, "The model is embedded in ReviewGuard, a reproducibility-oriented service with five stages: data normalization, training and checkpoint export, checkpoint loading, FastAPI inference, and a browser interface. The /health endpoint reports service and model readiness; /research-context returns the training scope and dataset warnings; and POST /analyze accepts a review and returns sentiment and authenticity labels with confidence scores.")
    add_body(doc, "The response also contains a lightweight transparency object: ranked task probabilities, token count, configured maximum length, a truncation flag, prediction margins, risk flags, and research-context warnings. This is an operational aid for a human reviewer, not a causal explanation of the model and not evidence that the predicted provenance label is factually true.")

    add_heading(doc, "6 Experimental Design", 1)
    add_body(doc, "Each Transformer variant was trained for five random seeds on the same fixed split. The classical baseline is deterministic under the fixed configuration. For Transformer systems, the article reports mean and standard deviation over seeds and a 95 percent t-interval for the mean. Paired comparisons use the same test examples: 2,000 bootstrap samples produce an interval for the seed-averaged difference, and an approximate randomization test uses 2,000 permutations. These intervals describe uncertainty within the chosen split; they do not quantify uncertainty from new sources, domains, or languages.")
    add_body(doc, "The comparison is intentionally bounded. It tests a shared encoder against two single-task encoders and a shallow text baseline under one data construction. It does not claim that the chosen source quotas represent the population of e-commerce reviews, that MAiDE-up labels cover human deception, or that a production threshold has been calibrated.")

    add_heading(doc, "7 Results", 1)
    add_heading(doc, "7.1 Main Test Metrics", 2)
    add_caption(doc, "Table 2. Test metrics. Transformer values are mean +/- standard deviation over five seeds.")
    add_table(doc, ["Model family", "Task", "Macro-F1", "Accuracy"], [
        ("TF-IDF + logistic regression", "Sentiment", "0.7161", "0.8320"),
        ("Single-task XLM-R", "Sentiment", "0.7721 +/- 0.0087", "0.8958 +/- 0.0038"),
        ("Multitask XLM-R", "Sentiment", "0.7709 +/- 0.0087", "0.8950 +/- 0.0047"),
        ("TF-IDF + logistic regression", "Provenance", "0.8998", "0.9000"),
        ("Single-task XLM-R", "Provenance", "0.9571 +/- 0.0088", "0.9571 +/- 0.0087"),
        ("Multitask XLM-R", "Provenance", "0.9526 +/- 0.0094", "0.9526 +/- 0.0094"),
    ], [2.5, 1.25, 1.65, 1.45])
    add_body(doc, "Multitask XLM-R exceeds the TF-IDF baseline by 0.0548 Macro-F1 for sentiment and 0.0528 for provenance. The single-task Transformer has slightly higher means on both tasks, but the gaps are small relative to seed variation. The main practical result is therefore a compact shared model with comparable average quality, not a demonstrated accuracy advantage over separate models.")

    add_heading(doc, "7.2 Paired Comparisons", 2)
    add_caption(doc, "Table 3. Paired Macro-F1 comparisons for the multitask model. Delta equals multitask minus comparator.")
    add_table(doc, ["Task", "Comparator", "Delta", "95% CI", "p (approx. randomization)"], [
        ("Sentiment", "Single-task XLM-R", "-0.0013", "[-0.0105, 0.0081]", "0.8016"),
        ("Sentiment", "TF-IDF + logistic regression", "+0.0548", "[0.0337, 0.0732]", "0.0005"),
        ("Provenance", "Single-task XLM-R", "-0.0046", "[-0.0109, 0.0020]", "0.1829"),
        ("Provenance", "TF-IDF + logistic regression", "+0.0528", "[0.0305, 0.0749]", "0.0005"),
    ], [1.0, 2.35, 0.85, 1.35, 1.3])
    add_body(doc, "The intervals for the multitask-versus-single-task comparisons cross zero, so the experiment does not establish positive transfer. Against the classical baseline, the intervals are positive and the randomization p-value is 0.0005 for both tasks, the smallest value resolvable with 2,000 permutations. The answer to RQ1 is positive within the fixed protocol; the answer to RQ2 is that no reliable difference is established.")

    add_heading(doc, "8 Discussion", 1)
    add_body(doc, "The result pattern is consistent across tasks: Transformer representations are clearly stronger than the selected TF-IDF reference, while the distinction between shared and single-task Transformer training is small. This may indicate that the two targets use partially shared lexical and semantic information without providing enough independent provenance supervision to produce measurable positive transfer. It may also reflect the non-isolated sampler difference. The current evidence cannot distinguish these explanations.")
    add_body(doc, "For the dissertation system, the compact shared encoder remains useful. It provides one exportable checkpoint, one API contract, and one web interface for two predictions. That engineering advantage should not be converted into a scientific claim that the shared model is more accurate. A deployment should expose confidence and truncation information, require human review for high-impact decisions, and retain the source and label warnings returned by the research context.")
    add_body(doc, "The web system also clarifies the boundary between a model and a moderation policy. A provenance score is a research signal tied to MAiDE-up's human-versus-AI definition. It is not a decision that a reviewer is fraudulent, and it should not be used for automatic penalties without independent validation, calibration, and an appeal process.")

    add_heading(doc, "9 Limitations and Next Steps", 1)
    add_bullet(doc, "Provenance validity: all provenance labels come from MAiDE-up and the hotel domain. They contrast human-authored and AI-generated reviews and do not test human-written deception, incentive abuse, or coordinated spam.")
    add_bullet(doc, "Heterogeneous sentiment supervision: Perekrestok and Wildberries labels are derived from ratings, so source and label-generation effects may be confounded with language and domain effects.")
    add_bullet(doc, "One fixed split: five training seeds measure initialization sensitivity but do not replace source-held-out, language-held-out, or external evaluation.")
    add_bullet(doc, "Duplicate structure and quota sampling: the audit prevents observable cross-partition text leakage, but normalized duplicates remain inside the merged corpus and fixed source quotas are not a random sample of all reviews.")
    add_bullet(doc, "Non-isolated ablation: the multitask system uses source-balanced sampling while the single-task system does not. A stricter architectural ablation must equalize sampling and optimization schedules.")
    add_bullet(doc, "Deployment validation: probabilities were not calibrated, a moderation threshold was not selected, and the web system is decision support rather than an autonomous enforcement service.")
    add_body(doc, "The next dissertation stage should collect or obtain independent provenance labels in an e-commerce domain, repeat the comparison with identical samplers, report source- and language-held-out results, calibrate probabilities, and evaluate the complete API workflow with representative user scenarios. These steps would test generalization rather than only extending the training corpus.")

    add_heading(doc, "10 Conclusion", 1)
    add_body(doc, "This article presents a reproducible multitask XLM-RoBERTa model and ReviewGuard web system for joint sentiment analysis and review provenance classification. On a leakage-controlled, partially labelled corpus of 30,000 records, multitask training increased Macro-F1 over TF-IDF with logistic regression from 0.7161 to 0.7709 for sentiment and from 0.8998 to 0.9526 for provenance. Single-task Transformer systems had slightly higher means, and paired intervals did not confirm a difference.")
    add_body(doc, "The defensible conclusion is therefore conditional. A shared encoder can support both predictions in one service without a clear average-quality loss under this protocol, but the experiment does not prove positive transfer and does not establish universal fake-review detection. The strongest next test is an external, source-held-out and domain-held-out evaluation with independently defined provenance labels.")

    add_heading(doc, "Data Availability and Reproducibility", 1)
    add_body(doc, "The project repository contains the data-normalization code, training CLI, configuration snapshot, exported model format, API service, and aggregated reports. The principal reproducibility artifacts include data/processed/joint_reviews.article30k.jsonl, configs/model.article30k.mac_m2_16gb.yaml, the multiseed report and statistics directories, and the split/leakage audit. The corpus digest is given in Section 4. Redistribution of source records follows the original dataset terms; exact reproduction should use the cited dataset cards and recorded configuration rather than silently substituting newer versions.")

    add_heading(doc, "Ethics Statement", 1)
    add_body(doc, "No participants were recruited and no new personal data were collected. The analysis uses textual records obtained from open research sources and does not reproduce raw reviews, author names, or direct user identifiers. Provenance predictions should be treated as research signals and should not be used as standalone evidence of author misconduct.")

    add_heading(doc, "Author Contributions", 1)
    add_body(doc, "Dias Kazikhanov: conceptualization, methodology, software, data curation, formal analysis, validation, visualization, writing - original draft, and writing - review and editing. Supervisor: A. Meirmanova.")

    add_heading(doc, "Funding and Conflict of Interest", 1)
    add_body(doc, "No external funding or conflict-of-interest information was supplied for this manuscript. Replace this statement with the official declaration required by the target venue before submission.")

    add_heading(doc, "AI Use Disclosure", 1)
    add_body(doc, "Generative AI was used as an auxiliary tool for manuscript structure and language drafting. Numerical results, experimental configuration, data claims, and references were checked against repository artifacts and the cited primary sources. The author remains responsible for the final manuscript, interpretation, and compliance with the submission venue's policies.")

    add_heading(doc, "References", 1)
    add_reference(doc, 1, "Caruana, R.", "Multitask Learning", "Machine Learning 28(1):41-75", "1997", "https://doi.org/10.1023/A:1007379606734")
    add_reference(doc, 2, "Conneau, A., Khandelwal, K., Goyal, N., et al.", "Unsupervised Cross-lingual Representation Learning at Scale", "Proceedings of ACL, pp. 8440-8451", "2020", "https://aclanthology.org/2020.acl-main.747/")
    add_reference(doc, 3, "Ignat, O., Xu, X., and Mihalcea, R.", "MAiDE-up: Multilingual Deception Detection of AI-generated Hotel Reviews", "Findings of NAACL, pp. 1636-1653", "2025", "https://aclanthology.org/2025.findings-naacl.88/")
    add_reference(doc, 4, "Smetanin, S.", "RuReviews: Russian Reviews Dataset", "GitHub repository", "2026", "https://github.com/sismetanin/rureviews")
    add_reference(doc, 5, "Dror, R., Baumer, G., Shlomov, S., and Reichart, R.", "The Hitchhiker's Guide to Testing Statistical Significance in Natural Language Processing", "Proceedings of ACL, pp. 1383-1392", "2018", "https://aclanthology.org/P18-1128/")
    add_reference(doc, 6, "lapki.", "Perekrestok Reviews", "Hugging Face dataset card", "2026", "https://huggingface.co/datasets/lapki/perekrestok-reviews")
    add_reference(doc, 7, "Hplss.", "WB Review Dataset", "Hugging Face dataset card; CC BY-NC-SA 4.0", "2026", "https://huggingface.co/datasets/Hplss/wb-review-dataset")

    doc.core_properties.title = "Development of a Multitask Transformer-Based Model and Web System for Joint Sentiment Analysis and Review Authenticity Detection in E-Commerce"
    doc.core_properties.author = "Dias Kazikhanov"
    doc.core_properties.subject = "Multitask Transformer review analysis"
    doc.core_properties.comments = "Improved English dissertation article"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
