"""PDF benchmark report meant to be shared with non-technical colleagues: a
narrative summary (the same text the CLI prints), a bar chart comparing
latency/first-token/quality and a pie chart of judge preference, and a
per-question section with both answers - the winning one highlighted green,
the losing one red, both orange on a tie - plus the judge's reasoning and the
raw timing numbers.

Rendered directly (matplotlib for the two charts, reportlab for page layout)
rather than as a spreadsheet, so it looks identical everywhere it's opened -
no dependency on which app's own layout engine renders it.
"""

import io
import re
import statistics
from datetime import datetime
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import HRFlowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

PAGE_SIZE = A4
MARGIN_CM = 2
CONTENT_WIDTH = PAGE_SIZE[0] - 2 * MARGIN_CM * cm

FC_COLOR = "#4472C4"
RAG_COLOR = "#C00000"
TIE_COLOR = "#ED9C28"

WON_BG, WON_FG = colors.HexColor("#C6EFCE"), colors.HexColor("#006100")
LOST_BG, LOST_FG = colors.HexColor("#FFC7CE"), colors.HexColor("#9C0006")
TIE_BG, TIE_FG = colors.HexColor("#FFE4B5"), colors.HexColor("#7F4A00")
NEUTRAL_BG, NEUTRAL_FG = colors.whitesmoke, colors.black

PREFERRED_LABELS = {"full_context": "Full-Context", "rag": "RAG", "tie": "Tie"}


def save_pdf_report(path: str, fc_results, rag_results, judge_results, summary_lines: list[str]) -> None:
    styles = _build_styles()
    story: list = []

    story.append(Paragraph("Full-Context vs. RAG Benchmark Report", styles["ReportTitle"]))
    story.append(Spacer(1, 0.6 * cm))
    story.extend(_summary_story(summary_lines, styles))
    story.append(Spacer(1, 0.4 * cm))

    for chart_buf, width_cm, height_cm in _chart_images(fc_results, rag_results, judge_results):
        story.append(Image(chart_buf, width=width_cm * cm, height=height_cm * cm))
        story.append(Spacer(1, 0.4 * cm))

    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph("Per-Question Results", styles["SectionHeading"]))
    story.append(Spacer(1, 0.2 * cm))

    judge_by_question = {jr.question: jr for jr in judge_results}
    for i, (fc, rag) in enumerate(zip(fc_results, rag_results), start=1):
        jr = judge_by_question.get(fc.question)
        story.extend(_question_block(i, fc, rag, jr, styles))

    doc = SimpleDocTemplate(
        path,
        pagesize=PAGE_SIZE,
        topMargin=MARGIN_CM * cm,
        bottomMargin=MARGIN_CM * cm,
        leftMargin=MARGIN_CM * cm,
        rightMargin=MARGIN_CM * cm,
        title="Full-Context vs. RAG Benchmark Report",
    )
    footer = _make_footer(datetime.now().strftime("%Y-%m-%d %H:%M"))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


CTXVSRAG_URL = "https://pypi.org/project/ctxvsrag/"


def _make_footer(generated_at: str):
    """Small "generated with ctxvsrag" + repo link + timestamp + page number
    footer, drawn on every page - captured once per report (not per page) so
    a multi-page PDF shows one consistent generation time throughout."""

    def _draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        left_text = f"Generated with ctxvsrag ({CTXVSRAG_URL}) — {generated_at}"
        canvas.drawString(MARGIN_CM * cm, 1.2 * cm, left_text)
        text_width = canvas.stringWidth(left_text, "Helvetica", 7)
        canvas.linkURL(CTXVSRAG_URL, (MARGIN_CM * cm, 1.0 * cm, MARGIN_CM * cm + text_width, 1.5 * cm), relative=0)
        canvas.drawRightString(PAGE_SIZE[0] - MARGIN_CM * cm, 1.2 * cm, f"Page {doc.page}")
        canvas.restoreState()

    return _draw


def _build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("ReportTitle", parent=styles["Title"], alignment=TA_CENTER))
    styles.add(ParagraphStyle("SectionHeading", parent=styles["Heading1"]))
    styles.add(ParagraphStyle("SummaryHeading", parent=styles["Heading2"], spaceBefore=6, spaceAfter=2))
    styles.add(ParagraphStyle("SummaryIndent", parent=styles["Normal"], leftIndent=12))
    styles.add(ParagraphStyle("QuestionHeading", parent=styles["Heading3"], spaceBefore=10))
    styles.add(ParagraphStyle("Answer", parent=styles["Normal"], fontSize=9, leading=12))
    styles.add(ParagraphStyle("Reasoning", parent=styles["Normal"], fontSize=9, leading=12, textColor=colors.grey, spaceBefore=4))
    styles.add(ParagraphStyle("Meta", parent=styles["Normal"], fontSize=8, leading=10, textColor=colors.grey, spaceBefore=2))
    return styles


def _escape(text: str) -> str:
    """Answers/reasoning are free-form LLM text and may contain "<", ">" or
    "&" - reportlab's Paragraph parses its input as a small XML dialect, so
    those need escaping before use, same as embedding untrusted text in HTML."""
    return escape(text).replace("\n", "<br/>")


_MD_HEADER_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$", re.MULTILINE)
_MD_BULLET_RE = re.compile(r"^[ \t]*[-*][ \t]+", re.MULTILINE)
_MD_BOLD_RE = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_MD_CODE_RE = re.compile(r"`([^`]+?)`")
_MD_ITALIC_RE = re.compile(r"\*(.+?)\*|_(.+?)_")


def _markdown(text: str) -> str:
    """Model answers are Markdown (LLMs default to it) - rendered literally
    they're cluttered with stray "**"/"#" characters, so this converts the
    common subset (bold, italic, inline code, headers, bullet lists) to the
    small XML dialect reportlab's Paragraph understands. Best-effort, not a
    full Markdown parser: input is escaped first (see _escape), so nothing
    here can inject unintended markup, and unmatched/unusual syntax just
    falls through as plain text rather than erroring."""
    text = escape(text)
    text = _MD_HEADER_RE.sub(lambda m: f"<b>{m.group(1)}</b>", text)
    text = _MD_BULLET_RE.sub("- ", text)
    text = _MD_BOLD_RE.sub(lambda m: f"<b>{m.group(1) or m.group(2)}</b>", text)
    text = _MD_CODE_RE.sub(lambda m: f'<font face="Courier">{m.group(1)}</font>', text)
    text = _MD_ITALIC_RE.sub(lambda m: f"<i>{m.group(1) or m.group(2)}</i>", text)
    return text.replace("\n", "<br/>")


def _summary_story(summary_lines: list[str], styles) -> list:
    story = []
    for line in summary_lines:
        if line == "":
            story.append(Spacer(1, 0.15 * cm))
        elif line == "SUMMARY":
            continue  # redundant next to the report title above it
        elif line.endswith(":") and not line.startswith(" "):
            story.append(Paragraph(_escape(line), styles["SummaryHeading"]))
        elif line.startswith(" "):
            story.append(Paragraph(_escape(line.strip()), styles["SummaryIndent"]))
        else:
            story.append(Paragraph(_escape(line), styles["Normal"]))
    return story


def _chart_images(fc_results, rag_results, judge_results) -> list[tuple[io.BytesIO, float, float]]:
    """Renders the bar and (if a judge ran) pie chart as PNGs and returns
    them with the width/height (cm) to place them at, preserving each
    figure's aspect ratio."""
    images = []

    fc_latencies = [r.result.total_duration_s for r in fc_results]
    rag_latencies = [r.result.total_duration_s + r.retrieval_s for r in rag_results]
    fc_ttfts = [r.result.time_to_first_token_s for r in fc_results if r.result.time_to_first_token_s is not None]
    rag_ttfts = [
        r.result.time_to_first_token_s + r.retrieval_s
        for r in rag_results if r.result.time_to_first_token_s is not None
    ]

    labels = ["Avg. Latency (s)"]
    fc_values = [statistics.mean(fc_latencies)]
    rag_values = [statistics.mean(rag_latencies)]
    if fc_ttfts and rag_ttfts:
        labels.append("Avg. TTFT (s)")
        fc_values.append(statistics.mean(fc_ttfts))
        rag_values.append(statistics.mean(rag_ttfts))
    if judge_results:
        for field, label in (("accuracy", "Accuracy"), ("completeness", "Completeness"), ("clarity", "Clarity")):
            labels.append(f"Avg. {label}\n(1-10)")
            fc_values.append(statistics.mean(j.full_context_scores[field] for j in judge_results))
            rag_values.append(statistics.mean(j.rag_scores[field] for j in judge_results))

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = range(len(labels))
    width = 0.35
    ax.bar([i - width / 2 for i in x], fc_values, width, label="Full-Context", color=FC_COLOR)
    ax.bar([i + width / 2 for i in x], rag_values, width, label="RAG", color=RAG_COLOR)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_title("Full-Context vs. RAG")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    images.append((_figure_to_buf(fig), 17, 17 * 4.2 / 9))

    if judge_results:
        prefs = [j.preferred for j in judge_results]
        counts = [prefs.count("full_context"), prefs.count("rag"), prefs.count("tie")]
        pie_labels = ["Full-Context preferred", "RAG preferred", "Tie"]
        pie_colors = [FC_COLOR, RAG_COLOR, TIE_COLOR]
        nonzero = [(lbl, c, col) for lbl, c, col in zip(pie_labels, counts, pie_colors) if c > 0]

        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.pie(
            [c for _, c, _ in nonzero],
            labels=[lbl for lbl, _, _ in nonzero],
            colors=[col for _, _, col in nonzero],
            autopct=lambda pct: f"{pct:.0f}%",
            startangle=90,
        )
        ax2.set_title("Judge Preference")
        fig2.tight_layout()
        images.append((_figure_to_buf(fig2), 12, 12 * 5 / 6))

    return images


def _figure_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def _status_bar(label: str, status: str, bg, fg, width) -> Table:
    style = ParagraphStyle("statusbar", fontName="Helvetica-Bold", fontSize=10, textColor=fg)
    t = Table([[Paragraph(f"{label} — {status}", style)]], colWidths=[width])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), bg),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _status_for(preferred: str | None, side: str) -> tuple:
    """Fill/text color and label for one side's status bar, from that side's
    own perspective: green if it won, red if it lost, orange on a tie."""
    if preferred is None:
        return NEUTRAL_BG, NEUTRAL_FG, "N/A (judge failed)"
    if preferred == "tie":
        return TIE_BG, TIE_FG, "Tie"
    if preferred == side:
        return WON_BG, WON_FG, "Won"
    return LOST_BG, LOST_FG, "Lost"


def _question_block(i: int, fc, rag, jr, styles) -> list:
    story = [Paragraph(f"Q{i}. {_escape(fc.question)}", styles["QuestionHeading"])]

    preferred = jr.preferred if jr else None
    fc_bg, fc_fg, fc_status = _status_for(preferred, "full_context")
    rag_bg, rag_fg, rag_status = _status_for(preferred, "rag")

    story.append(_status_bar("Full-Context", fc_status, fc_bg, fc_fg, CONTENT_WIDTH))
    story.append(Paragraph(_markdown(fc.answer), styles["Answer"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(_status_bar("RAG", rag_status, rag_bg, rag_fg, CONTENT_WIDTH))
    story.append(Paragraph(_markdown(rag.answer), styles["Answer"]))

    if jr is not None:
        story.append(Paragraph(f"<i>Judge: {_markdown(jr.reasoning)}</i>", styles["Reasoning"]))
        story.append(Paragraph(
            f"Scores (accuracy / completeness / clarity, 1-10) "
            f"— Full-Context: {jr.full_context_scores['accuracy']} / "
            f"{jr.full_context_scores['completeness']} / {jr.full_context_scores['clarity']} "
            f"| RAG: {jr.rag_scores['accuracy']} / {jr.rag_scores['completeness']} / {jr.rag_scores['clarity']}",
            styles["Meta"],
        ))

    rag_total_latency = rag.retrieval_s + rag.result.total_duration_s
    rag_ttft = (
        rag.retrieval_s + rag.result.time_to_first_token_s
        if rag.result.time_to_first_token_s is not None else None
    )
    meta = (
        f"Latency — Full-Context: {fc.result.total_duration_s:.2f}s"
        + (f" (first token {fc.result.time_to_first_token_s:.2f}s)" if fc.result.time_to_first_token_s is not None else "")
        + f" | RAG: {rag_total_latency:.2f}s"
        + (f" (first token {rag_ttft:.2f}s, incl. retrieval)" if rag_ttft is not None else "")
        + f" | Retrieved pages: {', '.join(str(p) for p in rag.retrieved_pages) or '-'}"
    )
    story.append(Paragraph(_escape(meta), styles["Meta"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
    story.append(Spacer(1, 0.3 * cm))
    return story
