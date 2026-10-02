"""Build the submission PDF: uv run --group submission python scripts/build_submission.py."""

import argparse
from html import escape
import json
from pathlib import Path

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer,
    Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/HotcatX/API-test"
NAVY = colors.HexColor("#18334A")
TEAL = colors.HexColor("#176B72")
LIGHT = colors.HexColor("#F1F5F8")
MUTED = colors.HexColor("#526171")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="He Xiao")
    parser.add_argument("--uni", default="hx2425")
    args = parser.parse_args()

    fonts = Path(reportlab.__file__).parent / "fonts"
    for name, file in [("Vera", "Vera.ttf"), ("Vera-Bold", "VeraBd.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily("Vera", normal="Vera", bold="Vera-Bold")
    body = ParagraphStyle("Body", fontName="Vera", fontSize=9.5, leading=14,
                          textColor=NAVY, spaceAfter=7)
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=11, textColor=MUTED)
    title = ParagraphStyle("Title", parent=body, fontName="Vera-Bold", fontSize=24, leading=29, spaceAfter=10)
    heading = ParagraphStyle("Heading", parent=body, fontName="Vera-Bold", fontSize=12,
                             leading=17, textColor=TEAL, spaceBefore=10, spaceAfter=7)
    mono = ParagraphStyle("Mono", fontName="Courier", fontSize=9, leading=13,
                          textColor=NAVY, alignment=TA_LEFT)
    cell = ParagraphStyle("Cell", parent=body, fontSize=8.5, leading=12, spaceAfter=0)
    story = []

    def p(text, style=body):
        story.append(Paragraph(text, style))

    def h(text):
        p(text, heading)

    def equations(text):
        box = Table([[Preformatted(text, mono)]], colWidths=[516])
        box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ]))
        story.extend([box, Spacer(1, 8)])

    def table(rows, widths):
        data = [[Paragraph(str(value), cell) for value in row] for row in rows]
        result = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E4EEF2")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#C4D5DF")),
        ]))
        story.extend([result, Spacer(1, 8)])

    evidence = json.loads((ROOT / "assignment/verification.json").read_text())
    example = json.loads((ROOT / "assignment/api_example.json").read_text())
    answers = json.loads((ROOT / "assignment/probability_answers.json").read_text())
    assert len(answers["questions"]) == 6
    assert example["dimensions"] == len(example["embedding"]) == 300

    p("ASSIGNMENT 1", small)
    p("Coding Environment Setup<br/>and Simple API Implementation", title)
    p(f"<b>{escape(args.name)}</b> &nbsp; | &nbsp; UNI: <b>{escape(args.uni)}</b>")
    p(f'<b>Code repository:</b> <link href="{REPO}" color="#176B72">{REPO}</link>')

    h("Part I. API implementation")
    p("The existing FastAPI bigram text generator is extended with a word-embedding "
      "endpoint. It uses spaCy and the pretrained en_core_web_md 3.8.0 model. "
      "For a valid single word, the API returns the model's complete 300-dimensional static vector.")
    table([
        ["<b>Endpoint</b>", "<b>Purpose</b>"],
        ["GET /embedding?word=apple", "Return word, model, dimensions, and embedding (300 floats)."],
        ["POST /generate", "Retain bigram generation using start_word and length."],
        ["GET / and GET /docs", "Original Hello World response and interactive Swagger documentation."],
    ], [212, 304])
    h("Run on the instructor's machine")
    equations("git clone https://github.com/HotcatX/API-test.git\n"
              "cd API-test\n"
              "docker build -t api-test .\n"
              "docker run --rm -p 127.0.0.1:8000:80 api-test")
    p("Open <b>http://127.0.0.1:8000/docs</b>, expand GET /embedding, select "
      "Try it out, enter apple, and Execute. Docker must be installed and running. "
      "The image build downloads the locked dependencies and model; requests need no API key or model download.")
    equations("curl 'http://127.0.0.1:8000/embedding?word=apple'")
    prefix = ", ".join(f"{number:.6f}" for number in example["embedding"][:5])
    p(f"<b>Observed result:</b> word=apple, model=en_core_web_md, dimensions=300. "
      f"First five coordinates, rounded for display: [{prefix}]. "
      "The API and assignment/api_example.json contain all 300 coordinates.")
    p("<b>Design:</b> app/main.py defines routes and schemas; app/embedding_model.py "
      "loads and caches the model and reads Token.vector. Input is trimmed and lowercased. "
      "Invalid or out-of-vocabulary words return 422; an unavailable model returns 503. "
      "Runtime dependencies are locked in uv.lock; local installation uses the project .venv.")

    story.append(PageBreak())
    p("Part II. Rules of Probability", title)
    p("Notation: 'A and B' denotes intersection, 'A or B' denotes union, and 'not A' "
      "denotes the complement. Conditional probabilities use P(A | B). "
      "The following solutions show the rule, substitution, and result for each question.")
    h("Question 1. Independent events")
    p("Given P(A)=0.4 and P(B)=0.3, with A and B independent.")
    equations("(a) P(A and B) = P(A) P(B)\n"
              "               = 0.4 * 0.3 = 0.12 = 3/25.\n\n"
              "(b) P(A or B) = P(A) + P(B) - P(A and B)\n"
              "              = 0.4 + 0.3 - 0.12 = 0.58 = 29/50.")
    p("The intersection is subtracted in part (b) so that outcomes in both events are counted only once.")
    h("Question 2. Are the events independent?")
    p("Given P(A)=0.5, P(B)=0.4, and P(A | B)=0.7. Because P(B)&gt;0, "
      "independence requires P(A | B)=P(A). Here, 0.7 differs from 0.5, so "
      "<b>A and B are not independent</b>. Knowing B occurred changes the probability of A.")
    equations("P(A and B) = P(A | B) P(B) = 0.7 * 0.4 = 0.28.\n"
              "P(A) P(B)  = 0.5 * 0.4 = 0.20.\n"
              "Since 0.28 != 0.20, the product rule also fails.")
    h("Question 3. Bayes' rule")
    p("Given P(A)=0.6, P(B | A)=0.5, and P(B | not A)=0.2. "
      "First use the law of total probability, then Bayes' rule.")
    equations("P(not A) = 1 - 0.6 = 0.4.\n"
              "P(B) = P(B | A) P(A) + P(B | not A) P(not A)\n"
              "     = 0.5 * 0.6 + 0.2 * 0.4\n"
              "     = 0.30 + 0.08 = 0.38.\n\n"
              "P(A | B) = P(B | A) P(A) / P(B)\n"
              "         = 0.30 / 0.38 = 15/19\n"
              "         = 0.789473684... (approximately 78.95%).")

    story.append(PageBreak())
    h("Question 4. Probability of disease after a positive test")
    p("Let D mean disease and T+ mean a positive result. The given rates are "
      "P(D)=0.02, P(T+ | D)=0.95, and P(T- | not D)=0.90. "
      "Therefore P(not D)=0.98, and the false-positive rate is 1-0.90=0.10.")
    equations("P(T+) = P(T+ | D) P(D) + P(T+ | not D) P(not D)\n"
              "      = 0.95 * 0.02 + 0.10 * 0.98\n"
              "      = 0.019 + 0.098 = 0.117.\n\n"
              "P(D | T+) = P(T+ | D) P(D) / P(T+)\n"
              "          = 0.019 / 0.117 = 19/117\n"
              "          = 0.162393162... (approximately 16.24%).")
    p("Using the supplied rates in a hypothetical group of 10,000: 200 have the "
      "disease, of whom 190 test positive; among 9,800 healthy people, 980 test positive. "
      "Thus 190/(190+980)=19/117. Low prevalence explains the difference between this posterior and sensitivity.")

    h("Question 5. Expectation, variance, and sample mean")
    table([
        ["<b>x</b>", "<b>P(X=x)</b>", "<b>x P(X=x)</b>", "<b>x<super>2</super> P(X=x)</b>"],
        [85, "0.375", "31.875", "2709.375"],
        [90, "0.375", "33.750", "3037.500"],
        [95, "0.125", "11.875", "1128.125"],
        [100, "0.125", "12.500", "1250.000"],
        ["<b>Total</b>", "<b>1</b>", "<b>90</b>", "<b>8125</b>"],
    ], [66, 130, 155, 165])
    equations("(a) E[X] = sum x P(X=x)\n"
              "         = 31.875 + 33.750 + 11.875 + 12.500 = 90.\n\n"
              "(b) E[X^2] = 2709.375 + 3037.500 + 1128.125\n"
              "             + 1250.000 = 8125.\n"
              "    Var(X) = E[X^2] - (E[X])^2\n"
              "           = 8125 - 90^2 = 25 points squared.\n\n"
              "(c) Sample mean = (85+90+85+95+90+85+100+90)/8\n"
              "                = 720/8 = 90 points.")
    p("The requested variance is the variance of the given distribution. "
      "The sample mean equals E[X] here because the sample proportions (3/8, 3/8, 1/8, 1/8) "
      "match the given probabilities. Other random samples need not have mean 90.")

    story.append(PageBreak())
    h("Question 6. Entropy in bits")
    p("(a) For message probabilities 0.4, 0.3, 0.2, and 0.1, use base-2 logarithms:")
    equations("H(X) = -sum p(x) log2 p(x)\n"
              "     = -[0.4 log2(0.4) + 0.3 log2(0.3)\n"
              "         + 0.2 log2(0.2) + 0.1 log2(0.1)].")
    table([
        ["<b>Message</b>", "<b>p</b>", "<b>log2(p)</b>", "<b>-p log2(p)</b>"],
        ["A", "0.4", "-1.321928095", "0.528771238"],
        ["B", "0.3", "-1.736965594", "0.521089678"],
        ["C", "0.2", "-2.321928095", "0.464385619"],
        ["D", "0.1", "-3.321928095", "0.332192809"],
    ], [80, 76, 180, 180])
    p("Adding the contributions without intermediate rounding gives "
      "<b>H(X)=1.8464393446710154 bits</b>, or approximately <b>1.84644 bits</b>.")
    p("(b) For four equally likely messages, each probability is 1/4, and log2(1/4)=-2.")
    equations("H(uniform) = -4[(1/4) log2(1/4)]\n"
              "           = -4[(1/4)(-2)] = 2 bits.")
    p("This equals log2(4), the maximum entropy for four possible messages, "
      "and exceeds the nonuniform distribution's entropy because the messages are equally unpredictable.")

    h("Reproducibility and verification")
    p(f"Local verification: <b>{evidence['tests_passed']} tests passed</b>; "
      "HTTP checks passed for the embedding endpoint, invalid/OOV input, original text generation, "
      "and Swagger documentation. The returned vectors were compared with an independently loaded spaCy model.")
    p(f"Environment: Python {escape(evidence['python'])}, spaCy {escape(evidence['spacy'])}, "
      f"{escape(evidence['model'])} {escape(evidence['model_version'])}.")
    if evidence.get("docker_validation") == "passed":
        p("<b>Docker verification passed on GitHub Actions (Linux):</b> the image was built, "
          "the container started, and real HTTP embedding and regression checks passed.")
        p(f'<link href="{escape(evidence["workflow_url"])}" color="#176B72">View the verified workflow run</link>.', small)
    else:
        p(f'<link href="{REPO}/actions/workflows/verify.yml" color="#176B72">GitHub Actions</link> '
          "contains a Docker build-and-HTTP-check job. Check its run status for container verification.")
    equations("uv run pytest -q\n"
              "uv run python scripts/check_probability.py\n"
              "uv run python scripts/smoke_api.py  # server must be running")
    p("The probability checker uses exact Fraction arithmetic for Questions 1-5 and "
      "math.log2 for Question 6. It checks the saved numerical answers against independent calculations.", small)
    p('Sources: Assignment1-1.pdf, Questions 1-6; '
      '<link href="https://spacy.io/usage/linguistic-features#vectors-similarity">spaCy vectors documentation</link>; '
      '<link href="https://spacy.io/models/en#en_core_web_md">en_core_web_md model documentation</link>.', small)

    output = ROOT / "output/pdf/Assignment1_Submission.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=letter, rightMargin=48, leftMargin=48,
                            topMargin=48, bottomMargin=46, title="Assignment 1 - He Xiao (hx2425)",
                            author=args.name)

    def decorate(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CFDBE3"))
        canvas.line(48, 755, 564, 755)
        canvas.setFont("Vera", 7.4)
        canvas.setFillColor(MUTED)
        canvas.drawString(48, 764, "APPLIED GENERATIVE AI  /  ASSIGNMENT 1")
        canvas.line(48, 35, 564, 35)
        canvas.drawString(48, 22, f"{args.name}  |  {args.uni}")
        canvas.drawRightString(564, 22, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(output)


if __name__ == "__main__":
    main()
