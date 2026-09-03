"""Generate the print-ready proofing HTML by injecting EN/KN paragraph pairs
(parsed from the authoritative sample file) into the styled template."""

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "scandal_in_bohemia_sample.txt"
TEMPLATE_PATH = ROOT / "data" / "scandal_in_bohemia_proof.html"
OUTPUT_HTML_PATH = ROOT / "data" / "scandal_in_bohemia_proof_final.html"

PAIR_RE = re.compile(
    r"\[(\d+)\] EN: (.*)\n\[\1\] KN: (.*)", re.MULTILINE
)

PARA_TEMPLATE = """\
    <div class="para">
      <span class="para-number">¶ {num}</span>
      <div class="row row-en">
        <span class="row-label">EN</span>
        <span class="row-text">{en}</span>
      </div>
      <div class="row row-kn">
        <span class="row-label kn">KN</span>
        <span class="row-text kn">{kn}</span>
      </div>
      <div class="correction">
        <div class="correction-label">Reviewer's correction</div>
        <div class="correction-line"></div>
        <div class="correction-line"></div>
      </div>
    </div>
"""

if __name__ == "__main__":
    sample_text = SAMPLE_PATH.read_text(encoding="utf-8")
    pairs = PAIR_RE.findall(sample_text)
    print(f"Parsed {len(pairs)} EN/KN pairs from {SAMPLE_PATH.name}")

    blocks = []
    for num, en, kn in pairs:
        blocks.append(
            PARA_TEMPLATE.format(num=num, en=html.escape(en), kn=html.escape(kn))
        )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    final_html = template.replace("<!--PARAGRAPHS-->", "\n".join(blocks))
    OUTPUT_HTML_PATH.write_text(final_html, encoding="utf-8")
    print(f"Wrote {OUTPUT_HTML_PATH}")