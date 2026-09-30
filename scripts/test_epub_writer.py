"""Smoke test for the EPUB writer (PRD FR-1.3 / FR-5): replace translated
paragraphs in place, repackage, and prove everything else survived. Uses FAKE
translations and the real Sherlock Holmes EPUB — no model or network needed.

Run: .venv/bin/python scripts/test_epub_writer.py
"""

import posixpath
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import lxml.etree as ET
from bs4 import BeautifulSoup

from kannada_epub.epub_io import load_epub_chapters
from kannada_epub.epub_writer import (
    MACHINE_TRANSLATION_CONTRIBUTOR,
    translations_from_batches,
    write_translated_epub,
)

ROOT = Path(__file__).resolve().parent.parent
EPUB = ROOT / "data" / "sherlock_holmes.epub"

OPF_NS = "http://www.idpf.org/2007/opf"
DC_NS = "http://purl.org/dc/elements/1.1/"
CONTAINER_NS = "urn:oasis:names:tc:opendocument:xmlns:container"


def _opf_path(zf: zipfile.ZipFile) -> str:
    root = ET.fromstring(zf.read("META-INF/container.xml"))
    rootfile = root.find(f".//{{{CONTAINER_NS}}}rootfile")
    return rootfile.get("full-path")


def _manifest(zf: zipfile.ZipFile, opf_path: str) -> dict[str, str]:
    root = ET.fromstring(zf.read(opf_path))
    return {
        item.get("id"): item.get("href")
        for item in root.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
    }


def main() -> None:
    chapters = load_epub_chapters(EPUB)
    by_id = {c.id: c for c in chapters}
    item4 = by_id["item4"]

    # --- translations_from_batches mapping ---------------------------------
    n = 10
    fake = [f"ಕನ್ನಡ ಪರೀಕ್ಷೆ {i}" for i in range(n)]
    batches = [
        {
            "chapter_id": "item4",
            "chapter_title": item4.title,
            "paragraph_start": 0,
            "paragraph_end": n,
            "source_english": [p.text for p in item4.paragraphs[:n]],
            "draft_kannada": fake,
            "edited_kannada": fake,
            "edited_emotions": ["neutral"] * n,
            "prior_context_used": "",
        }
    ]
    translations = translations_from_batches(chapters, batches)
    assert len(translations["item4"]) == n
    for k in range(n):
        assert translations["item4"][item4.paragraphs[k].index] == fake[k]

    # A batch starting mid-chapter maps k -> Paragraph.index of start + k.
    mid = translations_from_batches(
        chapters,
        [{"chapter_id": "item4", "paragraph_start": 5, "edited_kannada": ["ಐದು", "ಆರು"]}],
    )
    assert mid["item4"][item4.paragraphs[5].index] == "ಐದು"
    assert mid["item4"][item4.paragraphs[6].index] == "ಆರು"

    # Out-of-range position -> ValueError.
    try:
        translations_from_batches(
            chapters,
            [
                {
                    "chapter_id": "item4",
                    "paragraph_start": len(item4.paragraphs),
                    "edited_kannada": ["ಎಲ್ಲವೂ ಮೀರಿದೆ"],
                }
            ],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an out-of-range position")

    # Unknown chapter -> ValueError.
    try:
        translations_from_batches(
            chapters, [{"chapter_id": "does-not-exist", "paragraph_start": 0, "edited_kannada": ["x"]}]
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an unknown chapter")

    # --- write the translated epub -----------------------------------------
    tmpdir = Path(tempfile.mkdtemp())
    try:
        out = tmpdir / "sherlock_holmes.kn.epub"
        write_translated_epub(EPUB, translations, out)
        assert out.exists()

        with zipfile.ZipFile(EPUB) as zsrc, zipfile.ZipFile(out) as zout:
            src_names = zsrc.namelist()
            out_names = zout.namelist()
            assert set(src_names) <= set(out_names), set(src_names) - set(out_names)

            out_infos = zout.infolist()
            assert out_infos[0].filename == "mimetype"
            assert out_infos[0].compress_type == zipfile.ZIP_STORED

            opf_path = _opf_path(zsrc)
            doc_name = posixpath.normpath(
                posixpath.join(posixpath.dirname(opf_path), _manifest(zsrc, opf_path)["item4"])
            )
            modified = {opf_path, doc_name}

            # Everything we did not touch is byte-for-byte identical.
            for name in src_names:
                if name in modified:
                    continue
                assert zsrc.read(name) == zout.read(name), f"entry changed: {name}"

            # Added package files are present...
            for required in (
                "NotoSansKannada-Regular.ttf",
                "NotoSansKannada-Bold.ttf",
                "OFL.txt",
                "kannada.css",
            ):
                assert any(n.endswith(required) for n in out_names), required

            # ... and listed in the OPF manifest, with language switched to kn.
            opf_out = ET.fromstring(zout.read(opf_path))
            items = opf_out.findall(f".//{{{OPF_NS}}}manifest/{{{OPF_NS}}}item")
            out_hrefs = {item.get("href") for item in items}
            assert any(h.endswith("kannada.css") for h in out_hrefs)
            assert any(h.endswith("NotoSansKannada-Regular.ttf") for h in out_hrefs)
            assert any(h.endswith("NotoSansKannada-Bold.ttf") for h in out_hrefs)
            assert any(h.endswith("OFL.txt") for h in out_hrefs)
            languages = opf_out.findall(f".//{{{DC_NS}}}language")
            assert languages and all(lang.text == "kn" for lang in languages)
            contributors = [
                c.text for c in opf_out.findall(f".//{{{DC_NS}}}contributor")
            ]
            assert contributors.count(MACHINE_TRANSLATION_CONTRIBUTOR) == 1, contributors

            # Every modified XHTML document is well-formed XML.
            for name in modified:
                if name.endswith((".xhtml", ".html", ".htm")):
                    ET.fromstring(zout.read(name))

            # Every id in the source document survives in the output.
            src_soup = BeautifulSoup(zsrc.read(doc_name), "lxml")
            out_soup = BeautifulSoup(zout.read(doc_name), "lxml")
            src_ids = {tag["id"] for tag in src_soup.find_all(attrs={"id": True})}
            out_ids = {tag["id"] for tag in out_soup.find_all(attrs={"id": True})}
            assert src_ids <= out_ids, f"lost ids: {src_ids - out_ids}"

            # Language is switched and the Kannada stylesheet is linked last.
            assert out_soup.html.get("lang") == "kn"
            assert out_soup.html.get("xml:lang") == "kn"
            links = out_soup.head.find_all("link")
            assert links and links[-1].get("href", "").endswith("kannada.css")
            assert links[-1].get("rel") == ["stylesheet"]

        # --- re-load the OUTPUT through the ordinary loader -----------------
        reloaded = {c.id: c for c in load_epub_chapters(out)}
        item4_out = reloaded["item4"]
        assert len(item4_out.paragraphs) == len(item4.paragraphs)
        for k in range(n):
            assert item4_out.paragraphs[k].text == fake[k], (
                f"paragraph {k}: {item4_out.paragraphs[k].text!r} != {fake[k]!r}"
            )
        assert [p.text for p in item4_out.paragraphs[n:]] == [
            p.text for p in item4.paragraphs[n:]
        ]
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print("test_epub_writer: all assertions passed")


if __name__ == "__main__":
    main()
