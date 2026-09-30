# Changelog

All notable changes to this project are documented here. This changelog follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- An option to remove Project Gutenberg text, metadata, links and navigation
  entries from output EPUBs before sharing; the run log reports any Gutenberg
  mentions left.
- Project Gutenberg's own generated covers are replaced with a Kannada
  title/author cover when Gutenberg text is stripped; original scanned covers
  are kept.
- A "Using translations responsibly" section in the README and a rights and
  privacy notice on the app's Translate page.
- Output EPUBs carry a `dc:contributor` entry marking them as unreviewed
  machine translation.

### Changed

- Project Gutenberg boilerplate (`pg-boilerplate` elements) is skipped when
  reading, wherever it appears. Older Gutenberg EPUBs no longer have their
  licence machine-translated, and single-file Gutenberg books are no longer
  skipped entirely. The default `exclude_ids` is now `coverpage-wrapper`
  only; saved settings with the old default are migrated.

### Fixed

- Books whose manifest uses URL-encoded file names (e.g. `%20`) failed when
  writing the translated EPUB.

### Removed

- The EbookLib dependency, which is AGPL-3.0 and would have made packaged
  builds subject to the AGPL. EPUBs are now read with `zipfile` and `lxml`;
  chapters, paragraph indices and metadata are unchanged, so existing
  checkpoints stay valid.

## [0.1.0] - 2026-09-30

### Added

- English-to-Kannada EPUB translation with preserved source layout, links,
  images, and embedded Noto Sans Kannada fonts.
- Cell-by-cell translation of table text while retaining table structure.
- English and Kannada sentence splitting with abbreviation handling and
  protection against splitting quoted speech incorrectly.
- Optional QA back-translation and similarity scoring, with a single retry for
  middle-band scores and review flags for low scores.
- Cloud and local translation, editing, QA, and narration providers.
- A local web app for translation, settings, library browsing, and side-by-side
  reading/review.
- Optional Kannada audiobook narration.
- A headless browser test for the local web app.
