# Changelog

All notable changes to this project are documented here. This changelog follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
