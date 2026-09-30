(() => {
  "use strict";

  // API and state
  const token = document.querySelector('meta[name="app-token"]').content;
  const content = document.getElementById("content");
  const ollamaBaseUrl = "http://localhost:11434/v1";
  const sarvamBaseUrl = "https://api.sarvam.ai/v1";
  const appState = {
    state: null,
    book: null,
    run: null,
    queue: null,
    books: [],
    sectionSelections: null,
    stopConfirm: false,
    poll: null,
  };

  async function api(url, options = {}) {
    const headers = {
      "X-App-Token": token,
      ...(options.headers || {}),
    };
    if (options.body && !(options.body instanceof FormData)) {
      headers["Content-Type"] = "application/json";
    }

    const response = await fetch(url, { ...options, headers });
    const type = response.headers.get("content-type") || "";
    const data = type.includes("json") ? await response.json() : await response.text();
    if (!response.ok) {
      throw new Error(data?.detail || `Request failed (${response.status})`);
    }
    return data;
  }

  function json(body) {
    return JSON.stringify(body);
  }

  function escapeHtml(value) {
    const replacements = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;",
    };
    return String(value ?? "").replace(/[&<>"']/g, character => replacements[character]);
  }

  function inlineMessage(text, isError = false) {
    const errorClass = isError ? " error" : "";
    return `<div class="inline-message${errorClass}" role="status">` +
      `${escapeHtml(text)}</div>`;
  }

  function pageHeader(eyebrow, title, copy) {
    return `<header class="page-head">
      <p class="eyebrow">${escapeHtml(eyebrow)}</p>
      <h1>${escapeHtml(title)}</h1>
      <p class="lede">${escapeHtml(copy)}</p>
    </header>`;
  }

  function button(label, className = "", attributes = "") {
    return `<button class="button ${className}" ${attributes}>` +
      `${escapeHtml(label)}</button>`;
  }

  async function refreshState() {
    appState.state = await api("/api/state");
    return appState.state;
  }

  // Router
  function routeFromHash() {
    return decodeURIComponent(location.hash.slice(1) || "/translate").split("?")[0];
  }

  function setActiveNavigation(route) {
    document.querySelectorAll(".sidebar nav a").forEach(link => {
      link.classList.toggle("active", link.dataset.route === route);
    });
  }

  async function renderRoute() {
    clearInterval(appState.poll);
    appState.poll = null;

    const route = routeFromHash();
    if (route === "/translate") {
      setActiveNavigation("translate");
      await renderTranslate();
    } else if (route === "/queue") {
      setActiveNavigation("queue");
      await renderQueue();
    } else if (route === "/library") {
      setActiveNavigation("library");
      await renderLibrary();
    } else if (route.startsWith("/library/")) {
      setActiveNavigation("library");
      await renderLibraryBook(route);
    } else if (route === "/settings") {
      setActiveNavigation("settings");
      await renderSettings();
    } else {
      location.hash = "#/translate";
    }
  }

  window.addEventListener("hashchange", () => {
    renderRoute().catch(showFatalError);
  });

  function showFatalError(error) {
    content.innerHTML = pageHeader(
      "Kannada Book Translator",
      "The app couldn’t load",
      "Check that the local server is running and reload this page.",
    ) + inlineMessage(error.message, true);
  }

  // Translate screen
  function translationProviderSummary(settings) {
    const translation = settings.translation.provider === "indictrans2_local"
      ? "IndicTrans2 on this computer"
      : settings.translation.model === "sarvam-m"
        ? "Sarvam (cloud)"
        : "Cloud translation";
    const editor = settings.editor.provider === "anthropic"
      ? "Claude (cloud)"
      : settings.editor.provider === "ollama"
        ? "Ollama on this computer"
        : "Cloud editor";
    return `Translation: ${translation} · Editing: ${editor}`;
  }

  function editionNotesMarkup(edition) {
    if (!edition) return "";
    const parts = [];
    // Only a bare year is an edition date; e-text tools such as Project
    // Gutenberg put their release date (1999-03-01) in dc:date.
    if (/^\d{4}$/.test(edition.date || "")) {
      parts.push(`Edition date: ${escapeHtml(edition.date)}`);
    }
    if (Array.isArray(edition.front_matter) && edition.front_matter.length) {
      parts.push(`Front matter: ${escapeHtml(edition.front_matter.join(", "))}`);
    }
    const quiet = parts.length
      ? `<p class="edition-note">${parts.join(" · ")}</p>`
      : "";
    if (!edition.needs_review) return quiet;

    const findings = [];
    (edition.recent_years || []).forEach(entry => {
      findings.push(`<li>${escapeHtml(String(entry.year))} in ` +
        `“${escapeHtml(entry.chapter)}”: ${escapeHtml(entry.snippet)}</li>`);
    });
    (edition.copyright_notices || []).forEach(entry => {
      findings.push(`<li>Copyright notice in ` +
        `“${escapeHtml(entry.chapter)}”: ${escapeHtml(entry.snippet)}</li>`);
    });
    return `${quiet}<div class="source-warning edition-warning">
      <p>This edition may include text first published in ` +
        `${escapeHtml(String(edition.cutoff_year))} or later, such as a preface ` +
        `or notes. Those parts can still be under copyright even if the ` +
        `original book isn't. Check them before you share the translation.</p>
      <details class="details">
        <summary>Show what was found</summary>
        <ul>${findings.join("")}</ul>
      </details>
    </div>`;
  }

  function bookSelectionMarkup(book) {
    if (book) {
      const paragraphCount = book.chapters.reduce(
        (total, chapter) => total + chapter.paragraphs,
        0,
      );
      const problemCount = book.source_problem_count || 0;
      const problemList = Array.isArray(book.source_problems)
        ? book.source_problems
        : [];
      const sourceWarning = problemCount > 0
        ? `<div class="source-warning">
            <p>This EPUB has ${problemCount} problem(s) in the file itself.
              They'll carry over into the translation; they aren't caused by
              translating.</p>
            <details class="details">
              <summary>Show problems</summary>
              <ul>${problemList
                .map((problem) => `<li>${escapeHtml(problem)}</li>`)
                .join("")}</ul>
            </details>
          </div>`
        : "";
      return `<div class="card book-card">
        <div class="book-cover" aria-hidden="true">ಕ</div>
        <div class="book-info">
          <h3>${escapeHtml(book.title)}</h3>
          <p>${escapeHtml(book.author)} · ${book.chapters.length} chapters · ` +
            `${paragraphCount} paragraphs</p>
        </div>
        ${button("Change", "quiet", "id=change-book")}
      </div>${sourceWarning}${editionNotesMarkup(book.edition)}`;
    }

    return `<div class="dropzone" id="dropzone" tabindex="0">
      <div class="drop-icon" aria-hidden="true">⇧</div>
      <p>Drop an EPUB here, or choose a file from your computer.</p>
      <button class="button" id="choose-book">Choose EPUB…</button>
      <input class="sr-only" id="book-file" type="file"
        accept=".epub,application/epub+zip" aria-label="Choose an EPUB file">
    </div>`;
  }

  function optionsMarkup(settings) {
    return `<section class="step">
      <span class="step-number">2</span>
      <div class="step-body">
        <h2 class="step-title">Options</h2>
        <div class="card">
          <div class="option-list">
            <label class="toggle-row">
              <input class="switch" type="checkbox" id="qa-option"
                ${settings.qa.enabled ? "checked" : ""}>
              <span class="toggle-copy">
                <strong>Quality check (QA)</strong>
                <small>Compare each Kannada paragraph with the original and flag
                  anything to review.</small>
              </span>
            </label>
            <label class="toggle-row">
              <input class="switch" type="checkbox" id="audio-option">
              <span class="toggle-copy">
                <strong>Create audiobook</strong>
                <small>Make a spoken Kannada audio file after the book is
                  translated.</small>
              </span>
            </label>
            <label class="toggle-row">
              <input class="switch" type="checkbox" id="preview-option">
              <span class="toggle-copy">
                <strong>Quick preview</strong>
                <small>Translate only the first few paragraphs in each
                  chapter.</small>
              </span>
            </label>
            <div class="preview-input" id="preview-wrap">
              <label class="field-label" for="preview-count">
                Paragraphs per chapter
              </label>
              <input class="input" id="preview-count" type="number" min="1"
                value="2">
            </div>
          </div>
          <p class="provider-summary">
            ${escapeHtml(translationProviderSummary(settings))} ·
            <a href="#/settings">Change in Settings</a>
          </p>
        </div>
      </div>
    </section>`;
  }

  // Per-run section selection. Selections live in appState (never
  // localStorage) and are reset whenever another book is uploaded.
  function sectionSelection(book) {
    if (!appState.sectionSelections) {
      appState.sectionSelections = {};
      book.chapters.forEach(chapter => {
        appState.sectionSelections[chapter.id] = true;
      });
    }
    return appState.sectionSelections;
  }

  function sectionsSummary(selectedCount, total, paragraphs) {
    return `${selectedCount} of ${total} · ${paragraphs.toLocaleString()} paragraph${
      paragraphs === 1 ? "" : "s"
    }`;
  }

  function sectionsMarkup(book) {
    if (!book || !book.chapters.length) return "";
    const selections = sectionSelection(book);
    const total = book.chapters.length;
    const selected = book.chapters.filter(chapter => selections[chapter.id] !== false);
    const paragraphs = selected.reduce(
      (sum, chapter) => sum + Number(chapter.paragraphs || 0),
      0,
    );
    const rows = book.chapters.map(chapter => {
      const checked = selections[chapter.id] !== false;
      const count = Number(chapter.paragraphs) || 0;
      return `<label class="section-row">
        <input type="checkbox" data-section-id="${escapeHtml(chapter.id)}"
          data-paragraphs="${count}" ${checked ? "checked" : ""}>
        <span class="section-title">${escapeHtml(chapter.title || chapter.id)}</span>
        <span class="section-count">${count} paragraph${count === 1 ? "" : "s"}</span>
      </label>`;
    }).join("");

    return `<details class="sections" id="sections">
      <summary>Sections to translate (<span id="sections-summary">${
        sectionsSummary(selected.length, total, paragraphs)
      }</span>)</summary>
      <div class="sections-tools">
        <button class="button quiet" type="button" id="sections-all">Select all</button>
        <button class="button quiet" type="button" id="sections-none">Select none</button>
      </div>
      <div class="sections-list" id="sections-list">${rows}</div>
    </details>
    <div id="sections-warning">${
      selected.length ? "" : inlineMessage("Choose at least one section.")
    }</div>`;
  }

  function hasSelectedSections() {
    const boxes = document.querySelectorAll("[data-section-id]");
    if (!boxes.length) return true;
    return [...boxes].some(box => box.checked);
  }

  function connectSections() {
    const list = document.getElementById("sections-list");
    if (!list) return;
    list.addEventListener("change", event => {
      if (!event.target.matches("[data-section-id]")) return;
      if (appState.sectionSelections) {
        appState.sectionSelections[event.target.dataset.sectionId] = event.target.checked;
      }
      updateSectionsState();
    });
    document.getElementById("sections-all")?.addEventListener(
      "click",
      () => setAllSections(true),
    );
    document.getElementById("sections-none")?.addEventListener(
      "click",
      () => setAllSections(false),
    );
    updateSectionsState();
  }

  function setAllSections(checked) {
    document.querySelectorAll("[data-section-id]").forEach(box => {
      box.checked = checked;
      if (appState.sectionSelections) {
        appState.sectionSelections[box.dataset.sectionId] = checked;
      }
    });
    updateSectionsState();
  }

  function updateSectionsState() {
    const boxes = [...document.querySelectorAll("[data-section-id]")];
    if (!boxes.length) return;
    const selected = boxes.filter(box => box.checked);
    const paragraphs = selected.reduce(
      (sum, box) => sum + Number(box.dataset.paragraphs || 0),
      0,
    );
    const summary = document.getElementById("sections-summary");
    if (summary) {
      summary.textContent = sectionsSummary(selected.length, boxes.length, paragraphs);
    }
    const warning = document.getElementById("sections-warning");
    const startButton = document.getElementById("start-run");
    if (selected.length) {
      if (warning) warning.innerHTML = "";
      if (startButton && appState.book && appState.run?.state !== "running") {
        startButton.disabled = false;
      }
    } else {
      if (warning) warning.innerHTML = inlineMessage("Choose at least one section.");
      if (startButton) startButton.disabled = true;
    }
  }

  async function renderTranslate() {
    const state = await refreshState();
    const run = await api("/api/run");
    appState.run = run;
    const isRunning = run.state === "running";

    content.innerHTML = `${pageHeader(
      "Translate",
      "Bring a book to Kannada",
      "Choose an EPUB and we’ll take care of the translation, formatting, and " +
        "optional quality review.",
    )}
    <section class="step">
      <span class="step-number">1</span>
      <div class="step-body">
        <h2 class="step-title">Choose a book</h2>
        ${bookSelectionMarkup(appState.book)}
      </div>
    </section>
    ${optionsMarkup(state.settings)}
    <section class="step">
      <span class="step-number">3</span>
      <div class="step-body">
        <h2 class="step-title">Translate</h2>
        <div id="precheck"></div>
        ${sectionsMarkup(appState.book)}
        <div class="actions">
          ${button(
            isRunning ? "Translation in progress…" : "Translate book",
            "primary",
            "id=start-run disabled",
          )}
          ${button("Add to queue", "", "id=add-queue")}
        </div>
        <div id="run-panel">${renderRunPanel(run)}</div>
        <p class="rights-note">
          Translate only books you own or that are in the public domain where
          you live. Translations are unreviewed machine output: don’t share or
          sell translations of copyrighted books without the rights holder’s
          permission. With a cloud provider, the book’s text is sent to that
          provider.
        </p>
      </div>
    </section>`;

    connectBookPicker();
    connectTranslateOptions();
    connectSections();
    connectRunControls(run);
  }

  function connectBookPicker() {
    const input = document.getElementById("book-file");
    const chooseButton = document.getElementById("choose-book");
    chooseButton?.addEventListener("click", () => input.click());
    input?.addEventListener("change", () => {
      if (input.files[0]) uploadBook(input.files[0]);
    });

    const changeButton = document.getElementById("change-book");
    changeButton?.addEventListener("click", () => {
      appState.book = null;
      appState.sectionSelections = null;
      renderTranslate();
    });

    const dropzone = document.getElementById("dropzone");
    if (!dropzone) return;
    dropzone.addEventListener("dragover", event => {
      event.preventDefault();
      dropzone.classList.add("drag");
    });
    dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
    dropzone.addEventListener("drop", event => {
      event.preventDefault();
      dropzone.classList.remove("drag");
      const file = event.dataTransfer.files[0];
      if (file) uploadBook(file);
    });
    dropzone.addEventListener("keydown", event => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        input.click();
      }
    });
  }

  function connectTranslateOptions() {
    const previewToggle = document.getElementById("preview-option");
    previewToggle.addEventListener("change", () => {
      document.getElementById("preview-wrap").classList.toggle(
        "show",
        previewToggle.checked,
      );
    });
  }

  async function uploadBook(file) {
    const precheck = document.getElementById("precheck");
    if (!file.name.toLowerCase().endsWith(".epub")) {
      precheck.innerHTML = inlineMessage("Choose an EPUB file ending in .epub.", true);
      return;
    }

    precheck.innerHTML = inlineMessage("Reading your book…");
    const body = new FormData();
    body.append("file", file);
    try {
      appState.book = await api("/api/books", { method: "POST", body });
      appState.sectionSelections = null;
      await renderTranslate();
    } catch (error) {
      precheck.innerHTML = inlineMessage(error.message, true);
    }
  }

  async function startRun() {
    if (!appState.book) {
      document.getElementById("precheck").innerHTML = inlineMessage(
        "Choose an EPUB before starting.",
        true,
      );
      return;
    }

    if (!hasSelectedSections()) {
      document.getElementById("precheck").innerHTML = inlineMessage(
        "Choose at least one section.",
        true,
      );
      return;
    }

    const skipChapters = [...document.querySelectorAll("[data-section-id]")]
      .filter(box => !box.checked)
      .map(box => box.dataset.sectionId);
    const quickPreview = document.getElementById("preview-option").checked;
    const request = {
      book_id: appState.book.book_id,
      qa: document.getElementById("qa-option").checked,
      audiobook: document.getElementById("audio-option").checked,
      preview_paragraphs: quickPreview
        ? Number(document.getElementById("preview-count").value)
        : null,
      skip_chapters: skipChapters,
    };

    try {
      appState.run = await api("/api/run", {
        method: "POST",
        body: json(request),
      });
      document.getElementById("precheck").innerHTML = "";
      document.getElementById("start-run").disabled = true;
      appState.poll = setInterval(pollRun, 900);
      await pollRun();
    } catch (error) {
      showStartProblem(error);
    }
  }

  function showStartProblem(error) {
    const localProblem = error.message.includes("local") ||
      error.message.includes("Install");
    const settingsSection = localProblem ? "local-mode" : "keys";
    const settingsLabel = localProblem ? "Local mode" : "API keys";
    document.getElementById("precheck").innerHTML = `
      <div class="inline-message error">
        ${escapeHtml(error.message)}
        <a href="#/settings?section=${settingsSection}">
          Open Settings (${settingsLabel})
        </a>
      </div>`;
  }

  async function addCurrentToQueue() {
    if (!appState.book) {
      document.getElementById("precheck").innerHTML = inlineMessage(
        "Choose an EPUB before adding it to the queue.",
        true,
      );
      return;
    }
    if (!hasSelectedSections()) {
      document.getElementById("precheck").innerHTML = inlineMessage(
        "Choose at least one section.",
        true,
      );
      return;
    }
    const skipChapters = [...document.querySelectorAll("[data-section-id]")]
      .filter(box => !box.checked)
      .map(box => box.dataset.sectionId);
    const request = {
      items: [{
        book_id: appState.book.book_id,
        skip_chapters: skipChapters,
      }],
      qa: document.getElementById("qa-option").checked,
      audiobook: document.getElementById("audio-option").checked,
    };
    try {
      await api("/api/queue", { method: "POST", body: json(request) });
      document.getElementById("precheck").innerHTML = `
        <div class="inline-message" role="status">
          Added to the queue.
          <a href="#/queue">View queue</a>
        </div>`;
    } catch (error) {
      document.getElementById("precheck").innerHTML = inlineMessage(error.message, true);
    }
  }

  async function pollRun() {
    try {
      appState.run = await api("/api/run");
      const panel = document.getElementById("run-panel");
      if (panel) {
        const detailsOpen = Boolean(panel.querySelector("details")?.open);
        panel.innerHTML = renderRunPanel(appState.run);
        if (detailsOpen) panel.querySelector("details")?.setAttribute("open", "");
      }
      connectRunPanelButtons(appState.run);

      if (appState.run.state !== "running") {
        clearInterval(appState.poll);
        appState.poll = null;
        const startButton = document.getElementById("start-run");
        if (startButton) {
          startButton.disabled = !appState.book || !hasSelectedSections();
        }
      }
    } catch (error) {
      const panel = document.getElementById("run-panel");
      if (panel) panel.innerHTML = inlineMessage(error.message, true);
    }
  }

  function renderRunPanel(run) {
    if (!run || run.state === "idle") return "";
    if (run.state === "running") return runningPanel(run);
    if (run.state === "finished") return finishedPanel(run);
    if (run.state === "cancelled") return cancelledPanel(run);
    if (run.state === "failed") return failedPanel(run);
    return "";
  }

  function skippedSectionsNote(run) {
    const ids = run.skipped_chapters;
    if (!Array.isArray(ids) || !ids.length) return "";
    return `<p class="muted skipped-note">Skipping ${ids.length} section(s)</p>`;
  }

  function runningPanel(run) {
    const percent = run.chapter_total
      ? Math.min(100, Math.round(run.chapter_index / run.chapter_total * 100))
      : 3;
    const stage = run.narrating
      ? `Narrating ${run.narrating.done} of ${run.narrating.total}`
      : run.stage === "qa"
        ? "Checking quality"
        : run.stage === "epub"
          ? "Writing EPUB"
          : run.stage === "audiobook"
            ? "Preparing audiobook"
            : "Translating";
    const chapter = run.current_chapter_title
      ? `Chapter ${run.chapter_index} of ${run.chapter_total} — ` +
        escapeHtml(run.current_chapter_title)
      : "Preparing your book…";
    run.stopConfirm = appState.stopConfirm;

    return `<div class="card section-gap">
      <div class="run-meta"><strong>${chapter}</strong><span>${stage}</span></div>
      ${skippedSectionsNote(run)}
      <div class="progress-track" role="progressbar" aria-valuenow="${percent}"
        aria-valuemin="0" aria-valuemax="100">
        <div class="progress-fill" style="width:${percent}%"></div>
      </div>
      <div class="run-meta">
        <span>${formatElapsed(run.elapsed_seconds)} elapsed</span>
        ${button("Stop", "quiet danger", "id=stop-run")}
      </div>
      ${run.stopConfirm ? stopConfirmation() : ""}
      ${detailsDisclosure(run.log_tail)}
    </div>`;
  }

  function stopConfirmation() {
    return `<div class="inline-message">
      Stop after the current chapter? Completed chapters will be kept.
      <button class="button danger" id="confirm-stop">Stop translation</button>
      <button class="button quiet" id="keep-going">Keep translating</button>
    </div>`;
  }

  // "Preview: 24 of 2,538 paragraphs translated."
  function coverageText(info) {
    const done = Number(info.paragraphs_translated || 0).toLocaleString();
    const total = Number(info.paragraphs_total || 0).toLocaleString();
    return `Preview: ${done} of ${total} paragraphs translated.`;
  }

  function finishedPanel(run) {
    const result = run.result || {};
    const downloads = [
      result.epub && `<a class="button" href="${escapeHtml(result.epub)}">Download EPUB</a>`,
      result.qa_report && `<a class="button" href="${escapeHtml(result.qa_report)}">QA report</a>`,
      result.audiobook && `<a class="button" href="${escapeHtml(result.audiobook)}">Audiobook</a>`,
    ].filter(Boolean).join("");
    const summary = result.qa_summary
      ? qaChips(result.qa_summary, `#/library/${encodeURIComponent(run.book_id)}?flagged=1`)
      : "";

    const heading = result.preview ? "Your preview is ready" : "Your Kannada book is ready";
    const intro = result.preview
      ? `${coverageText(result)} The rest of the book stays in English. Turn off
        Quick preview to translate the whole book.`
      : "Open it in your reading app, or find the files in the output folder.";

    return `<div class="card result-card section-gap">
      <div class="result-icon" aria-hidden="true">✓</div>
      <h2>${escapeHtml(heading)}</h2>
      <p>${escapeHtml(intro)}</p>
      ${skippedSectionsNote(run)}
      ${summary}
      <div class="actions">
        ${result.epub ? button("Open in Books app", "primary", "data-open=epub") : ""}
        ${button("Show in folder", "", "data-open=folder")}
        ${downloads}
      </div>
      ${detailsDisclosure(run.log_tail)}
    </div>`;
  }

  function cancelledPanel(run) {
    return `<div class="card section-gap">
      <h2>Translation stopped</h2>
      <p class="muted">Completed chapters are saved. Start again when you’re ready;
        the app will resume where it can.</p>
      ${button("Show in folder", "", "data-open=folder")}
      ${detailsDisclosure(run.log_tail)}
    </div>`;
  }

  function failedPanel(run) {
    return `<div class="card section-gap">
      <h2>Translation couldn’t finish</h2>
      <p>${escapeHtml(run.error || "An unexpected error occurred.")}</p>
      ${button("Show in folder", "", "data-open=folder")}
      ${detailsDisclosure(run.log_tail)}
    </div>`;
  }

  function detailsDisclosure(lines = []) {
    if (!lines.length) return "";
    return `<details class="details">
      <summary>Show details</summary>
      <pre class="log">${escapeHtml(lines.join("\n"))}</pre>
    </details>`;
  }

  function formatElapsed(seconds) {
    const total = Number(seconds) || 0;
    return total < 60
      ? `${total} sec`
      : `${Math.floor(total / 60)} min ${total % 60} sec`;
  }

  function qaChips(summary, href) {
    return `<div class="qa-chips">
      <span class="chip pass">✓ ${summary.pass} passed</span>
      <span class="chip retry">↻ ${summary.retry} retried</span>
      <a class="chip flag" href="${href}">⚑ ${summary.flagged} flagged</a>
    </div>`;
  }

  // Manifest "epubcheck" summary: nothing when the check never ran.
  function epubcheckMarkup(epubcheck) {
    if (!epubcheck) return "";
    const plural = (count, word) => `${count} ${word}${count === 1 ? "" : "s"}`;
    if (epubcheck.error) {
      return `<p class="epubcheck-note muted">EPUBCheck could not run: ` +
        `${escapeHtml(epubcheck.error)}</p>`;
    }
    const output = epubcheck.output || {};
    const errors = (output.fatals || 0) + (output.errors || 0);
    const newErrors = epubcheck.new_errors || 0;
    let summary;
    if (errors === 0) {
      summary = "EPUBCheck: passed (0 errors)";
    } else if (newErrors === 0) {
      summary = `EPUBCheck: ${plural(errors, "error")} ` +
        `(all also in the source file)`;
    } else {
      summary = `EPUBCheck: ${plural(newErrors, "new error")}`;
    }
    const messages = Array.isArray(output.messages) ? output.messages : [];
    const details = messages.length
      ? `<details class="details">
          <summary>Show EPUBCheck messages</summary>
          <ul>${messages.map(message => `<li>${escapeHtml(message)}</li>`).join("")}</ul>
        </details>`
      : "";
    return `<div class="epubcheck-note">
      <p>${escapeHtml(summary)}</p>
      ${details}
    </div>`;
  }

  function connectRunControls(run) {
    const startButton = document.getElementById("start-run");
    if (run.state === "running") {
      startButton.disabled = true;
      appState.poll = setInterval(pollRun, 900);
    } else if (appState.book) {
      startButton.disabled = !hasSelectedSections();
    }
    startButton.addEventListener("click", startRun);
    connectRunPanelButtons(run);
    document.getElementById("add-queue")?.addEventListener("click", addCurrentToQueue);
  }

  function connectRunPanelButtons(run) {
    document.getElementById("stop-run")?.addEventListener("click", () => {
      appState.stopConfirm = true;
      const panel = document.getElementById("run-panel");
      if (panel) panel.innerHTML = renderRunPanel(run);
      connectRunPanelButtons(run);
    });
    document.getElementById("confirm-stop")?.addEventListener("click", async () => {
      appState.stopConfirm = false;
      try {
        appState.run = await api("/api/run/stop", { method: "POST" });
        await pollRun();
      } catch (error) {
        document.getElementById("precheck").innerHTML = inlineMessage(error.message, true);
      }
    });
    document.getElementById("keep-going")?.addEventListener("click", () => {
      appState.stopConfirm = false;
      run.stopConfirm = false;
      const panel = document.getElementById("run-panel");
      if (panel) panel.innerHTML = renderRunPanel(run);
      connectRunPanelButtons(run);
    });
    document.querySelectorAll("[data-open]").forEach(element => {
      element.addEventListener("click", () => openOutput(run.book_id, element.dataset.open));
    });
  }

  async function openOutput(bookId, what) {
    try {
      await api(`/api/library/${encodeURIComponent(bookId)}/open`, {
        method: "POST",
        body: json({ what }),
      });
    } catch (error) {
      document.getElementById("precheck").innerHTML = inlineMessage(error.message, true);
    }
  }

  // Queue screen
  function queueStateLabel(item, run) {
    if (item.state === "pending") return "Waiting";
    if (item.state === "running") {
      if (run && run.state === "running" && run.queue_item_id === item.id) {
        const progress = run.chapter_total
          ? ` — chapter ${run.chapter_index} of ${run.chapter_total}`
          : "";
        return `Translating…${progress}`;
      }
      return "Translating…";
    }
    if (item.state === "done") return "Done";
    if (item.state === "failed") return `Failed: ${item.error || "Unknown error"}`;
    if (item.state === "cancelled") return "Stopped";
    return item.state;
  }

  function queueItemMarkup(item, run, index, total) {
    const running = item.state === "running";
    const canRetry = item.state === "failed" || item.state === "cancelled";
    const title = escapeHtml(item.title || item.book_id);
    const itemId = escapeHtml(item.id);
    const buttons = [];
    if (!running) {
      buttons.push(`<button class="button quiet" data-queue-action="up" ` +
        `data-item-id="${itemId}" aria-label="Move ${title} up" ` +
        `${index === 0 ? "disabled" : ""}>Up</button>`);
      buttons.push(`<button class="button quiet" data-queue-action="down" ` +
        `data-item-id="${itemId}" aria-label="Move ${title} down" ` +
        `${index === total - 1 ? "disabled" : ""}>Down</button>`);
      buttons.push(`<button class="button quiet danger" data-queue-action="remove" ` +
        `data-item-id="${itemId}" aria-label="Remove ${title}">Remove</button>`);
    }
    if (canRetry) {
      buttons.push(`<button class="button quiet" data-queue-action="retry" ` +
        `data-item-id="${itemId}" aria-label="Retry ${title}">Retry</button>`);
    }
    const library = item.state === "done"
      ? `<a href="#/library/${encodeURIComponent(item.book_id)}">Read in Library</a>`
      : "";
    return `<article class="card queue-row">
      <div class="book-info">
        <h3>${title}</h3>
        <p class="queue-state ${escapeHtml(item.state)}">${
          escapeHtml(queueStateLabel(item, run))
        }</p>
        ${library}
      </div>
      <div class="actions queue-row-actions">${buttons.join("")}</div>
    </article>`;
  }

  function queueListMarkup(queue, run) {
    const items = queue.items || [];
    if (!items.length) {
      return `<div class="card empty"><p>The queue is empty. Add books above.</p></div>`;
    }
    return items.map((item, index) =>
      queueItemMarkup(item, run, index, items.length)).join("");
  }

  function queueControlsMarkup(queue, run) {
    const running = run && run.state === "running";
    const startButton = queue.active
      ? button("Pause queue", "", "data-queue-control=pause")
      : button("Start queue", "primary", "data-queue-control=start");
    return `<div class="actions">
        ${startButton}
        <button class="button quiet danger" data-queue-control="stop"
          aria-label="Stop the book that is translating" ${
            running ? "" : "disabled"
          }>Stop current book</button>
        <button class="button quiet" data-queue-control="clear">Clear finished</button>
      </div>
      <p class="muted">${queue.active
        ? "The queue is running. Pausing lets the current book finish; no new " +
          "book starts until you press Start queue."
        : "The queue is paused. Press Start queue to begin the waiting books."}</p>`;
  }

  function queueBooksMarkup(books) {
    if (!books.length) {
      return `<p class="muted">No uploaded books yet. Upload an EPUB above.</p>`;
    }
    const rows = books.map(book => `<label class="section-row">
      <input type="checkbox" data-queue-book="${escapeHtml(book.book_id)}"
        aria-label="Queue ${escapeHtml(book.title)}">
      <span class="section-title">${escapeHtml(book.title)}${
        book.author ? ` — ${escapeHtml(book.author)}` : ""
      }</span>
      ${book.unreadable ? `<span class="section-count">Unreadable</span>` : ""}
    </label>`).join("");
    return `<div class="sections-list">${rows}</div>`;
  }

  async function renderQueue() {
    const [queue, run] = await Promise.all([api("/api/queue"), api("/api/run")]);
    appState.queue = queue;
    appState.run = run;
    content.innerHTML = `${pageHeader(
      "One after another",
      "Queue",
      "Add several books and translate them in order. The queue is kept when " +
        "the app is closed and starts again paused.",
    )}
    <section class="card section-gap">
      <h2>Add books</h2>
      <p class="muted">Upload one or more EPUB files, then tick the books to
        queue.</p>
      <div class="actions">
        ${button("Choose EPUBs…", "", "id=queue-choose")}
        <input class="sr-only" id="queue-file" type="file"
          accept=".epub,application/epub+zip" multiple
          aria-label="Choose EPUB files to upload">
      </div>
      <div id="queue-upload-result"></div>
      <div id="queue-books">${queueBooksMarkup(appState.books || [])}</div>
      <label class="toggle-row section-gap">
        <input class="switch" type="checkbox" id="queue-audiobook">
        <span class="toggle-copy">
          <strong>Also make audiobooks</strong>
          <small>Narrate each queued book after it is translated.</small>
        </span>
      </label>
      <div class="actions section-gap">
        ${button("Add selected to queue", "primary", "id=queue-add")}
        <span id="queue-add-note" class="saved" aria-live="polite"></span>
      </div>
    </section>
    <section class="section-gap">
      <h2>Queue</h2>
      <div id="queue-controls">${queueControlsMarkup(queue, run)}</div>
      <div id="queue-list" class="book-list section-gap">${
        queueListMarkup(queue, run)
      }</div>
    </section>
    <p class="rights-note">The queue is kept when the app is closed. It always
      starts paused after reopening; press Start queue to continue where each
      book stopped.</p>`;
    connectQueuePage();
    await loadQueueBooks();
    startQueuePollingIfNeeded(queue, run);
  }

  async function loadQueueBooks() {
    const container = document.getElementById("queue-books");
    try {
      appState.books = await api("/api/books");
      if (container) container.innerHTML = queueBooksMarkup(appState.books);
    } catch (error) {
      if (container) container.innerHTML = inlineMessage(error.message, true);
    }
  }

  function connectQueuePage() {
    const input = document.getElementById("queue-file");
    document.getElementById("queue-choose")?.addEventListener("click", () => input.click());
    input?.addEventListener("change", () => {
      if (input.files.length) uploadQueueBooks([...input.files]);
    });
    document.getElementById("queue-add")?.addEventListener("click", addSelectedToQueue);
    document.getElementById("queue-controls").addEventListener("click", event => {
      const control = event.target.closest("[data-queue-control]");
      if (control) queueControl(control.dataset.queueControl);
    });
    document.getElementById("queue-list").addEventListener("click", event => {
      const action = event.target.closest("[data-queue-action]");
      if (action) queueItemAction(action.dataset.queueAction, action.dataset.itemId);
    });
  }

  async function uploadQueueBooks(files) {
    const result = document.getElementById("queue-upload-result");
    const messages = [];
    result.innerHTML = inlineMessage(`Uploading ${files.length} file(s)…`);
    for (const file of files) {
      if (!file.name.toLowerCase().endsWith(".epub")) {
        messages.push(inlineMessage(`${file.name}: choose an EPUB file ending in .epub.`, true));
        continue;
      }
      const body = new FormData();
      body.append("file", file);
      try {
        const uploaded = await api("/api/books", { method: "POST", body });
        messages.push(inlineMessage(`${uploaded.title} uploaded.`));
      } catch (error) {
        messages.push(inlineMessage(`${file.name}: ${error.message}`, true));
      }
    }
    result.innerHTML = messages.join("");
    await loadQueueBooks();
  }

  async function addSelectedToQueue() {
    const note = document.getElementById("queue-add-note");
    const bookIds = [...document.querySelectorAll("[data-queue-book]:checked")]
      .map(box => box.dataset.queueBook);
    if (!bookIds.length) {
      note.textContent = "Tick at least one book.";
      return;
    }
    const audiobook = document.getElementById("queue-audiobook").checked;
    try {
      await api("/api/queue", {
        method: "POST",
        body: json({ items: bookIds.map(book_id => ({ book_id })), audiobook }),
      });
      note.textContent = "Added to the queue.";
      await refreshQueueView();
    } catch (error) {
      note.textContent = error.message;
    }
  }

  async function queueControl(action) {
    const note = document.getElementById("queue-add-note");
    const endpoints = {
      start: "/api/queue/start",
      pause: "/api/queue/pause",
      stop: "/api/run/stop",
      clear: "/api/queue/clear",
    };
    if (!endpoints[action]) return;
    try {
      await api(endpoints[action], { method: "POST" });
      await refreshQueueView();
    } catch (error) {
      if (note) note.textContent = error.message;
    }
  }

  async function queueItemAction(action, itemId) {
    const note = document.getElementById("queue-add-note");
    const encoded = encodeURIComponent(itemId);
    try {
      if (action === "up" || action === "down") {
        await api(`/api/queue/${encoded}/move`, {
          method: "POST",
          body: json({ direction: action }),
        });
      } else if (action === "remove") {
        await api(`/api/queue/${encoded}`, { method: "DELETE" });
      } else if (action === "retry") {
        await api(`/api/queue/${encoded}/retry`, { method: "POST" });
      } else {
        return;
      }
      await refreshQueueView();
    } catch (error) {
      if (note) note.textContent = error.message;
    }
  }

  async function refreshQueueView() {
    const [queue, run] = await Promise.all([api("/api/queue"), api("/api/run")]);
    appState.queue = queue;
    appState.run = run;
    const controls = document.getElementById("queue-controls");
    if (controls) controls.innerHTML = queueControlsMarkup(queue, run);
    const list = document.getElementById("queue-list");
    if (list) list.innerHTML = queueListMarkup(queue, run);
    startQueuePollingIfNeeded(queue, run);
  }

  function startQueuePollingIfNeeded(queue, run) {
    const shouldPoll = queue.active || (run && run.state === "running");
    if (shouldPoll && !appState.poll) {
      appState.poll = setInterval(pollQueue, 1200);
    } else if (!shouldPoll && appState.poll) {
      clearInterval(appState.poll);
      appState.poll = null;
    }
  }

  async function pollQueue() {
    if (routeFromHash() !== "/queue") {
      clearInterval(appState.poll);
      appState.poll = null;
      return;
    }
    try {
      await refreshQueueView();
    } catch (error) {
      clearInterval(appState.poll);
      appState.poll = null;
      const list = document.getElementById("queue-list");
      if (list) list.innerHTML = inlineMessage(error.message, true);
    }
  }

  // Library screen
  function formatUpdatedAt(value) {
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "";
    return new Intl.DateTimeFormat("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hourCycle: "h23",
    }).format(date);
  }

  function libraryBookCard(book) {
    const availability = book.has_epub ? "EPUB" : "No EPUB";
    const audiobook = book.has_audiobook ? " · Audiobook" : "";
    const qa = book.qa_summary
      ? qaChips(book.qa_summary, `#/library/${encodeURIComponent(book.book_id)}?flagged=1`)
      : "";

    return `<article class="card library-row">
      <div class="book-cover" aria-hidden="true">ಕ</div>
      <div class="book-info" data-book="${escapeHtml(book.book_id)}">
        <h3>${escapeHtml(book.title)}</h3>
        <p>${escapeHtml(book.author || "Unknown author")} · ` +
          `${formatUpdatedAt(book.updated_at)} · ${book.chapters} chapters · ` +
          `${availability}${audiobook}</p>
        ${book.preview ? `<p class="preview-note">${escapeHtml(coverageText(book))}</p>` : ""}
        ${qa}
      </div>
      <div class="actions">
        ${book.has_epub ? button("Open", "", `data-book-open="${escapeHtml(book.book_id)}"`) : ""}
      </div>
    </article>`;
  }

  function emptyLibraryMarkup() {
    return `<div class="card empty">
      <div class="result-icon" aria-hidden="true">▤</div>
      <h2>Your library is ready</h2>
      <p>Translated books will appear here. Choose an EPUB on the Translate screen
        to get started.</p>
      <a class="button primary" href="#/translate">Choose a book</a>
    </div>`;
  }

  async function renderLibrary() {
    const books = await api("/api/library");
    const bookList = books.length
      ? `<div class="book-list">${books.map(libraryBookCard).join("")}</div>`
      : emptyLibraryMarkup();
    content.innerHTML = pageHeader(
      "Your books",
      "Library",
      "Open a translated book to read it chapter by chapter and review flagged " +
        "paragraphs.",
    ) + bookList;

    document.querySelectorAll("[data-book]").forEach(element => {
      element.addEventListener("click", () => {
        location.hash = `#/library/${encodeURIComponent(element.dataset.book)}`;
      });
    });
    document.querySelectorAll("[data-book] a").forEach(link => {
      link.addEventListener("click", event => event.stopPropagation());
    });
    document.querySelectorAll("[data-book-open]").forEach(element => {
      element.addEventListener("click", async () => {
        try {
          await api(`/api/library/${encodeURIComponent(element.dataset.bookOpen)}/open`, {
            method: "POST",
            body: json({ what: "epub" }),
          });
        } catch (error) {
          content.insertAdjacentHTML("afterbegin", inlineMessage(error.message, true));
        }
      });
    });
  }

  async function renderLibraryBook(route) {
    const parts = route.split("/").filter(Boolean);
    const bookId = parts[1];
    if (parts.length === 2) {
      await renderChapterList(bookId);
    } else {
      await renderReader(bookId, parts[2]);
    }
  }

  async function renderChapterList(bookId) {
    const [books, chapters] = await Promise.all([
      api("/api/library"),
      api(`/api/library/${encodeURIComponent(bookId)}/chapters`),
    ]);
    const book = books.find(item => item.book_id === bookId);
    const flaggedOnly = location.hash.includes("flagged=1");
    const visibleChapters = flaggedOnly
      ? await chaptersWithReviewItems(bookId, chapters)
      : chapters;
    const summary = book?.qa_summary
      ? qaChips(book.qa_summary, `#/library/${encodeURIComponent(bookId)}?flagged=1`)
      : "";
    const chapterLinks = visibleChapters.map(chapter => {
      const index = chapters.indexOf(chapter);
      return chapterLink(bookId, chapter, index, flaggedOnly);
    }).join("");

    content.innerHTML = `${pageHeader(
      "Library",
      book?.title || bookId,
      flaggedOnly
        ? `${visibleChapters.length} chapters with paragraphs to review.`
        : `${chapters.length} chapters · Read in the book’s original order.`,
    )}
    <div class="toolbar">
      <a href="#/library">← Back to Library</a>
      ${summary}
    </div>
    ${epubcheckMarkup(book?.epubcheck)}
    ${flaggedOnly
      ? `<div class="inline-message">Showing chapters with paragraphs marked
          retry or review.</div>`
      : ""}
    <div class="chapter-list">
      ${chapterLinks || `<div class="card empty">No chapters have paragraphs to review.</div>`}
    </div>`;
  }

  async function chaptersWithReviewItems(bookId, chapters) {
    const chapterResults = await Promise.all(chapters.map(async chapter => {
      const data = await api(
        `/api/library/${encodeURIComponent(bookId)}/chapters/` +
          encodeURIComponent(chapter.id),
      );
      const flagged = data.paragraphs.some(paragraph =>
        ["RETRY", "FLAGGED_FOR_REVIEW"].includes(paragraph.qa_status),
      );
      return flagged ? chapter : null;
    }));
    return chapterResults.filter(Boolean);
  }

  function chapterLink(bookId, chapter, index, flaggedOnly) {
    const query = flaggedOnly ? "?flagged=1" : "";
    return `<a class="chapter-link" href="#/library/${encodeURIComponent(bookId)}/` +
      `${encodeURIComponent(chapter.id)}${query}">
      <span class="chapter-num">${String(index + 1).padStart(2, "0")}</span>
      <span>${escapeHtml(chapter.title || chapter.id)}</span>
      <span class="muted chapter-count">${chapter.paragraphs} paragraphs&nbsp; →</span>
    </a>`;
  }

  async function renderReader(bookId, chapterId) {
    const [chapter, chapters] = await Promise.all([
      api(`/api/library/${encodeURIComponent(bookId)}/chapters/` +
        encodeURIComponent(chapterId)),
      api(`/api/library/${encodeURIComponent(bookId)}/chapters`),
    ]);
    const currentIndex = chapters.findIndex(item => item.id === chapterId);
    const flaggedOnly = location.hash.includes("flagged=1");
    const paragraphs = chapter.paragraphs.filter(paragraph =>
      !flaggedOnly || ["RETRY", "FLAGGED_FOR_REVIEW"].includes(paragraph.qa_status),
    );
    const previous = chapters[currentIndex - 1] || null;
    const next = chapters[currentIndex + 1] || null;
    const query = flaggedOnly ? "?flagged=1" : "";

    content.innerHTML = `${pageHeader(
      "Reader",
      chapter.title,
      "English above, Kannada below. Quality notes are available on paragraphs " +
        "checked by QA.",
    )}
    <div class="reader-tools">
      <a href="#/library/${encodeURIComponent(bookId)}${query}">← Chapter list</a>
      <label class="toggle-row">
        <input class="switch" type="checkbox" id="flagged-only"
          ${flaggedOnly ? "checked" : ""}>
        <span>Only show paragraphs to review</span>
      </label>
    </div>
    ${chapterNavigation(bookId, previous, next, currentIndex, chapters.length, query)}
    <div class="card">
      ${paragraphs.length
        ? paragraphs.map(paragraphMarkup).join("")
        : `<div class="empty">No paragraphs match this filter.</div>`}
    </div>
    ${chapterNavigation(bookId, previous, next, currentIndex, chapters.length, query)}`;

    document.getElementById("flagged-only").addEventListener("change", event => {
      const queryString = event.target.checked ? "?flagged=1" : "";
      location.hash = `#/library/${encodeURIComponent(bookId)}/` +
        `${encodeURIComponent(chapterId)}${queryString}`;
    });
  }

  function chapterNavigation(bookId, previous, next, index, total, query) {
    const chapterHref = chapter => `#/library/${encodeURIComponent(bookId)}/` +
      `${encodeURIComponent(chapter.id)}${query}`;
    const previousLink = previous
      ? `<a class="previous" href="${chapterHref(previous)}">← Previous chapter</a>`
      : `<span class="previous muted" aria-disabled="true">← Previous chapter</span>`;
    const nextLink = next
      ? `<a class="next" href="${chapterHref(next)}">Next chapter →</a>`
      : `<span class="next muted" aria-disabled="true">Next chapter →</span>`;

    return `<nav class="chapter-navigation" aria-label="Chapter navigation">
      ${previousLink}
      <span>Chapter ${index + 1} of ${total}</span>
      ${nextLink}
    </nav>`;
  }

  function paragraphMarkup(paragraph) {
    const status = paragraph.qa_status;
    const badgeClass = status === "PASS" ? "pass" : status === "RETRY" ? "retry" : "flag";
    const badgeText = status === "PASS"
      ? "✓ Passed"
      : status === "RETRY"
        ? "↻ Retried"
        : "⚑ Review";
    const qa = status
      ? `<div class="qa-label">
          <span class="chip ${badgeClass}">${badgeText}</span>
          <details>
            <summary class="muted">Quality details</summary>
            <div class="qa-extra">
              Score: ${paragraph.qa_score == null
                ? "—"
                : Number(paragraph.qa_score).toFixed(2)}
              ${paragraph.back_translation
                ? `<br>Back-translation: ${escapeHtml(paragraph.back_translation)}`
                : ""}
            </div>
          </details>
        </div>`
      : "";

    return `<article class="paragraph">
      ${qa}
      <p class="paragraph-en">${escapeHtml(paragraph.en)}</p>
      <p class="paragraph-kn" lang="kn">${escapeHtml(paragraph.kn)}</p>
    </article>`;
  }

  // Settings screen
  function radioChoices(name, choices, selected, localStatus) {
    const items = choices.map(([value, title, description, requirements]) => {
      const unavailable = requirements && requirements.some(module => !localStatus[module]);
      const disabledClass = unavailable ? " disabled" : "";
      const checked = selected === value ? "checked" : "";
      const disabled = unavailable ? "disabled" : "";
      const note = unavailable
        ? ` · Needs local mode (<a href="#/settings?section=local-mode">details</a>)`
        : "";
      return `<label class="choice${disabledClass}">
        <input type="radio" name="${name}" value="${value}" ${checked} ${disabled}>
        <span><strong>${title}</strong><small>${description}${note}</small></span>
      </label>`;
    });
    return `<div class="choices">${items.join("")}</div>`;
  }

  function textField(id, label, value, type = "text") {
    return `<div class="field">
      <label class="field-label" for="${id}">${label}</label>
      <input class="input" id="${id}" type="${type}" value="${escapeHtml(value)}">
    </div>`;
  }

  function advancedSettings(fields) {
    return `<details class="advanced">
      <summary>Advanced settings</summary>
      <div class="advanced-grid">${fields}</div>
    </details>`;
  }

  function providerCard(name, title, description, choices, selected, status, fields) {
    return `<section class="card settings-card">
      <h2>${title}</h2>
      <p class="section-intro">${description}</p>
      ${radioChoices(name, choices, selected, status)}
      ${advancedSettings(fields)}
    </section>`;
  }

  function settingsSectionMarkup(settings, state) {
    const local = state.local_mode;
    const translationChoice = settings.translation.provider === "indictrans2_local"
      ? "local"
      : settings.translation.model === "sarvam-m"
        ? "sarvam"
        : "custom";
    const editorChoice = settings.editor.provider === "anthropic"
      ? "claude"
      : settings.editor.provider === "ollama"
        ? "ollama"
        : "custom";
    const voiceChoice = settings.tts.provider === "parler_local"
      ? "local"
      : settings.tts.provider === "sarvam"
        ? "sarvam"
        : "custom";
    const embeddingChoice = settings.qa.embedding === "local_minilm"
      ? "minilm"
      : settings.qa.embedding_model === "nomic-embed-text"
        ? "ollama"
        : "openai";

    return `<div class="settings-stack">
      ${translationSettingsCard(settings, translationChoice, local)}
      ${editingSettingsCard(settings, editorChoice, local)}
      ${audiobookSettingsCard(settings, voiceChoice, local)}
      ${qualitySettingsCard(settings, embeddingChoice, local)}
      ${advancedSettingsCard(settings, state)}
      ${apiKeysCard(state.key_status)}
      ${localModeCard(local)}
      <div class="actions">
        <button class="button primary" id="save-settings">Save settings</button>
        <span id="save-result" class="saved" aria-live="polite"></span>
      </div>
    </div>`;
  }

  function translationSettingsCard(settings, selected, local) {
    const choices = [
      ["sarvam", "Sarvam (cloud)", "A fast Kannada translation service."],
      ["custom", "Other OpenAI-compatible service", "Use another cloud translation provider."],
      ["local", "IndicTrans2 on this computer", "Runs locally and needs downloaded models.",
        ["ctranslate2", "sentencepiece", "IndicTransToolkit"]],
    ];
    const fields = textField("t-model", "Model", settings.translation.model) +
      textField("t-base", "Base URL", settings.translation.base_url || "") +
      textField("t-key", "API key name", settings.translation.api_key_env || "OPENAI_API_KEY");
    return providerCard("translation-choice", "Translation",
      "Choose what translates English into Kannada.", choices, selected, local, fields);
  }

  function editingSettingsCard(settings, selected, local) {
    const choices = [
      ["claude", "Claude (Anthropic)", "Polishes the translation using an Anthropic model."],
      ["ollama", "Ollama on this computer", "Use a model already running in Ollama."],
      ["custom", "Other OpenAI-compatible service", "Use a compatible cloud or local endpoint."],
    ];
    const fields = textField("e-model", "Model", settings.editor.model) +
      textField("e-base", "Base URL", settings.editor.base_url || "") +
      textField("e-key", "API key name", settings.editor.api_key_env || "ANTHROPIC_API_KEY");
    return providerCard("editing-choice", "Editing",
      "Keep names, voice, and style consistent.", choices, selected, local, fields);
  }

  function audiobookSettingsCard(settings, selected, local) {
    const choices = [
      ["sarvam", "Sarvam Bulbul (cloud)", "Natural Kannada narration."],
      ["custom", "Other OpenAI-compatible service", "Connect another compatible speech service."],
      ["local", "Indic Parler-TTS on this computer", "Runs a downloaded speech model locally.",
        ["torch", "transformers", "parler_tts"]],
    ];
    const fields = textField("a-model", "Model", settings.tts.model) +
      textField("a-voice", "Voice", settings.tts.voice) +
      textField("a-base", "Base URL", settings.tts.base_url || "") +
      textField("a-key", "API key name", settings.tts.api_key_env || "SARVAM_API_KEY");
    return providerCard("audio-choice", "Audiobook voice",
      "Choose a voice provider for narrated audio.", choices, selected, local, fields);
  }

  function qualitySettingsCard(settings, selected, local) {
    const backTranslation = [
      ["llm", "Use the editing model", "Send Kannada back through your editing model."],
      ["indictrans2_local", "IndicTrans2 on this computer", "Runs back-translation locally.",
        ["ctranslate2", "sentencepiece", "IndicTransToolkit"]],
    ];
    const embeddings = [
      ["openai", "OpenAI embeddings", "Use a hosted embeddings service."],
      ["ollama", "Ollama embeddings (nomic-embed-text)", "Use Ollama on this computer."],
      ["minilm", "MiniLM on this computer", "Runs a small local model.",
        ["torch", "transformers"]],
    ];
    const fields = textField("q-model", "Embedding model", settings.qa.embedding_model) +
      textField("q-base", "Base URL", settings.qa.embedding_base_url || "") +
      textField("q-key", "API key name", settings.qa.embedding_api_key_env || "OPENAI_API_KEY") +
      textField("q-pass", "Pass threshold", settings.qa.pass_threshold, "number") +
      textField("q-flag", "Review threshold", settings.qa.flag_threshold, "number");

    return `<section class="card settings-card">
      <h2>Quality check</h2>
      <p class="section-intro">Compare translations with the original to find
        paragraphs that may need attention.</p>
      <label class="toggle-row">
        <input class="switch" type="checkbox" id="qa-enabled"
          ${settings.qa.enabled ? "checked" : ""}>
        <span class="toggle-copy">
          <strong>Turn on quality checks by default</strong>
          <small>You can also change this for each book.</small>
        </span>
      </label>
      <div class="field">
        <span class="field-label">Back-translation</span>
        ${radioChoices("qa-back", backTranslation, settings.qa.back_translation, local)}
      </div>
      <div class="field">
        <span class="field-label">Similarity check</span>
        ${radioChoices("qa-embed", embeddings, selected, local)}
      </div>
      ${advancedSettings(fields)}
    </section>`;
  }

  function advancedSettingsCard(settings, state) {
    const epubcheck = state?.epubcheck || {};
    const epubcheckStatus = epubcheck.available
      ? `<p class="muted">EPUBCheck is installed` +
        `${epubcheck.jar ? ` (${escapeHtml(epubcheck.jar)})` : ""}.</p>`
      : `<p class="muted">EPUBCheck isn't installed. ` +
        `${escapeHtml(epubcheck.how_to || "")}</p>`;
    return `<section class="card settings-card">
      <h2>Advanced</h2>
      <p class="section-intro">Fine-tune how chapters are treated, paragraph
        grouping, skipped sections, and optional cleanup for Project Gutenberg
        books.</p>
      <label class="toggle-row">
        <input class="switch" type="checkbox" id="strip-gutenberg"
          ${settings.strip_gutenberg ? "checked" : ""}>
        <span class="toggle-copy">
          <strong>Remove Project Gutenberg text from the output</strong>
          <small>Recommended before sharing a translation of a Gutenberg book.</small>
        </span>
      </label>
      <label class="toggle-row">
        <input class="switch" type="checkbox" id="epubcheck-enabled"
          ${settings.epubcheck ? "checked" : ""}>
        <span class="toggle-copy">
          <strong>Check the translated EPUB with EPUBCheck when it's installed</strong>
          <small>EPUBCheck validates the finished EPUB and reports errors in the
            Library.</small>
        </span>
      </label>
      ${epubcheckStatus}
      <div class="field">
        <span class="field-label">Chapters are…</span>
        ${radioChoices("chapter-context", [
          ["auto", "Detect automatically",
            "Guess from the chapter titles whether they belong together."],
          ["carry", "Parts of one story (carry context between chapters)",
            "Keep track of names and pronouns across chapter boundaries."],
          ["reset", "Separate stories (start each chapter fresh)",
            "Treat every chapter as an independent story."],
        ], settings.chapter_context, {})}
      </div>
      <div class="advanced-grid">
        ${textField("batch-size", "Batch size", settings.batch_size, "number")}
        ${textField("register", "Writing style", settings.tone_register)}
        <div class="field full-width">
          ${textField("exclude", "Exclude chapter IDs (comma-separated)",
            settings.exclude_ids.join(", "))}
        </div>
      </div>
    </section>`;
  }

  function apiKeysCard(statuses) {
    const rows = Object.entries(statuses).map(([name, isSet]) => `
      <div class="key-row">
        <strong>${escapeHtml(name)}</strong>
        <span class="status-pill${isSet ? " is-set" : ""}" id="status-${name}">
          ${isSet ? "Set" : "Not set"}
        </span>
        <input class="input" type="password" autocomplete="new-password"
          aria-label="${escapeHtml(name)}" id="key-${name}" placeholder="Paste a key to save">
        <button class="button" data-save-key="${name}">Save</button>
        <button class="button quiet danger" data-remove-key="${name}"
          ${isSet ? "" : "disabled"}>Remove</button>
      </div>`).join("");
    return `<section class="card settings-card" id="keys">
      <h2>API keys</h2>
      <p class="section-intro">Keys are stored securely in your user data folder.
        Saved values are never displayed.</p>
      <div id="keys-list">${rows}</div>
      <p id="key-status" class="saved" aria-live="polite"></p>
    </section>`;
  }

  function localModeCard(status) {
    const checklist = Object.entries(status).map(([name, installed]) =>
      `<li>${installed ? "✓" : "○"} ${escapeHtml(name)}` +
        `${installed ? "" : " — not installed"}</li>`,
    ).join("");
    return `<section class="card settings-card" id="local-mode">
      <h2>Local mode</h2>
      <p class="section-intro">Local mode runs translation, speech, or quality
        models on this computer instead of sending text to a cloud provider. It
        needs Python libraries and downloaded model files.</p>
      <div class="local-note">
        <p>From a repository checkout, install local tools and download models:</p>
        <code>git clone &lt;the repository&gt;<br>
          cd &lt;the repository&gt;<br>
          python -m venv .venv<br>
          source .venv/bin/activate&nbsp; # Windows: .venv&#92;Scripts&#92;activate<br>
          pip install -e ".[local,gui]"<br>
          python scripts/download_qa_models.py<br>
          python scripts/download_tts_model.py<br>
          python -m kannada_epub.app</code>
        <p>Then restart the app. Some steps are only needed for the local
          providers you choose.</p>
      </div>
      <p class="muted">Library checklist:</p>
      <ul class="muted">${checklist}</ul>
    </section>`;
  }

  async function renderSettings() {
    const state = await refreshState();
    content.innerHTML = pageHeader(
      "Make it yours",
      "Settings",
      "Choose how each stage works. Your preferences and API keys stay on this " +
        "computer.",
    ) + settingsSectionMarkup(state.settings, state);

    connectSettingsControls(state.settings);
    const requestedSection = new URLSearchParams(location.hash.split("?")[1] || "")
      .get("section");
    if (requestedSection) {
      document.getElementById(requestedSection)?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }

  function connectSettingsControls(settings) {
    document.getElementById("save-settings").addEventListener("click", saveSettings);
    document.querySelectorAll("[data-save-key]").forEach(buttonElement => {
      buttonElement.addEventListener("click", () => saveKey(buttonElement.dataset.saveKey));
    });
    document.querySelectorAll("[data-remove-key]").forEach(buttonElement => {
      buttonElement.addEventListener("click", () =>
        saveKey(buttonElement.dataset.removeKey, true),
      );
    });
    connectProviderChoices(settings);
    connectEmbeddingChoice(settings);
  }

  function connectProviderChoices(settings) {
    const choices = {
      "translation-choice": {
        prefix: "t",
        defaults: {
          sarvam: ["sarvam-m", sarvamBaseUrl, "SARVAM_API_KEY"],
          custom: [settings.translation.model, settings.translation.base_url || "",
            settings.translation.api_key_env || "OPENAI_API_KEY"],
        },
      },
      "editing-choice": {
        prefix: "e",
        defaults: {
          claude: ["claude-haiku-4-5", settings.editor.base_url || "",
            "ANTHROPIC_API_KEY"],
          ollama: ["gemma4:26b", ollamaBaseUrl, ""],
          custom: [settings.editor.model, settings.editor.base_url || "",
            settings.editor.api_key_env || "OPENAI_API_KEY"],
        },
      },
      "audio-choice": {
        prefix: "a",
        defaults: {
          sarvam: ["bulbul:v3", sarvamBaseUrl, "SARVAM_API_KEY"],
          custom: [settings.tts.model, settings.tts.base_url || "",
            settings.tts.api_key_env || "OPENAI_API_KEY"],
          local: [settings.tts.model, settings.tts.base_url || "", ""],
        },
      },
    };

    Object.entries(choices).forEach(([name, config]) => {
      document.querySelectorAll(`input[name="${name}"]`).forEach(input => {
        input.addEventListener("change", () => {
          const values = config.defaults[input.value];
          if (!values) return;
          document.getElementById(`${config.prefix}-model`).value = values[0];
          document.getElementById(`${config.prefix}-base`).value = values[1];
          document.getElementById(`${config.prefix}-key`).value = values[2];
        });
      });
    });
  }

  function connectEmbeddingChoice(settings) {
    document.querySelectorAll('input[name="qa-embed"]').forEach(input => {
      input.addEventListener("change", () => {
        const defaults = {
          ollama: {
            model: "nomic-embed-text",
            base: ollamaBaseUrl,
            key: "",
          },
          minilm: {
            model: "sentence-transformers/all-MiniLM-L6-v2",
            base: "",
            key: "",
          },
          openai: {
            model: "text-embedding-3-small",
            base: settings.qa.embedding_base_url || "",
            key: settings.qa.embedding_api_key_env || "OPENAI_API_KEY",
          },
        }[input.value];
        if (!defaults) return;
        document.getElementById("q-model").value = defaults.model;
        document.getElementById("q-base").value = defaults.base;
        document.getElementById("q-key").value = defaults.key;
      });
    });
  }

  function currentChoice(name, fallback) {
    return document.querySelector(`input[name="${name}"]:checked`)?.value || fallback;
  }

  function fieldValue(id) {
    return document.getElementById(id).value.trim();
  }

  function optionalFieldValue(id) {
    return fieldValue(id) || null;
  }

  function settingsPayload(old) {
    const translationChoice = currentChoice("translation-choice", "sarvam");
    const editorChoice = currentChoice("editing-choice", "claude");
    const voiceChoice = currentChoice("audio-choice", "sarvam");
    const embeddingChoice = currentChoice("qa-embed", "openai");
    const settings = {
      ...old,
      translation: {
        ...old.translation,
        provider: translationChoice === "local" ? "indictrans2_local" : "openai_compatible",
        model: fieldValue("t-model"),
        base_url: optionalFieldValue("t-base"),
        api_key_env: optionalFieldValue("t-key"),
      },
      editor: {
        ...old.editor,
        provider: editorChoice === "claude"
          ? "anthropic"
          : editorChoice === "ollama"
            ? "ollama"
            : "openai_compatible",
        model: fieldValue("e-model"),
        base_url: optionalFieldValue("e-base"),
        api_key_env: optionalFieldValue("e-key"),
      },
      tts: {
        ...old.tts,
        provider: voiceChoice === "local"
          ? "parler_local"
          : voiceChoice === "sarvam"
            ? "sarvam"
            : "openai_compatible",
        model: fieldValue("a-model"),
        voice: fieldValue("a-voice"),
        base_url: optionalFieldValue("a-base"),
        api_key_env: optionalFieldValue("a-key"),
      },
      qa: {
        ...old.qa,
        enabled: document.getElementById("qa-enabled").checked,
        back_translation: currentChoice("qa-back", "llm"),
        embedding: embeddingChoice === "minilm" ? "local_minilm" : "openai_compatible",
        embedding_model: fieldValue("q-model"),
        embedding_base_url: optionalFieldValue("q-base"),
        embedding_api_key_env: optionalFieldValue("q-key"),
        pass_threshold: Number(fieldValue("q-pass")),
        flag_threshold: Number(fieldValue("q-flag")),
      },
      batch_size: Number(fieldValue("batch-size")),
      tone_register: fieldValue("register"),
      exclude_ids: fieldValue("exclude").split(",").map(value => value.trim()).filter(Boolean),
      strip_gutenberg: document.getElementById("strip-gutenberg").checked,
      epubcheck: document.getElementById("epubcheck-enabled").checked,
      chapter_context: currentChoice("chapter-context", old.chapter_context || "auto"),
    };

    if (embeddingChoice === "ollama") {
      settings.qa.embedding_model = "nomic-embed-text";
      settings.qa.embedding_base_url = optionalFieldValue("q-base") || ollamaBaseUrl;
      settings.qa.embedding_api_key_env = "OPENAI_API_KEY";
    }
    return settings;
  }

  async function saveSettings() {
    const status = document.getElementById("save-result");
    status.textContent = "Saving…";
    status.classList.remove("error");
    try {
      await api("/api/settings", {
        method: "PUT",
        body: json(settingsPayload(appState.state.settings)),
      });
      status.textContent = "Saved";
      await refreshState();
    } catch (error) {
      status.textContent = `Could not save: ${error.message}`;
      status.classList.add("error");
    }
  }

  async function saveKey(name, remove = false) {
    const status = document.getElementById("key-status");
    const value = remove ? "" : fieldValue(`key-${name}`);
    if (!remove && !value) {
      status.textContent = "Paste a key to save, or choose Remove to delete the saved key.";
      return;
    }

    try {
      const result = await api("/api/keys", {
        method: "PUT",
        body: json({ [name]: value }),
      });
      const isSet = result.key_status[name];
      const pill = document.getElementById(`status-${name}`);
      pill.textContent = isSet ? "Set" : "Not set";
      pill.classList.toggle("is-set", isSet);
      document.querySelector(`[data-remove-key="${name}"]`).disabled = !isSet;
      document.getElementById(`key-${name}`).value = "";
      status.textContent = remove
        ? "Key removed."
        : "Key saved. The value is not shown again.";
    } catch (error) {
      status.textContent = `Could not save key: ${error.message}`;
    }
  }

  // Start the initial route
  renderRoute().catch(showFatalError);
})();
