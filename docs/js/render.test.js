/**
 * @vitest-environment jsdom
 *
 * Tests for output/docs/js/render.js
 *
 * render.js runs on DOMContentLoaded, reads URL params, and renders
 * headings, collapsible sections, tabs, images, and TSV tables.
 *
 * Strategy:
 *   - vi.hoisted() + vi.mock("./util.js") to control all util imports
 *   - stub globalThis.fetch for TSV fetching
 *   - set window.location.search before firing DOMContentLoaded
 *   - validate the generated DOM structure
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// Hoisted mocks
// ---------------------------------------------------------------------------
const {
  mockFetchExperimentLink,
  mockGenerateExperimentGroupFiles,
  mockGenerateExperimentFiles,
  mockHandleHashChange,
  mockActivateSingleTab,
  mockAddCollapsibleEventListeners,
  mockInitializeLightbox,
} = vi.hoisted(() => ({
  mockFetchExperimentLink: vi.fn(),
  mockGenerateExperimentGroupFiles: vi.fn(),
  mockGenerateExperimentFiles: vi.fn(),
  mockHandleHashChange: vi.fn(),
  mockActivateSingleTab: vi.fn(),
  mockAddCollapsibleEventListeners: vi.fn(),
  mockInitializeLightbox: vi.fn(),
}));

vi.mock("./util.js", () => ({
  fetchExperimentLink: mockFetchExperimentLink,
  generateExperimentGroupFiles: mockGenerateExperimentGroupFiles,
  generateExperimentFiles: mockGenerateExperimentFiles,
  handleHashChange: mockHandleHashChange,
  activateSingleTab: mockActivateSingleTab,
  addCollapsibleEventListeners: mockAddCollapsibleEventListeners,
  initializeLightbox: mockInitializeLightbox,
}));

// Import render.js once — it registers a DOMContentLoaded listener
import "./render.js";

/** Set up the minimal DOM that render.js expects. */
function setUpRenderPageDOM() {
  document.body.innerHTML = `
    <h1 id="render-heading"></h1>
    <div id="render-output"></div>
    <div id="lightbox" style="display:none">
      <span id="close-lightbox">×</span>
      <img id="lightbox-img" src="" />
    </div>
  `;
}

/**
 * Helper: set window.location.search params, set up DOM, and fire DOMContentLoaded.
 */
async function triggerWithParams(search) {
  const url = new URL(`http://localhost${search}`);
  delete window.location;
  window.location = url;

  setUpRenderPageDOM();

  document.dispatchEvent(new Event("DOMContentLoaded"));
  await new Promise((r) => setTimeout(r, 50));
}

// =========================================================================
// DOMContentLoaded – error state
// =========================================================================
describe("render.js DOMContentLoaded – no params", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("shows error message when no experiments are selected", async () => {
    await triggerWithParams("");

    const output = document.getElementById("render-output");
    expect(output.innerHTML).toContain("No experiments selected");
  });

  it("does not call any rendering functions", async () => {
    await triggerWithParams("");

    expect(mockGenerateExperimentFiles).not.toHaveBeenCalled();
    expect(mockGenerateExperimentGroupFiles).not.toHaveBeenCalled();
    expect(mockAddCollapsibleEventListeners).not.toHaveBeenCalled();
    expect(mockInitializeLightbox).not.toHaveBeenCalled();
  });
});

// =========================================================================
// DOMContentLoaded – experiment group mode
// =========================================================================
describe("render.js DOMContentLoaded – group mode", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentLink.mockResolvedValue("https://example.com/sheet&range=X1");
    mockGenerateExperimentGroupFiles.mockReturnValue([
      { title: "Metrics Mean", files: { pr: ["pr/mean_cmp.png"] } },
    ]);
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("renders the group heading with a link", async () => {
    await triggerWithParams("?group=W");

    const heading = document.getElementById("render-heading");
    expect(heading.textContent).toContain("Experiment Group:");
    expect(heading.textContent).toContain("W");

    const link = heading.querySelector("a");
    expect(link).not.toBeNull();
    expect(link.href).toBe("https://example.com/sheet&range=X1");
    expect(link.target).toBe("_blank");
  });

  it("calls generateExperimentGroupFiles with the group name", async () => {
    await triggerWithParams("?group=BCSD");

    expect(mockGenerateExperimentGroupFiles).toHaveBeenCalledWith("BCSD");
    expect(mockGenerateExperimentFiles).not.toHaveBeenCalled();
  });

  it("renders collapsible sections with correct title", async () => {
    await triggerWithParams("?group=W");

    const collapsibles = document.querySelectorAll(".collapsible");
    expect(collapsibles.length).toBe(1);
    expect(collapsibles[0].textContent).toBe("Metrics Mean");
  });

  it("renders content div after each collapsible", async () => {
    await triggerWithParams("?group=W");

    const content = document.querySelectorAll(".content");
    expect(content.length).toBe(1);
  });

  it("calls addCollapsibleEventListeners and initializeLightbox", async () => {
    await triggerWithParams("?group=W");

    expect(mockAddCollapsibleEventListeners).toHaveBeenCalled();
    expect(mockInitializeLightbox).toHaveBeenCalled();
  });

  it("calls handleHashChange", async () => {
    await triggerWithParams("?group=W");

    expect(mockHandleHashChange).toHaveBeenCalled();
  });

  it("uses 'comparisons' folder for group mode", async () => {
    await triggerWithParams("?group=W");

    const img = document.querySelector("img");
    expect(img.src).toContain("comparisons/W/");
  });
});

// =========================================================================
// DOMContentLoaded – experiment comparison mode
// =========================================================================
describe("render.js DOMContentLoaded – experiment mode", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentLink.mockResolvedValue(null);
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "[all] Metrics", files: ["metrics_mean.png"] },
    ]);
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("renders single experiment heading", async () => {
    await triggerWithParams("?exp1=W-a");

    const heading = document.getElementById("render-heading");
    expect(heading.textContent).toContain("Experiment:");
    expect(heading.textContent).toContain("W-a");
  });

  it("renders comparison heading with two experiments", async () => {
    mockFetchExperimentLink.mockImplementation((key) =>
      Promise.resolve(key === "W-a" ? "https://link/a" : "https://link/b")
    );

    await triggerWithParams("?exp1=W-a&exp2=W-b");

    const heading = document.getElementById("render-heading");
    expect(heading.textContent).toContain("Comparison:");
    expect(heading.textContent).toContain("W-a");
    expect(heading.textContent).toContain("vs.");
    expect(heading.textContent).toContain("W-b");

    const links = heading.querySelectorAll("a");
    expect(links.length).toBe(2);
  });

  it("calls generateExperimentFiles with exp1 and exp2", async () => {
    await triggerWithParams("?exp1=A&exp2=B");

    expect(mockGenerateExperimentFiles).toHaveBeenCalledWith("A", "B");
    expect(mockGenerateExperimentGroupFiles).not.toHaveBeenCalled();
  });

  it("calls generateExperimentFiles with exp1 only when exp2 absent", async () => {
    await triggerWithParams("?exp1=A");

    expect(mockGenerateExperimentFiles).toHaveBeenCalledWith("A", null);
  });

  it("uses 'experiments' folder for experiment mode", async () => {
    await triggerWithParams("?exp1=E1");

    const img = document.querySelector("img");
    expect(img.src).toContain("experiments/E1/");
  });
});

// =========================================================================
// renderCollapsibleSection – flat file list (array)
// =========================================================================
describe("renderCollapsibleSection – flat file array", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentLink.mockResolvedValue(null);
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false });
  });

  afterEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("renders flat file rows (no tabs) for array files", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      {
        title: "Training Loss",
        files: ["training_loss_regression.png", "training_loss_diffusion.png"],
      },
    ]);

    await triggerWithParams("?exp1=X");

    const rows = document.querySelectorAll(".render-row-container");
    expect(rows.length).toBe(2);
    expect(document.querySelectorAll(".tab").length).toBe(0);
  });

  it("creates image elements for .png files", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "Test", files: ["plot.png"] },
    ]);

    await triggerWithParams("?exp1=E1");

    const imgs = document.querySelectorAll(".render-plot");
    expect(imgs.length).toBe(1);
    expect(imgs[0].src).toContain("experiments/E1/plot.png");
    expect(imgs[0].alt).toContain("E1 - plot.png");
  });

  it("creates two images when exp2 is provided", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "Test", files: ["plot.png"] },
    ]);

    await triggerWithParams("?exp1=E1&exp2=E2");

    const imgs = document.querySelectorAll(".render-plot");
    expect(imgs.length).toBe(2);
    expect(imgs[0].src).toContain("experiments/E1/plot.png");
    expect(imgs[1].src).toContain("experiments/E2/plot.png");
  });

  it("row has correct id and anchor link", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "Test", files: ["path/to/file.png"] },
    ]);

    await triggerWithParams("?exp1=E1");

    const row = document.querySelector(".render-row-container");
    expect(row.id).toBe("path_to_file_png");

    const link = row.querySelector("a.render-title-link");
    expect(link.href).toContain("#path_to_file_png");
    expect(link.textContent).toBe("path/to/file.png");
  });

  it("row title is an h3 with correct class", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "Test", files: ["a.png"] },
    ]);

    await triggerWithParams("?exp1=E1");

    const title = document.querySelector(".render-row-title");
    expect(title).not.toBeNull();
    expect(title.tagName).toBe("H3");
  });
});

// =========================================================================
// renderCollapsibleSection – tabbed files (object)
// =========================================================================
describe("renderCollapsibleSection – tabbed files (object)", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentLink.mockResolvedValue(null);
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false });
  });

  afterEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("renders tabs for object-keyed files", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      {
        title: "[all] Metrics",
        files: {
          overview: ["overview.png"],
          pr: ["pr/pdf.png"],
          tas: ["tas/pdf.png"],
        },
      },
    ]);

    await triggerWithParams("?exp1=X");

    const tabs = document.querySelectorAll(".tab");
    expect(tabs.length).toBe(3);
    expect(tabs[0].textContent).toBe("overview");
    expect(tabs[1].textContent).toBe("pr");
    expect(tabs[2].textContent).toBe("tas");
  });

  it("first tab and content are active by default", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      {
        title: "Test",
        files: { a: ["a.png"], b: ["b.png"] },
      },
    ]);

    await triggerWithParams("?exp1=X");

    const tabs = document.querySelectorAll(".tab");
    const tabContents = document.querySelectorAll(".tab-content");

    expect(tabs[0].classList.contains("active")).toBe(true);
    expect(tabContents[0].classList.contains("active")).toBe(true);
    expect(tabs[1].classList.contains("active")).toBe(false);
    expect(tabContents[1].classList.contains("active")).toBe(false);
  });

  it("clicking a tab calls activateSingleTab", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      {
        title: "Test",
        files: { a: ["a.png"], b: ["b.png"] },
      },
    ]);

    await triggerWithParams("?exp1=X");

    document.querySelectorAll(".tab")[1].click();
    expect(mockActivateSingleTab).toHaveBeenCalled();
  });

  it("adds tab-group-start class to pr when not first", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      {
        title: "Test",
        files: { overview: ["o.png"], pr: ["p.png"] },
      },
    ]);

    await triggerWithParams("?exp1=X");

    const prTab = Array.from(document.querySelectorAll(".tab")).find(
      (t) => t.textContent === "pr"
    );
    expect(prTab.classList.contains("tab-group-start")).toBe(true);
  });

  it("does NOT add tab-group-start when pr is first", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      {
        title: "Test",
        files: { pr: ["p.png"], tas: ["t.png"] },
      },
    ]);

    await triggerWithParams("?exp1=X");

    const tabs = document.querySelectorAll(".tab");
    expect(tabs[0].textContent).toBe("pr");
    expect(tabs[0].classList.contains("tab-group-start")).toBe(false);
  });

  it("renders file rows inside each tab-content", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      {
        title: "Test",
        files: { a: ["a1.png", "a2.png"], b: ["b1.png"] },
      },
    ]);

    await triggerWithParams("?exp1=X");

    const tabContents = document.querySelectorAll(".tab-content");
    expect(tabContents[0].querySelectorAll(".render-row-container").length).toBe(2);
    expect(tabContents[1].querySelectorAll(".render-row-container").length).toBe(1);
  });

  it("tabs container has .tabs class", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: { a: ["a.png"] } },
    ]);

    await triggerWithParams("?exp1=X");

    expect(document.querySelector(".tabs")).not.toBeNull();
  });
});

// =========================================================================
// createImage – fallback behaviour
// =========================================================================
describe("createImage fallback", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentLink.mockResolvedValue(null);
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("image has render-plot class and correct src/alt", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["img.png"] },
    ]);

    await triggerWithParams("?exp1=E");

    const img = document.querySelector("img.render-plot");
    expect(img).not.toBeNull();
    expect(img.src).toContain("experiments/E/img.png");
    expect(img.alt).toBe("E - img.png");
  });

  it("switches to fallback on error", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["bad.png"] },
    ]);
    vi.spyOn(console, "warn").mockImplementation(() => { });

    await triggerWithParams("?exp1=E");

    const img = document.querySelector("img.render-plot");
    img.onerror();

    expect(img.className).toBe("fallback-plot");
    expect(img.alt).toBe("Plot not available");
    expect(img.src).toContain("no_plot_table_available.png");
  });

  it("onerror is cleared after fallback to prevent loop", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["bad.png"] },
    ]);
    vi.spyOn(console, "warn").mockImplementation(() => { });

    await triggerWithParams("?exp1=E");

    const img = document.querySelector("img.render-plot");
    img.onerror();

    expect(img.onerror).toBeNull();
  });
});

// =========================================================================
// renderTSVRow / fetchTSV
// =========================================================================
describe("TSV rendering", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentLink.mockResolvedValue(null);
  });

  afterEach(() => {
    document.body.innerHTML = "";
    vi.restoreAllMocks();
  });

  it("renders a TSV file as an HTML table", async () => {
    const tsvContent = "Name\tValue\nA\t1\nB\t2";
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      text: () => Promise.resolve(tsvContent),
    });

    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["data.tsv"] },
    ]);

    await triggerWithParams("?exp1=E");
    await new Promise((r) => setTimeout(r, 100));

    const wrapper = document.querySelector(".render-table");
    expect(wrapper).not.toBeNull();

    const table = wrapper.querySelector("table");
    expect(table).not.toBeNull();

    const headers = table.querySelectorAll("th");
    expect(headers.length).toBe(2);
    expect(headers[0].textContent).toBe("Name");
    expect(headers[1].textContent).toBe("Value");

    const cells = table.querySelectorAll("td");
    expect(cells.length).toBe(4);
    expect(cells[0].textContent).toBe("A");
    expect(cells[1].textContent).toBe("1");
  });

  it("renders fallback image when TSV fetch fails", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({ ok: false });

    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["missing.tsv"] },
    ]);

    await triggerWithParams("?exp1=E");
    await new Promise((r) => setTimeout(r, 100));

    const fallback = document.querySelector("img.fallback-plot");
    expect(fallback).not.toBeNull();
    expect(fallback.alt).toBe("Table not available");
    expect(fallback.src).toContain("no_plot_table_available.png");
  });

  it("renders fallback when TSV content is empty", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      text: () => Promise.resolve(""),
    });

    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["empty.tsv"] },
    ]);

    await triggerWithParams("?exp1=E");
    await new Promise((r) => setTimeout(r, 100));

    expect(document.querySelector("img.fallback-plot")).not.toBeNull();
  });

  it("renders fallback when fetch throws", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("network"));
    vi.spyOn(console, "warn").mockImplementation(() => { });

    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["err.tsv"] },
    ]);

    await triggerWithParams("?exp1=E");
    await new Promise((r) => setTimeout(r, 100));

    expect(document.querySelector("img.fallback-plot")).not.toBeNull();
  });

  it("renders two TSV tables when exp2 is provided", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      text: () => Promise.resolve("H\nR"),
    });

    mockGenerateExperimentFiles.mockReturnValue([
      { title: "T", files: ["data.tsv"] },
    ]);

    await triggerWithParams("?exp1=E1&exp2=E2");
    await new Promise((r) => setTimeout(r, 100));

    expect(document.querySelectorAll(".render-table").length).toBe(2);

    const calls = globalThis.fetch.mock.calls.map(([url]) => url);
    expect(calls).toContain("experiments/E1/data.tsv");
    expect(calls).toContain("experiments/E2/data.tsv");
  });
});

// =========================================================================
// renderHeading edge cases
// =========================================================================
describe("renderHeading edge cases", () => {
  beforeEach(() => {
    vi.resetAllMocks();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("renders experiment name as plain text when link is null", async () => {
    mockFetchExperimentLink.mockResolvedValue(null);
    mockGenerateExperimentFiles.mockReturnValue([]);

    await triggerWithParams("?exp1=NoLink");

    const heading = document.getElementById("render-heading");
    expect(heading.textContent).toContain("NoLink");
    expect(heading.querySelectorAll("a").length).toBe(0);
  });

  it("renders group heading as plain text when link is null", async () => {
    mockFetchExperimentLink.mockResolvedValue(null);
    mockGenerateExperimentGroupFiles.mockReturnValue([]);

    await triggerWithParams("?group=TestGrp");

    const heading = document.getElementById("render-heading");
    expect(heading.textContent).toContain("TestGrp");
    expect(heading.querySelectorAll("a").length).toBe(0);
  });

  it("handles missing render-heading element gracefully", async () => {
    mockFetchExperimentLink.mockResolvedValue("https://link");
    mockGenerateExperimentFiles.mockReturnValue([]);

    const url = new URL("http://localhost?exp1=X");
    delete window.location;
    window.location = url;

    // Set up DOM without heading
    document.body.innerHTML = `<div id="render-output"></div>`;

    document.dispatchEvent(new Event("DOMContentLoaded"));
    await new Promise((r) => setTimeout(r, 50));

    // Should not throw
    expect(document.getElementById("render-heading")).toBeNull();
  });

  it("renders comparison links with correct hrefs", async () => {
    mockFetchExperimentLink.mockImplementation((key) =>
      Promise.resolve(`https://sheet/${key}`)
    );
    mockGenerateExperimentFiles.mockReturnValue([]);

    await triggerWithParams("?exp1=A&exp2=B");

    const links = document.querySelectorAll("#render-heading a");
    expect(links.length).toBe(2);
    expect(links[0].href).toBe("https://sheet/A");
    expect(links[0].rel).toBe("noopener noreferrer");
    expect(links[1].href).toBe("https://sheet/B");
  });

  it("renders links with target=_blank", async () => {
    mockFetchExperimentLink.mockResolvedValue("https://link");
    mockGenerateExperimentFiles.mockReturnValue([]);

    await triggerWithParams("?exp1=X");

    const link = document.querySelector("#render-heading a");
    expect(link.target).toBe("_blank");
    expect(link.rel).toContain("noopener");
  });
});

// =========================================================================
// Multiple collapsible sections
// =========================================================================
describe("multiple collapsible sections", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentLink.mockResolvedValue(null);
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("renders multiple sections in order", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "Section A", files: ["a.png"] },
      { title: "Section B", files: ["b.png"] },
      { title: "Section C", files: ["c.png"] },
    ]);

    await triggerWithParams("?exp1=X");

    const collapsibles = document.querySelectorAll(".collapsible");
    expect(collapsibles.length).toBe(3);
    expect(collapsibles[0].textContent).toBe("Section A");
    expect(collapsibles[1].textContent).toBe("Section B");
    expect(collapsibles[2].textContent).toBe("Section C");
  });

  it("each collapsible has a corresponding content div", async () => {
    mockGenerateExperimentFiles.mockReturnValue([
      { title: "S1", files: ["a.png"] },
      { title: "S2", files: { tab1: ["b.png"] } },
    ]);

    await triggerWithParams("?exp1=X");

    const output = document.getElementById("render-output");
    const children = Array.from(output.children);

    // Pattern: collapsible, content, collapsible, content
    expect(children[0].classList.contains("collapsible")).toBe(true);
    expect(children[1].classList.contains("content")).toBe(true);
    expect(children[2].classList.contains("collapsible")).toBe(true);
    expect(children[3].classList.contains("content")).toBe(true);
  });
});
