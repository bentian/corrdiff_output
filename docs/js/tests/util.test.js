/**
 * @vitest-environment jsdom
 *
 * Tests for output/docs/js/util.js
 *
 * Pure-logic functions are tested directly; DOM/fetch-dependent functions
 * are tested with minimal jsdom stubs.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// Module-level mocks must be set up BEFORE dynamic import so the module
// picks up globalThis.fetch during evaluation.
// ---------------------------------------------------------------------------

/** Helper: reset module caches by re-importing a fresh copy. */
async function loadFreshModule() {
  // Vitest caches modules; bust it by appending a unique query param.
  const id = `../util.js?t=${Date.now()}-${Math.random()}`;
  return await import(id);
}

// =========================================================================
// generateExperimentGroupFiles
// =========================================================================
describe("generateExperimentGroupFiles", () => {
  let generateExperimentGroupFiles;

  beforeEach(async () => {
    ({ generateExperimentGroupFiles } = await loadFreshModule());
  });

  it("returns an array of file groups", () => {
    const result = generateExperimentGroupFiles("CorrDiff");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeGreaterThanOrEqual(1);
    result.forEach((group) => {
      expect(group).toHaveProperty("title");
      expect(group).toHaveProperty("files");
    });
  });

  it("includes decadal trends for non-DM/LR groups", () => {
    const result = generateExperimentGroupFiles("CorrDiff");
    expect(result[0].title).toContain("Decadal Trends");
    // Each variable should have a nyear_cmp.png file
    for (const files of Object.values(result[0].files)) {
      expect(files.some((f) => f.includes("nyear_cmp.png"))).toBe(true);
    }
  });

  it("excludes decadal trends for DM group", () => {
    const result = generateExperimentGroupFiles("DM");
    expect(result[0].title).not.toContain("Decadal Trends");
    for (const files of Object.values(result[0].files)) {
      expect(files.every((f) => !f.includes("nyear_cmp.png"))).toBe(true);
    }
  });

  it("excludes decadal trends for LR group", () => {
    const result = generateExperimentGroupFiles("LR");
    expect(result[0].title).not.toContain("Decadal Trends");
  });

  it("limits BCSD to pr and tas variables only", () => {
    const result = generateExperimentGroupFiles("BCSD");
    const vars = Object.keys(result[0].files);
    expect(vars).toEqual(["pr", "tas"]);
  });

  it("includes four variables for non-BCSD groups", () => {
    const result = generateExperimentGroupFiles("CorrDiff");
    const vars = Object.keys(result[0].files);
    expect(vars).toEqual(["pr", "tas", "uas", "vas"]);
  });

  it("includes mean_cmp.png for every variable", () => {
    const result = generateExperimentGroupFiles("CorrDiff");
    for (const [varName, files] of Object.entries(result[0].files)) {
      expect(files).toContain(`${varName}/mean_cmp.png`);
    }
  });
});

// =========================================================================
// generateExperimentFiles
// =========================================================================
describe("generateExperimentFiles", () => {
  let generateExperimentFiles;

  beforeEach(async () => {
    ({ generateExperimentFiles } = await loadFreshModule());
  });

  // --- basic structure ---------------------------------------------------

  it("returns an array of file groups", () => {
    const result = generateExperimentFiles("CropW1");
    expect(Array.isArray(result)).toBe(true);
    result.forEach((g) => {
      expect(g).toHaveProperty("title");
      expect(g).toHaveProperty("files");
    });
  });

  // --- non-BCSD experiments -----------------------------------------------

  it("produces 'all', 'reg', minus_reg, training, and config groups for non-BCSD", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const titles = result.map((g) => g.title);
    expect(titles).toContain("[all] Metrics");
    expect(titles).toContain("[reg] Metrics");
    expect(titles).toContain("[all - reg] Metrics");
    expect(titles).toContain("Training Loss");
    expect(titles).toContain("Config");
  });

  it("includes four variables for non-BCSD experiments", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files).toHaveProperty("pr");
    expect(allGroup.files).toHaveProperty("tas");
    expect(allGroup.files).toHaveProperty("uas");
    expect(allGroup.files).toHaveProperty("vas");
  });

  // --- BCSD experiments ---------------------------------------------------

  it("limits BCSD to pr and tas and only 'all' prefix", () => {
    const result = generateExperimentFiles("BCSD-a", "BCSD-b");
    const titles = result.map((g) => g.title);
    expect(titles).toContain("[all] Metrics");
    expect(titles).not.toContain("[reg] Metrics");
    expect(titles).not.toContain("Training Loss");

    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files).toHaveProperty("pr");
    expect(allGroup.files).toHaveProperty("tas");
    expect(allGroup.files).not.toHaveProperty("uas");
  });

  it("includes decadal metrics for BCSD experiments", () => {
    const result = generateExperimentFiles("BCSD-a", "BCSD-b");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files).toHaveProperty("decadal");
  });

  it("does not include decadal metrics for non-BCSD experiments", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files).not.toHaveProperty("decadal");
  });

  // --- p90_by_nyear (only for BCSD with pr/tas) ---------------------------

  it("includes p90_by_nyear for BCSD pr/tas variables", () => {
    const result = generateExperimentFiles("BCSD-a");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files.pr.some((f) => f.includes("p90_by_nyear.png"))).toBe(true);
    expect(allGroup.files.tas.some((f) => f.includes("p90_by_nyear.png"))).toBe(true);
  });

  it("does not include p90_by_nyear for non-BCSD experiments", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files.pr.every((f) => !f.includes("p90_by_nyear.png"))).toBe(true);
  });

  // --- ensemble-specific files (currently hardcoded off) -------------------

  it("does not include spread_vs_rmse (hasMultipleEnsembles is false)", () => {
    const result = generateExperimentFiles("CropW1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    for (const files of Object.values(allGroup.files)) {
      if (Array.isArray(files)) {
        expect(files.every((f) => !f.includes("spread_vs_rmse"))).toBe(true);
      }
    }
  });

  it("does not include metrics_v_ensembles (has64Ensembles is false)", () => {
    const result = generateExperimentFiles("CropW1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    for (const files of Object.values(allGroup.files)) {
      if (Array.isArray(files)) {
        expect(files.every((f) => !f.includes("metrics_v_ensembles"))).toBe(true);
      }
    }
  });

  // --- single vs dual experiment ------------------------------------------

  it("handles a single experiment (exp2 undefined)", () => {
    const result = generateExperimentFiles("W1");
    expect(result.length).toBeGreaterThan(0);
  });

  it("handles two experiments", () => {
    const result = generateExperimentFiles("W1", "W2");
    expect(result.length).toBeGreaterThan(0);
  });

  // --- overview group structure -------------------------------------------

  it("overview group contains overview and monthly keys", () => {
    const result = generateExperimentFiles("W1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files).toHaveProperty("overview");
    expect(allGroup.files).toHaveProperty("monthly");
    expect(allGroup.files.overview.some((f) => f.includes("metrics_mean"))).toBe(true);
  });

  // --- metrics list -------------------------------------------------------

  it("generates files for all six metrics (rmse, mae, corr, crps, std_dev, ssr)", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");

    const metrics = ["rmse", "mae", "corr", "crps", "std_dev", "ssr"];
    for (const metric of metrics) {
      // Monthly files should contain each metric as both .tsv and .png
      expect(allGroup.files.monthly.some((f) => f.includes(`/${metric}.tsv`))).toBe(true);
      expect(allGroup.files.monthly.some((f) => f.includes(`/${metric}.png`))).toBe(true);
    }
  });

  it("overview includes metrics_mean as both .tsv and .png", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files.overview).toContain("all/overview/metrics_mean.tsv");
    expect(allGroup.files.overview).toContain("all/overview/metrics_mean.png");
  });

  // --- variable files list ------------------------------------------------

  it("includes rank histogram files in variable file lists", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    expect(allGroup.files.pr.some((f) => f.includes("rank_histogram.png"))).toBe(true);
    expect(allGroup.files.pr.some((f) => f.includes("monthly_rank_histogram.png"))).toBe(true);
  });

  it("includes core variable files (pdf, monthly_error, cnt_rmse, top_samples)", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");

    const expectedPatterns = [
      "pdf.png",
      "monthly_error.png",
      "cnt_rmse.png",
      "top_samples_rmse.png",
      "cnt_mae.png",
      "top_samples_mae.png",
    ];

    for (const pattern of expectedPatterns) {
      expect(allGroup.files.pr.some((f) => f.includes(pattern))).toBe(true);
    }
  });

  // --- training loss & config for non-BCSD --------------------------------

  it("training loss group has expected files", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const training = result.find((g) => g.title === "Training Loss");
    expect(training.files).toContain("training_loss_regression.png");
    expect(training.files).toContain("training_loss_diffusion.png");
  });

  it("config group has expected files", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const config = result.find((g) => g.title === "Config");
    expect(config.files).toContain("train_config.tsv");
    expect(config.files).toContain("generate_config.tsv");
  });

  // --- [all - reg] minus_reg structure ------------------------------------

  it("minus_reg group has overview and monthly keys", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const minusReg = result.find((g) => g.title === "[all - reg] Metrics");
    expect(minusReg.files).toHaveProperty("overview");
    expect(minusReg.files).toHaveProperty("monthly");
  });

  it("minus_reg group does not have decadal for non-BCSD", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const minusReg = result.find((g) => g.title === "[all - reg] Metrics");
    expect(minusReg.files).not.toHaveProperty("decadal");
  });

  // --- [reg] prefix structure ---------------------------------------------

  it("reg group includes same variables as all group", () => {
    const result = generateExperimentFiles("CorrDiff1");
    const allGroup = result.find((g) => g.title === "[all] Metrics");
    const regGroup = result.find((g) => g.title === "[reg] Metrics");

    const allVars = Object.keys(allGroup.files).filter(
      (k) => !["overview", "monthly", "decadal"].includes(k)
    );
    const regVars = Object.keys(regGroup.files).filter(
      (k) => !["overview", "monthly", "decadal"].includes(k)
    );
    expect(regVars).toEqual(allVars);
  });
});

// =========================================================================
// fetchExperimentKeys / fetchExperimentLink / loadExperimentMap
// =========================================================================
describe("fetch helpers (loadExperimentMap, fetchExperimentKeys, fetchExperimentLink)", () => {
  let fetchExperimentKeys, fetchExperimentLink, URL_EXP_SHEET;

  const fakeExperiments = { expA: "A1", expB: "B2" };
  const fakeGroups = { groupX: "X1" };

  beforeEach(async () => {
    // Stub global fetch
    globalThis.fetch = vi.fn((url) => {
      const body = url.includes("comparisons") ? fakeGroups : fakeExperiments;
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(body),
      });
    });

    ({ fetchExperimentKeys, fetchExperimentLink, URL_EXP_SHEET } =
      await loadFreshModule());
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  // --- fetchExperimentKeys ------------------------------------------------

  it("returns experiment keys", async () => {
    const keys = await fetchExperimentKeys();
    expect(keys).toEqual(["expA", "expB"]);
  });

  it("returns group keys when isExperimentGroup=true", async () => {
    const keys = await fetchExperimentKeys(true);
    expect(keys).toEqual(["groupX"]);
  });

  it("caches results (fetch called once per type)", async () => {
    await fetchExperimentKeys();
    await fetchExperimentKeys();
    // fetch for experiments/list.json should be called only once
    const experimentCalls = globalThis.fetch.mock.calls.filter(
      ([url]) => url.includes("experiments")
    );
    expect(experimentCalls.length).toBe(1);
  });

  // --- fetchExperimentLink ------------------------------------------------

  it("returns a full link for a known key", async () => {
    const link = await fetchExperimentLink("expA");
    expect(link).toBe(`${URL_EXP_SHEET}&range=A1`);
  });

  it("returns null for an unknown key", async () => {
    const link = await fetchExperimentLink("nonexistent");
    expect(link).toBeNull();
  });

  it("returns link from group map when isExperimentGroup=true", async () => {
    const link = await fetchExperimentLink("groupX", true);
    expect(link).toBe(`${URL_EXP_SHEET}&range=X1`);
  });

  // --- error handling -----------------------------------------------------

  it("throws when fetch fails", async () => {
    globalThis.fetch = vi.fn(() =>
      Promise.resolve({ ok: false, statusText: "Not Found" })
    );
    const { fetchExperimentKeys: freshKeys } = await loadFreshModule();
    await expect(freshKeys()).rejects.toThrow("Fetch failed: Not Found");
  });
});

// =========================================================================
// activateSingleTab
// =========================================================================
describe("activateSingleTab", () => {
  let activateSingleTab;

  beforeEach(async () => {
    ({ activateSingleTab } = await loadFreshModule());
  });

  it("activates the specified tab and deactivates others", () => {
    document.body.innerHTML = `
      <div id="content">
        <div class="tab active" id="tab1">Tab1</div>
        <div class="tab" id="tab2">Tab2</div>
        <div class="tab-content active" id="tc1">Content1</div>
        <div class="tab-content" id="tc2">Content2</div>
      </div>
    `;

    const content = document.getElementById("content");
    const tab2 = document.getElementById("tab2");
    const tc2 = document.getElementById("tc2");

    activateSingleTab(content, tab2, tc2);

    expect(tab2.classList.contains("active")).toBe(true);
    expect(tc2.classList.contains("active")).toBe(true);
    // previously active ones should be deactivated
    expect(document.getElementById("tab1").classList.contains("active")).toBe(false);
    expect(document.getElementById("tc1").classList.contains("active")).toBe(false);
  });

  it("handles null tab and tabContent gracefully", () => {
    document.body.innerHTML = `<div id="content"></div>`;
    const content = document.getElementById("content");

    // Should not throw
    expect(() => activateSingleTab(content, null, null)).not.toThrow();
  });
});

// =========================================================================
// addCollapsibleEventListeners
// =========================================================================
describe("addCollapsibleEventListeners", () => {
  let addCollapsibleEventListeners;

  beforeEach(async () => {
    ({ addCollapsibleEventListeners } = await loadFreshModule());
  });

  it("toggles content display on click", () => {
    document.body.innerHTML = `
      <div class="collapsible">Header</div>
      <div class="content" style="display:none">
        <div class="tab">Tab1</div>
        <div class="tab-content">TC1</div>
      </div>
    `;

    addCollapsibleEventListeners();
    const collapsible = document.querySelector(".collapsible");
    const content = document.querySelector(".content");

    // First click → open
    collapsible.click();
    expect(content.style.display).toBe("block");
    expect(collapsible.classList.contains("active")).toBe(true);

    // Second click → close
    collapsible.click();
    expect(content.style.display).toBe("none");
    expect(collapsible.classList.contains("active")).toBe(false);
  });

  it("activates the first tab when opening", () => {
    document.body.innerHTML = `
      <div class="collapsible">Header</div>
      <div class="content" style="display:none">
        <div class="tab" id="t1">Tab1</div>
        <div class="tab" id="t2">Tab2</div>
        <div class="tab-content" id="tc1">TC1</div>
        <div class="tab-content" id="tc2">TC2</div>
      </div>
    `;

    addCollapsibleEventListeners();
    document.querySelector(".collapsible").click();

    expect(document.getElementById("t1").classList.contains("active")).toBe(true);
    expect(document.getElementById("tc1").classList.contains("active")).toBe(true);
  });
});

// =========================================================================
// handleHashChange
// =========================================================================
describe("handleHashChange", () => {
  let handleHashChange;

  beforeEach(async () => {
    ({ handleHashChange } = await loadFreshModule());
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("does nothing when hash is empty", () => {
    window.location.hash = "";
    expect(() => handleHashChange()).not.toThrow();
  });

  it("does nothing when target element is missing", () => {
    window.location.hash = "#nonexistent";
    expect(() => handleHashChange()).not.toThrow();
  });

  it("opens the parent content and activates collapsible header", () => {
    document.body.innerHTML = `
      <div class="collapsible">Header</div>
      <div class="content" style="display:none">
        <div id="target-row">Target</div>
      </div>
    `;

    // jsdom doesn't implement scrollIntoView
    document.getElementById("target-row").scrollIntoView = vi.fn();

    window.location.hash = "#target-row";
    handleHashChange();

    const content = document.querySelector(".content");
    expect(content.style.display).toBe("block");
    expect(document.querySelector(".collapsible").classList.contains("active")).toBe(true);
  });

  it("activates the tab containing the target element", () => {
    document.body.innerHTML = `
      <div class="collapsible">Header</div>
      <div class="content" style="display:none">
        <div class="tab active" id="tab1">Tab1</div>
        <div class="tab" id="tab2">Tab2</div>
        <div class="tab-content active" id="tc1">TC1</div>
        <div class="tab-content" id="tc2">
          <div id="target-in-tab2">Target</div>
        </div>
      </div>
    `;

    // Attach tab reference (render.js sets tabContent.tab during creation)
    const tc2 = document.getElementById("tc2");
    tc2.tab = document.getElementById("tab2");

    document.getElementById("target-in-tab2").scrollIntoView = vi.fn();

    window.location.hash = "#target-in-tab2";
    handleHashChange();

    // Tab2 should be activated
    expect(document.getElementById("tab2").classList.contains("active")).toBe(true);
    expect(tc2.classList.contains("active")).toBe(true);
    // Tab1 should be deactivated
    expect(document.getElementById("tab1").classList.contains("active")).toBe(false);
    expect(document.getElementById("tc1").classList.contains("active")).toBe(false);
  });

  it("does not throw when target is not inside a .content element", () => {
    document.body.innerHTML = `<div id="standalone-row">Target</div>`;
    document.getElementById("standalone-row").scrollIntoView = vi.fn();

    window.location.hash = "#standalone-row";
    expect(() => handleHashChange()).not.toThrow();
  });

  it("does not throw when collapsible header is missing", () => {
    document.body.innerHTML = `
      <div class="content" style="display:none">
        <div id="orphan-row">Target</div>
      </div>
    `;
    document.getElementById("orphan-row").scrollIntoView = vi.fn();

    window.location.hash = "#orphan-row";
    expect(() => handleHashChange()).not.toThrow();
  });
});

// =========================================================================
// initializeLightbox
// =========================================================================
describe("initializeLightbox", () => {
  let initializeLightbox;

  beforeEach(async () => {
    ({ initializeLightbox } = await loadFreshModule());

    document.body.innerHTML = `
      <div id="lightbox" style="display:none">
        <span id="close-lightbox">x</span>
        <img id="lightbox-img" src="" />
      </div>
      <img class="render-plot" src="test-image.png" />
    `;
  });

  it("opens lightbox when a render-plot image is clicked", () => {
    initializeLightbox();

    const plot = document.querySelector(".render-plot");
    plot.click();

    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");

    expect(lightbox.style.display).toBe("flex");
    expect(lightboxImg.src).toContain("test-image.png");
  });

  it("closes lightbox when backdrop is clicked", () => {
    initializeLightbox();

    const lightbox = document.getElementById("lightbox");
    lightbox.style.display = "flex";

    // Click on the lightbox backdrop itself
    lightbox.click();
    expect(lightbox.style.display).toBe("none");
  });

  it("closes lightbox when close button is clicked", () => {
    initializeLightbox();

    const lightbox = document.getElementById("lightbox");
    lightbox.style.display = "flex";

    document.getElementById("close-lightbox").click();
    expect(lightbox.style.display).toBe("none");
  });

  it("does not close lightbox when clicking the image itself", () => {
    initializeLightbox();

    const lightbox = document.getElementById("lightbox");
    lightbox.style.display = "flex";

    document.getElementById("lightbox-img").click();
    // Should remain open — click target is neither lightbox nor close button
    expect(lightbox.style.display).toBe("flex");
  });
});

// =========================================================================
// URL_EXP_SHEET constant
// =========================================================================
describe("URL_EXP_SHEET", () => {
  it("exports the expected Google Sheets URL", async () => {
    const { URL_EXP_SHEET } = await loadFreshModule();
    expect(URL_EXP_SHEET).toMatch(/^https:\/\/docs\.google\.com\/spreadsheets/);
  });
});
