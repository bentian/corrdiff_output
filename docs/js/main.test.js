/**
 * @vitest-environment jsdom
 *
 * Tests for output/docs/js/main.js
 *
 * main.js wires up DOMContentLoaded to populate dropdowns and handle form
 * submissions. We mock util.js and trigger DOMContentLoaded to test the
 * integration, then exercise form handlers directly.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

// ---------------------------------------------------------------------------
// Hoisted mocks — vi.mock factories are hoisted above imports, so
// variables referenced inside must also be hoisted via vi.hoisted().
// ---------------------------------------------------------------------------
const { mockFetchExperimentKeys, MOCK_URL_EXP_SHEET } = vi.hoisted(() => ({
  mockFetchExperimentKeys: vi.fn(),
  MOCK_URL_EXP_SHEET: "https://example.com/sheet",
}));

vi.mock("./util.js", () => ({
  fetchExperimentKeys: mockFetchExperimentKeys,
  URL_EXP_SHEET: MOCK_URL_EXP_SHEET,
}));

// Import main.js once — it registers a DOMContentLoaded listener
import "./main.js";

// ---------------------------------------------------------------------------
// Standard DOM fixture
// ---------------------------------------------------------------------------
function setUpMainPageDOM() {
  document.body.innerHTML = `
    <a id="exp-sheet-link" href="#">Sheet</a>

    <select id="exp1">
      <option value="">-- select --</option>
    </select>
    <select id="exp2">
      <option value="">-- select --</option>
    </select>
    <select id="summary-grp"></select>

    <form id="comparison-form">
      <button type="submit">Compare</button>
    </form>
    <form id="summary-form">
      <button type="submit">Summarize</button>
    </form>
  `;
}

/**
 * Fire DOMContentLoaded and wait for async handlers to settle.
 */
async function initWithDOMContentLoaded() {
  document.dispatchEvent(new Event("DOMContentLoaded"));
  await new Promise((r) => setTimeout(r, 50));
}

// =========================================================================
// DOMContentLoaded integration
// =========================================================================
describe("DOMContentLoaded integration", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    setUpMainPageDOM();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("sets the experiment sheet link href and target", async () => {
    mockFetchExperimentKeys.mockResolvedValue([]);

    await initWithDOMContentLoaded();

    const link = document.getElementById("exp-sheet-link");
    expect(link.href).toBe(MOCK_URL_EXP_SHEET);
    expect(link.target).toBe("_blank");
  });

  it("populates exp1/exp2 dropdowns with grouped experiments", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) =>
      isGroup
        ? Promise.resolve(["W", "BCSD"])
        : Promise.resolve(["W-a", "W-b", "BCSD-x"])
    );

    await initWithDOMContentLoaded();

    const exp1 = document.getElementById("exp1");
    const groups = exp1.querySelectorAll("optgroup");
    expect(groups.length).toBe(2);
    expect(groups[0].label).toBe("W");
    expect(groups[1].label).toBe("BCSD");

    const options = exp1.querySelectorAll("optgroup option");
    expect(options.length).toBe(3);
    expect(options[0].value).toBe("W-a");
    expect(options[1].value).toBe("W-b");
    expect(options[2].value).toBe("BCSD-x");
  });

  it("populates exp2 dropdown identically to exp1", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) =>
      isGroup
        ? Promise.resolve(["W"])
        : Promise.resolve(["W-a", "W-b"])
    );

    await initWithDOMContentLoaded();

    const exp2 = document.getElementById("exp2");
    const options = exp2.querySelectorAll("optgroup option");
    expect(options.length).toBe(2);
  });

  it("sets exp1 value to the first option (placeholder)", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) =>
      isGroup
        ? Promise.resolve(["W"])
        : Promise.resolve(["W-a", "W-b"])
    );

    await initWithDOMContentLoaded();

    // The placeholder option (value="") is preserved and is opts[0],
    // so exp1.value ends up as "" per the source code logic.
    const exp1 = document.getElementById("exp1");
    const firstOpt = exp1.querySelector("option");
    expect(exp1.value).toBe(firstOpt.value);
  });

  it("preserves placeholder option in exp1", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) =>
      isGroup
        ? Promise.resolve(["W"])
        : Promise.resolve(["W-a"])
    );

    await initWithDOMContentLoaded();

    const exp1 = document.getElementById("exp1");
    const firstChild = exp1.children[0];
    expect(firstChild.tagName).toBe("OPTION");
    expect(firstChild.value).toBe("");
  });

  it("populates summary-grp dropdown with full names", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) =>
      isGroup
        ? Promise.resolve(["CropW", "BCSD", "DM"])
        : Promise.resolve([])
    );

    await initWithDOMContentLoaded();

    const grp = document.getElementById("summary-grp");
    const options = grp.querySelectorAll("option");
    expect(options.length).toBe(3);
    expect(options[0].textContent).toContain("SSP scenarios");
    expect(options[1].textContent).toContain("bias-corrected");
    expect(options[2].textContent).toContain("input domains");
  });

  it("selects the first group option by default", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) =>
      isGroup
        ? Promise.resolve(["W"])
        : Promise.resolve([])
    );

    await initWithDOMContentLoaded();

    const grp = document.getElementById("summary-grp");
    expect(grp.value).toBeTruthy();
  });

  it("shows error in exp dropdowns when fetchExperimentKeys rejects", async () => {
    mockFetchExperimentKeys.mockRejectedValue(new Error("Network error"));
    vi.spyOn(console, "error").mockImplementation(() => { });

    await initWithDOMContentLoaded();

    expect(document.getElementById("exp1").innerHTML).toContain("Error loading");
    expect(document.getElementById("exp2").innerHTML).toContain("Error loading");
  });

  it("shows error in summary-grp when group fetch rejects", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) => {
      if (isGroup) return Promise.reject(new Error("fail"));
      return Promise.resolve(["X-1"]);
    });
    vi.spyOn(console, "error").mockImplementation(() => { });

    await initWithDOMContentLoaded();

    expect(document.getElementById("summary-grp").innerHTML).toContain(
      "Error loading"
    );
  });

  it("handles empty experiment list gracefully", async () => {
    mockFetchExperimentKeys.mockResolvedValue([]);
    const spy = vi.spyOn(console, "error").mockImplementation(() => { });

    await initWithDOMContentLoaded();

    expect(spy).toHaveBeenCalled();
    spy.mockRestore();
  });

  it("handles empty group list gracefully", async () => {
    mockFetchExperimentKeys.mockImplementation((isGroup) =>
      isGroup ? Promise.resolve([]) : Promise.resolve(["X"])
    );
    const spy = vi.spyOn(console, "error").mockImplementation(() => { });

    await initWithDOMContentLoaded();

    expect(spy).toHaveBeenCalledWith(
      expect.stringContaining("No experiment groups")
    );
    spy.mockRestore();
  });
});

// =========================================================================
// handleComparisonSubmit
// =========================================================================
describe("handleComparisonSubmit", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentKeys.mockResolvedValue([]);
    setUpMainPageDOM();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("navigates to render.html with exp1 only", async () => {
    await initWithDOMContentLoaded();

    const exp1 = document.getElementById("exp1");
    const opt = document.createElement("option");
    opt.value = "TestExp";
    exp1.appendChild(opt);
    exp1.value = "TestExp";
    document.getElementById("exp2").value = "";

    const origLocation = window.location;
    delete window.location;
    window.location = { ...origLocation, href: "" };
    const hrefSpy = vi.spyOn(window.location, "href", "set");

    document
      .getElementById("comparison-form")
      .dispatchEvent(new Event("submit", { cancelable: true }));

    expect(hrefSpy).toHaveBeenCalledWith("render.html?exp1=TestExp");
    window.location = origLocation;
  });

  it("navigates to render.html with exp1 and exp2", async () => {
    await initWithDOMContentLoaded();

    const exp1 = document.getElementById("exp1");
    const exp2 = document.getElementById("exp2");

    [exp1, exp2].forEach((el, i) => {
      const opt = document.createElement("option");
      opt.value = ["A", "B"][i];
      el.appendChild(opt);
      el.value = ["A", "B"][i];
    });

    const origLocation = window.location;
    delete window.location;
    window.location = { ...origLocation, href: "" };
    const hrefSpy = vi.spyOn(window.location, "href", "set");

    document
      .getElementById("comparison-form")
      .dispatchEvent(new Event("submit", { cancelable: true }));

    expect(hrefSpy).toHaveBeenCalledWith("render.html?exp1=A&exp2=B");
    window.location = origLocation;
  });

  it("shows alert when exp1 is not selected", async () => {
    await initWithDOMContentLoaded();

    const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => { });
    document.getElementById("exp1").value = "";

    document
      .getElementById("comparison-form")
      .dispatchEvent(new Event("submit", { cancelable: true }));

    expect(alertSpy).toHaveBeenCalledWith(
      "Please select at least one experiment to show."
    );
    alertSpy.mockRestore();
  });
});

// =========================================================================
// handleSummarySubmit
// =========================================================================
describe("handleSummarySubmit", () => {
  beforeEach(() => {
    vi.resetAllMocks();
    mockFetchExperimentKeys.mockResolvedValue([]);
    setUpMainPageDOM();
  });

  afterEach(() => {
    document.body.innerHTML = "";
  });

  it("navigates to render.html with group prefix from full name", async () => {
    await initWithDOMContentLoaded();

    const grp = document.getElementById("summary-grp");
    grp.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "CropW* (SSP scenarios cropped)";
    grp.appendChild(opt);
    grp.value = "CropW* (SSP scenarios cropped)";

    const origLocation = window.location;
    delete window.location;
    window.location = { ...origLocation, href: "" };
    const hrefSpy = vi.spyOn(window.location, "href", "set");

    document
      .getElementById("summary-form")
      .dispatchEvent(new Event("submit", { cancelable: true }));

    expect(hrefSpy).toHaveBeenCalledWith(
      "render.html?group=CropW#prcp_mean_cmp_png"
    );
    window.location = origLocation;
  });

  it("extracts 'W' prefix from 'W* (SSP scenarios)'", async () => {
    await initWithDOMContentLoaded();

    const grp = document.getElementById("summary-grp");
    grp.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "W* (SSP scenarios)";
    grp.appendChild(opt);
    grp.value = "W* (SSP scenarios)";

    const origLocation = window.location;
    delete window.location;
    window.location = { ...origLocation, href: "" };
    const hrefSpy = vi.spyOn(window.location, "href", "set");

    document
      .getElementById("summary-form")
      .dispatchEvent(new Event("submit", { cancelable: true }));

    expect(hrefSpy).toHaveBeenCalledWith(
      "render.html?group=W#prcp_mean_cmp_png"
    );
    window.location = origLocation;
  });

  it("extracts 'BCSD' prefix from 'BCSD-* (bias-corrected...)'", async () => {
    await initWithDOMContentLoaded();

    const grp = document.getElementById("summary-grp");
    grp.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "BCSD-* (bias-corrected spatial disaggregation)";
    grp.appendChild(opt);
    grp.value = "BCSD-* (bias-corrected spatial disaggregation)";

    const origLocation = window.location;
    delete window.location;
    window.location = { ...origLocation, href: "" };
    const hrefSpy = vi.spyOn(window.location, "href", "set");

    document
      .getElementById("summary-form")
      .dispatchEvent(new Event("submit", { cancelable: true }));

    expect(hrefSpy).toHaveBeenCalledWith(
      "render.html?group=BCSD#prcp_mean_cmp_png"
    );
    window.location = origLocation;
  });

  it("shows alert when no group is selected", async () => {
    await initWithDOMContentLoaded();

    const alertSpy = vi.spyOn(window, "alert").mockImplementation(() => { });
    document.getElementById("summary-grp").value = "";

    document
      .getElementById("summary-form")
      .dispatchEvent(new Event("submit", { cancelable: true }));

    expect(alertSpy).toHaveBeenCalledWith(
      "Please select an experiment group to summarize."
    );
    alertSpy.mockRestore();
  });
});
