import {
    fetchExperimentValue,
    generateFileGroups,
    handleHashChange,
    activateSingleTab,
    addCollapsibleEventListeners,
    initializeLightbox
} from "./util.js";

const FALLBACK_PNG = "./no_plot_table_available.png";

document.addEventListener("DOMContentLoaded", async () => {
    const params = new URLSearchParams(window.location.search);
    const exp1 = params.get("exp1");
    const exp2 = params.get("exp2");
    if (!exp1 && !exp2) {
        document.getElementById("render-output").innerHTML =
            "<p>Error: No experiments selected.</p>";
        return;
    }

    // Render heading with links
    await renderExperimentHeading(exp1, exp2);

    // Render collapsible sections
    generateFileGroups(exp1, exp2).forEach(
        ({ title, files }) => renderCollapsibleSection(title, files, exp1, exp2)
    );

    addCollapsibleEventListeners();
    initializeLightbox();

    handleHashChange();
    window.addEventListener("hashchange", handleHashChange);
});

/**
 * Renders the page heading for a single experiment or a comparison view.
 *
 * The experiment names (`exp1`, `exp2`) are displayed as hyperlinks whose
 * URLs are looked up from `experiments/list.json`. Links open in a new tab.
 *
 * @param {string|null} exp1 - Primary experiment key.
 * @param {string|null} exp2 - Secondary experiment key (optional).
 */
async function renderExperimentHeading(exp1, exp2) {
    const headingEl = document.getElementById("render-heading");
    if (!headingEl) return;

    // Helper to build a hyperlink
    const makeLink = (label, url) => {
        if (!url) return label; // render text only

        const a = document.createElement("a");
        a.textContent = label;
        a.href = url ?? "#";
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        return a;
    };

    headingEl.textContent = ""; // clear existing content

    if (exp1 && exp2) {
        const [url1, url2] = await Promise.all([
            fetchExperimentValue(exp1),
            fetchExperimentValue(exp2),
        ]);

        headingEl.append("Comparison: ");
        headingEl.append(makeLink(exp1, url1));
        headingEl.append(" vs. ");
        headingEl.append(makeLink(exp2, url2));
    } else {
        const exp = exp1 || exp2;
        if (!exp) return;

        const url = await fetchExperimentValue(exp);

        headingEl.append("Experiment: ");
        headingEl.append(makeLink(exp, url));
    }
}

/**
 * Renders a collapsible section for a file group.
 * @param {string} title - The title of the section.
 * @param {string[]} files - The list of files in the section.
 * @param {string} exp1 - Experiment 1 name.
 * @param {string} exp2 - Experiment 2 name (optional).
 */
function renderCollapsibleSection(title, files, exp1, exp2) {
    const renderOutput = document.getElementById("render-output");

    const collapsible = document.createElement("div");
    collapsible.className = "collapsible";
    collapsible.textContent = title;

    const content = document.createElement("div");
    content.className = "content";

    if (Array.isArray(files)) {
        files.forEach((file) => renderFileRow(file, content, exp1, exp2));
    } else {
        renderTabs(content, files, exp1, exp2);
    }

    renderOutput.appendChild(collapsible);
    renderOutput.appendChild(content);
}

/**
 * Renders tabs and their associated content within a given container.
 *
 * @param {HTMLElement} content - The container element where tabs and content will be appended.
 * @param {Object} files - An object where keys are tab names and values are arrays of files.
 * @param {string} exp1 - Experiment 1 name.
 * @param {string} exp2 - Experiment 2 name (optional).
 */
function renderTabs(content, files, exp1, exp2) {
    const tabs = document.createElement("div");
    tabs.className = "tabs";
    content.appendChild(tabs);

    Object.entries(files).forEach(([key, value], index) => {
        const tab = document.createElement("div");
        tab.className = "tab";
        tab.textContent = key;
        tabs.appendChild(tab);

        const tabContent = document.createElement("div");
        tabContent.className = "tab-content";
        tabContent.tab = tab;
        value.forEach((file) => renderFileRow(file, tabContent, exp1, exp2));
        content.appendChild(tabContent);

        // Make "overview" tab active by default
        if (index === 0) {
            tab.classList.add("active");
            tabContent.classList.add("active");
        }

        tab.addEventListener("click", () => activateSingleTab(content, tab, tabContent));
    });
}

/**
 * Renders a file row inside a collapsible section.
 *
 * @param {string} file - The file name.
 * @param {HTMLElement} content - The content container.
 * @param {string} exp1 - Experiment 1 name.
 * @param {string} exp2 - Experiment 2 name (optional).
 */
function renderFileRow(file, content, exp1, exp2) {
    const fileExtension = file.split(".").pop();
    const rowId = file.replace(/[^a-zA-Z0-9]/g, "_");

    const rowContainer = document.createElement("div");
    rowContainer.className = "render-row-container";
    rowContainer.id = rowId;

    const rowTitle = document.createElement("h3");
    rowTitle.className = "render-row-title";
    const link = document.createElement("a");
    link.href = `#${rowId}`;
    link.className = "render-title-link";
    link.textContent = file; // Directly set text content

    rowTitle.appendChild(link);
    rowContainer.appendChild(rowTitle);

    const row = document.createElement("div");
    row.className = "render-row";

    if (fileExtension === "png") {
        renderImageRow(row, file, exp1, exp2);
    } else if (fileExtension === "tsv") {
        renderTSVRow(row, file, exp1, exp2);
    }

    rowContainer.appendChild(row);
    content.appendChild(rowContainer);
}

/**
 * Creates an <img> element with a built-in fallback.
 *
 * If the image fails to load (e.g., missing file or 404), the source is
 * automatically replaced with the shared fallback image while preserving
 * layout size and styling. The fallback is applied only once to prevent
 * infinite error loops.
 *
 * @param {string} src - Image source URL.
 * @param {string} alt - Alternative text for the image.
 * @returns {HTMLImageElement} Configured image element.
 */
function createImage(src, alt) {
    const img = document.createElement("img");
    img.className = "render-plot";
    img.alt = alt;
    img.src = src;

    img.onerror = () => {
        console.warn(`Error loading image: ${src}`);
        img.onerror = null; // prevent loop

        img.className = "fallback-plot";
        img.alt = "Plot not available";
        img.src = FALLBACK_PNG;
    };

    return img;
}

/**
 * Renders one or two experiment plot images into a row.
 *
 * For each experiment, an <img> element is created using the provided file name.
 * If an image fails to load, it automatically falls back to the shared
 * "no_plot_table_available.png" placeholder while preserving layout size.
 *
 * @param {HTMLElement} row  - Container element for the row.
 * @param {string} file      - Plot image file name.
 * @param {string} exp1      - Primary experiment name.
 * @param {string} [exp2]    - Optional secondary experiment name.
 */
function renderImageRow(row, file, exp1, exp2) {
    row.appendChild(createImage(`experiments/${exp1}/${file}`, `${exp1} - ${file}`));

    if (exp2) {
        row.appendChild(createImage(`experiments/${exp2}/${file}`, `${exp2} - ${file}`));
    }
}

/**
 * Renders one or two TSV tables into a row.
 *
 * Each TSV file is fetched and converted into an HTML table. If a TSV file
 * is unavailable, empty, or fails to parse, a fallback image is rendered
 * instead, using the same sizing and styling as plot images to maintain
 * visual alignment between table rows and image rows.
 *
 * @param {HTMLElement} row  - Container element for the row.
 * @param {string} file      - TSV file name.
 * @param {string} exp1      - Primary experiment name.
 * @param {string} [exp2]    - Optional secondary experiment name.
 * @returns {Promise<void>}
 */
async function renderTSVRow(row, file, exp1, exp2) {
    const urls = [`experiments/${exp1}/${file}`];
    if (exp2) urls.push(`experiments/${exp2}/${file}`);

    const results = await Promise.all(urls.map(fetchTSV));

    results.forEach((tableHTML) => {
        if (!tableHTML) {
            const img = document.createElement("img");
            img.className = "fallback-plot";
            img.alt = "Table not available";
            img.src = FALLBACK_PNG;

            row.appendChild(img);
            return;
        }

        const wrapper = document.createElement("div");
        wrapper.className = "render-table";
        wrapper.innerHTML = tableHTML;
        row.appendChild(wrapper);
    });
}

/**
 * Fetches a TSV file and converts it into an HTML table.
 *
 * Returns null if the file cannot be fetched, is empty, or an error occurs.
 * The caller is responsible for handling fallback rendering.
 *
 * @param {string} url - URL of the TSV file.
 * @returns {Promise<string|null>} HTML table markup or null if unavailable.
 */
async function fetchTSV(url) {
    try {
        const res = await fetch(url);
        if (!res.ok) return null;

        const text = await res.text();
        const lines = text.split("\n").filter(Boolean);
        if (!lines.length) return null;

        const table = document.createElement("table");

        lines.forEach((line, i) => {
            const tr = document.createElement("tr");
            line.split("\t").forEach((cell) => {
                const el = document.createElement(i === 0 ? "th" : "td");
                el.textContent = cell.trim();
                tr.appendChild(el);
            });
            table.appendChild(tr);
        });

        return table.outerHTML;
    } catch {
        console.warn(`Error loading TSV: ${url}`);
        return null;
    }
}
