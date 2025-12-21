import {
    fetchExperimentValue,
    generateFileGroups,
    handleHashChange,
    activateSingleTab,
    addCollapsibleEventListeners,
    initializeLightbox
} from "./util.js";

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
 * Renders an image row.
 *
 * @param {HTMLElement} row - The row container.
 * @param {string} file - The file name.
 * @param {string} exp1 - Experiment 1 name.
 * @param {string} exp2 - Experiment 2 name (optional).
 */
function renderImageRow(row, file, exp1, exp2) {
    const img1 = document.createElement("img");
    img1.src = `experiments/${exp1}/${file}`;
    img1.alt = `${exp1} - ${file}`;
    img1.className = "render-plot";
    row.appendChild(img1);

    if (exp2) {
        const img2 = document.createElement("img");
        img2.src = `experiments/${exp2}/${file}`;
        img2.alt = `${exp2} - ${file}`;
        img2.className = "render-plot";
        row.appendChild(img2);
    }
}

/**
 * Renders a TSV row.
 *
 * @param {HTMLElement} row - The row container.
 * @param {string} file - The file name.
 * @param {string} exp1 - Experiment 1 name.
 * @param {string} exp2 - Experiment 2 name (optional).
 */
async function renderTSVRow(row, file, exp1, exp2) {
    const tablePromises = [fetchTSV(`experiments/${exp1}/${file}`)];
    if (exp2) tablePromises.push(fetchTSV(`experiments/${exp2}/${file}`));

    const tablesHTML = await Promise.all(tablePromises);
    tablesHTML.forEach((tableHTML) => {
        const tableDiv = document.createElement("div");
        tableDiv.className = "render-table";
        tableDiv.innerHTML = tableHTML;
        row.appendChild(tableDiv);
    });
}

/**
 * Fetches and parses TSV files into HTML tables.
 *
 * @param {string} url - The TSV file URL.
 * @returns {Promise<string>} - HTML table string.
 */
async function fetchTSV(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to fetch ${url}`);

        const tsvText = await response.text();
        const rows = tsvText.split("\n").filter((row) => row.trim() !== "");
        const table = document.createElement("table");

        rows.forEach((row, index) => {
            const tr = document.createElement("tr");
            row.split("\t").forEach((cell) => {
                const td = document.createElement(index === 0 ? "th" : "td");
                td.textContent = cell.trim();
                tr.appendChild(td);
            });
            table.appendChild(tr);
        });

        return table.outerHTML;
    } catch (error) {
        return `<p>Error loading TSV: ${url}</p>`;
    }
}
