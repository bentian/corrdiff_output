document.addEventListener("DOMContentLoaded", async () => {
    // Define the file groups with titles
    const FILE_GROUPS = [
        {
            title: "[all] Metrics",
            files: {
                "overview": [
                    "all/metrics_mean.tsv", "all/metrics_mean.png",
                    "all/monthly_mae.tsv", "all/monthly_mae.png",
                    "all/monthly_rmse.tsv", "all/monthly_rmse.png"
                ],
                "prcp": [
                    "all/pdf_prcp.png", "all/monthly_error_prcp.png"
                ],
                "t2m": [
                    "all/pdf_t2m.png", "all/monthly_error_t2m.png"
                ],
                "u10m": [
                    "all/pdf_u10m.png", "all/monthly_error_u10m.png"
                ],
                "v10m": [
                    "all/pdf_v10m.png", "all/monthly_error_v10m.png"
                ]
            },
        },
        {
            title: "[reg] Metrics",
            files: {
                "overview": [
                    "reg/metrics_mean.tsv", "reg/metrics_mean.png",
                    "reg/monthly_mae.tsv", "reg/monthly_mae.png",
                    "reg/monthly_rmse.tsv", "reg/monthly_rmse.png"
                ],
                "prcp": [
                    "reg/pdf_prcp.png", "reg/monthly_error_prcp.png"
                ],
                "t2m": [
                    "reg/pdf_t2m.png", "reg/monthly_error_t2m.png"
                ],
                "u10m": [
                    "reg/pdf_u10m.png", "reg/monthly_error_u10m.png"
                ],
                "v10m": [
                    "reg/pdf_v10m.png", "reg/monthly_error_v10m.png"
                ]
            },
        },
        {
            title: "[all - reg] Metrics",
            files: [
                "minus_reg/metrics_mean.tsv", "minus_reg/metrics_mean.png",
                "minus_reg/monthly_mae.tsv", "minus_reg/monthly_mae.png",
                "minus_reg/monthly_rmse.tsv", "minus_reg/monthly_rmse.png",
            ],
        },
        {
            title: "Training Loss",
            files: ["training_loss_regression.png", "training_loss_diffusion.png"]
        },
        {
            title: "Config",
            files: ["train_config.tsv", "generate_config.tsv"]
        },

    ];

    // Parse query parameters
    const params = new URLSearchParams(window.location.search);
    const exp1 = params.get("exp1");
    const exp2 = params.get("exp2");
    if (!exp1 && !exp2) {
        document.getElementById("render-output").innerHTML =
            "<p>Error: No experiments selected.</p>";
        return;
    }

    // Set heading
    document.getElementById("render-heading").textContent = exp2
        ? `Comparison: ${exp1} vs. ${exp2}`
        : `Experiment: ${exp1 || exp2}`;

    // Render collapsible sections
    FILE_GROUPS.forEach(({ title, files }) => renderCollapsibleSection(title, files, exp1, exp2));

    // Add collapsible event listeners
    addCollapsibleEventListeners();

    // Initialize lightbox event listeners
    initializeLightbox();

    // Handle the initial hash (if present)
    handleHashChange();

    // Listen for hash changes
    window.addEventListener("hashchange", handleHashChange);
});

/**
 * Handles scrolling and expanding based on hash change.
 */
function handleHashChange() {
    const hash = window.location.hash.substring(1);
    if (!hash) return;

    const targetRow = document.getElementById(hash);
    if (!targetRow) return;

    // Find the collapsible content container
    const targetContent = targetRow.closest(".content");
    if (!targetContent) return;
    targetContent.style.display = "block"; // Expand collapsible section

    const collapsibleHeader = targetContent.previousElementSibling;
    collapsibleHeader?.classList.contains("collapsible") &&
        collapsibleHeader.classList.add("active");

    // Activate corresponding tabContent
    const targetTabContent = targetRow.closest(".tab-content");
    if (targetTabContent) {
        deactivateTabs(targetContent);
        targetTabContent.classList.add("active");
        targetTabContent.tab.classList.add("active");
    }

    targetRow.scrollIntoView({ behavior: "smooth", block: "center" });
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

        tab.addEventListener("click", () => {
            deactivateTabs(content);
            tab.classList.add("active");
            tabContent.classList.add("active");
        });
    });
}

/**
 * Renders a file row inside a collapsible section.
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

/**
 * Adds collapsible toggle functionality.
 */
function addCollapsibleEventListeners() {
    document.querySelectorAll(".collapsible").forEach((collapsible) => {
        collapsible.addEventListener("click", () => {
            collapsible.classList.toggle("active");
            const content = collapsible.nextElementSibling;

            // Expand or collapse the content
            const isVisible = content.style.display === "block";
            content.style.display = isVisible ? "none" : "block";
            if (isVisible) return;

            // Activate the first tab and its content if available
            const [firstTab] = content.querySelectorAll(".tab");
            const [firstTabContent] = content.querySelectorAll(".tab-content");
            deactivateTabs(content);
            if (firstTab) firstTab.classList.add("active");
            if (firstTabContent) firstTabContent.classList.add("active");
        });
    });
}

/**
 * Deactivates all tabs and content within a given container.
 * @param {HTMLElement} content - The content container.
 */
function deactivateTabs(content){
    content.querySelectorAll(".tab, .tab-content").forEach(el => el.classList.remove("active"));
}

/**
 * Initializes lightbox functionality for enlarging images.
 */
function initializeLightbox() {
    const lightbox = document.getElementById("lightbox");
    const lightboxImg = document.getElementById("lightbox-img");
    const closeLightbox = document.getElementById("close-lightbox");

    document.body.addEventListener("click", (event) => {
        if (event.target.classList.contains("render-plot")) {
            lightbox.style.display = "flex";
            lightboxImg.src = event.target.src;
        }
    });

    // Close lightbox when clicking outside the image
    closeLightbox.addEventListener("click", () => {
        lightbox.style.display = "none";
    });

    lightbox.addEventListener("click", (event) => {
        if (event.target === lightbox) {
            lightbox.style.display = "none";
        }
    });
}