document.addEventListener("DOMContentLoaded", () => {
    // Define the file groups with titles
    const FILE_GROUPS = [
        {
            title: "Generate Config",
            files: ["generate_overrides.tsv"],
        },        
        {
            title: "[all] Metrics Mean",
            files: [
                "all-metrics_mean.tsv", "all-metrics_mean.png",
                "all-monthly_mae.tsv", "all-monthly_mae.png",
                "all-monthly_rmse.tsv", "all-monthly_rmse.png",
            ],
        },
        {
            title: "[all] Probability Density Function",
            files: [
                "all-pdf_prcp.png",
            ],
        },
        {
            title: "[all] Monthly Errors",
            files: [
                "all-monthly_error_prcp.png",
            ],
        },
        {
            title: "[all - reg] Metrics Mean",
            files: [
                "minus_reg-metrics_mean.tsv", "minus_reg-metrics_mean.png",
                "minus_reg-monthly_mae.tsv", "minus_reg-monthly_mae.png",
                "minus_reg-monthly_rmse.tsv", "minus_reg-monthly_rmse.png",
            ],
        },
    ];

    // Parse query parameters
    const params = new URLSearchParams(window.location.search);
    const exp1 = params.get("exp1");
    const exp2 = params.get("exp2");

    if (!exp1 && !exp2) {
        document.getElementById("render-output").innerHTML = "<p>Error: No experiments selected.</p>";
        return;
    }

    // Set heading
    const heading = exp2
        ? `Comparison: ${exp1} vs. ${exp2}`
        : `Experiment: ${exp1 || exp2}`;
    document.getElementById("render-heading").textContent = heading;

    // Render collapsible sections
    FILE_GROUPS.forEach((group) => {
        renderCollapsibleSection(group.title, group.files, exp1, exp2);
    });

    // Add collapsible event listeners
    addCollapsibleEventListeners();
});

// Renders a collapsible section for a file group
function renderCollapsibleSection(title, files, exp1, exp2) {
    const renderOutput = document.getElementById("render-output");

    // Create collapsible header
    const collapsible = document.createElement("div");
    collapsible.className = "collapsible";
    collapsible.textContent = title;

    // Create collapsible content container
    const content = document.createElement("div");
    content.className = "content";

    // Render rows for each file
    files.forEach((file) => {
        const fileExtension = file.split(".").pop();

        const rowContainer = document.createElement("div");
        rowContainer.className = "render-row-container";

        // Add file name title
        const rowTitle = document.createElement("h3");
        rowTitle.className = "render-row-title";
        rowTitle.textContent = file;
        rowContainer.appendChild(rowTitle);

        const row = document.createElement("div");
        row.className = "render-row";

        if (fileExtension === "png") {
            // Render images
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
        } else if (fileExtension === "tsv") {
            // Render tables
            const tablePromises = [fetchTSV(`experiments/${exp1}/${file}`)];
            if (exp2) tablePromises.push(fetchTSV(`experiments/${exp2}/${file}`));

            Promise.all(tablePromises).then((tablesHTML) => {
                tablesHTML.forEach((tableHTML) => {
                    const tableDiv = document.createElement("div");
                    tableDiv.className = "render-table";
                    tableDiv.innerHTML = tableHTML;
                    row.appendChild(tableDiv);
                });
            });
        }  else if (fileExtension === "json") {
            // Render JSON data
            const jsonPromises = [fetchJSON(`experiments/${exp1}/${file}`)];
            if (exp2) jsonPromises.push(fetchJSON(`experiments/${exp2}/${file}`));

            Promise.all(jsonPromises).then((jsonHTML) => {
                jsonHTML.forEach((jsonDiv) => {
                    row.appendChild(jsonDiv);
                });
            });
        }

        rowContainer.appendChild(row);
        content.appendChild(rowContainer);
    });

    renderOutput.appendChild(collapsible);
    renderOutput.appendChild(content);
}

// Fetch and parse TSV files into HTML tables
function fetchTSV(url) {
    return fetch(url)
        .then((response) => {
            if (!response.ok) throw new Error(`Failed to fetch ${url}`);
            return response.text();
        })
        .then((tsvText) => {
            const rows = tsvText.split("\n").filter((row) => row.trim() !== "");
            const table = document.createElement("table");

            rows.forEach((row, index) => {
                const tr = document.createElement("tr");
                row.split('\t').forEach((cell) => {
                    const td = document.createElement(index === 0 ? "th" : "td");
                    td.textContent = cell.trim();
                    tr.appendChild(td);
                });
                table.appendChild(tr);
            });

            return table.outerHTML;
        })
        .catch((error) => `<p>Error loading TSV: ${url}</p>`);
}

// Add collapsible toggle functionality
function addCollapsibleEventListeners() {
    const collapsibles = document.querySelectorAll(".collapsible");
    collapsibles.forEach((collapsible) => {
        collapsible.addEventListener("click", () => {
            collapsible.classList.toggle("active");
            const content = collapsible.nextElementSibling;
            content.style.display = content.style.display === "block" ? "none" : "block";
        });
    });
}
