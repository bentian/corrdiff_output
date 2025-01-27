// Define the ordered file list
const FILE_ORDER_LIST = [
    // regression + diffusion model
    "all-metrics_mean.csv", "all-metrics_mean.png",
    "all-monthly_mae.csv", "all-monthly_mae.png",
    "all-monthly_rmse.csv", "all-monthly_rmse.png",
    "all-pdf_prcp.png", // "pdf_t2m.png", "pdf_u10m.png", "pdf_v10m.png",
    "all-monthly_error_prcp.png", // "monthly_error_t2m.png",
    // "monthly_error_u10m.png", "monthly_error_v10m.png",

    // regression + diffusion model minus regression model only
    "minus_reg-metrics_mean.csv", "minus_reg-metrics_mean.png",
    "minus_reg-monthly_mae.csv", "minus_reg-monthly_mae.png",
    "minus_reg-monthly_rmse.csv", "minus_reg-monthly_rmse.png",
];

document.addEventListener("DOMContentLoaded", () => {
    // Parse query parameters
    const params = new URLSearchParams(window.location.search);
    const exp1 = params.get("exp1");
    const exp2 = params.get("exp2");

    if (!exp1 && !exp2) {
        document.getElementById("render-output").innerHTML = "<p>Error: No experiments selected.</p>";
        return;
    }

    // Set heading based on the number of experiments
    const heading = exp2
        ? `Comparison: ${exp1} vs. ${exp2}`
        : `Experiment: ${exp1 || exp2}`;
    document.getElementById("render-heading").textContent = heading;

    // Render content
    if (exp2) {
        renderComparison(exp1, exp2, FILE_ORDER_LIST);
    } else {
        renderSingleExperiment(exp1 || exp2, FILE_ORDER_LIST);
    }
});

function renderSingleExperiment(exp1, FILE_ORDER_LIST) {
    const renderOutput = document.getElementById("render-output");

    FILE_ORDER_LIST.forEach((file) => {
        const fileExtension = file.split(".").pop();

        const rowContainer = document.createElement("div");
        rowContainer.className = "render-row-container";

        // Add row title
        const rowTitle = document.createElement("h3");
        rowTitle.className = "render-row-title";
        rowTitle.textContent = file;
        rowContainer.appendChild(rowTitle);

        // Create a new summary row
        const row = document.createElement("div");
        row.className = "render-row";

        if (fileExtension === "png") {
            // Render images side-by-side
            const img1 = document.createElement("img");
            img1.src = `experiments/${exp1}/${file}`;
            img1.alt = `${exp1} - ${file}`;
            img1.className = "render-plot";

            row.appendChild(img1);
        } else if (fileExtension === "csv") {
            // Fetch and render CSV files as HTML tables
            Promise.all([
                fetchCSV(`experiments/${exp1}/${file}`),
            ]).then(([table1HTML]) => {
                const table1 = document.createElement("div");
                table1.className = "render-table";
                table1.innerHTML = table1HTML;

                row.appendChild(table1);
            });
        } else {
            console.warn(`Unsupported file type: ${file}`);
        }

        rowContainer.appendChild(row);
        renderOutput.appendChild(rowContainer);
    });
}


function renderComparison(exp1, exp2, FILE_ORDER_LIST) {
    const renderOutput = document.getElementById("render-output");

    FILE_ORDER_LIST.forEach((file) => {
        const fileExtension = file.split(".").pop();

        const rowContainer = document.createElement("div");
        rowContainer.className = "render-row-container";

        // Add row title
        const rowTitle = document.createElement("h3");
        rowTitle.className = "render-row-title";
        rowTitle.textContent = file;
        rowContainer.appendChild(rowTitle);

        // Create a new comparison row
        const row = document.createElement("div");
        row.className = "render-row";

        if (fileExtension === "png") {
            // Render images side-by-side
            const img1 = document.createElement("img");
            img1.src = `experiments/${exp1}/${file}`;
            img1.alt = `${exp1} - ${file}`;
            img1.className = "render-plot";

            const img2 = document.createElement("img");
            img2.src = `experiments/${exp2}/${file}`;
            img2.alt = `${exp2} - ${file}`;
            img2.className = "render-plot";

            row.appendChild(img1);
            row.appendChild(img2);
        } else if (fileExtension === "csv") {
            // Fetch and render CSV files as HTML tables
            Promise.all([
                fetchCSV(`experiments/${exp1}/${file}`),
                fetchCSV(`experiments/${exp2}/${file}`),
            ]).then(([table1HTML, table2HTML]) => {
                const table1 = document.createElement("div");
                table1.className = "render-table";
                table1.innerHTML = table1HTML;

                const table2 = document.createElement("div");
                table2.className = "render-table";
                table2.innerHTML = table2HTML;

                row.appendChild(table1);
                row.appendChild(table2);
            });
        } else {
            console.warn(`Unsupported file type: ${file}`);
        }

        rowContainer.appendChild(row);
        renderOutput.appendChild(rowContainer);
    });
}

// Utility function to fetch and parse CSV files into HTML tables
function fetchCSV(url) {
    return fetch(url)
        .then((response) => {
            if (!response.ok) throw new Error(`Failed to fetch ${url}`);
            return response.text();
        })
        .then((csvText) => {
            const rows = csvText.split("\n").filter((row) => row.trim() !== "");
            const table = document.createElement("table");
            table.border = "1";

            rows.forEach((row, index) => {
                const tr = document.createElement("tr");
                row.split(",").forEach((cell) => {
                    const td = document.createElement(index === 0 ? "th" : "td");
                    td.textContent = cell.trim();
                    tr.appendChild(td);
                });
                table.appendChild(tr);
            });

            return table.outerHTML;
        })
        .catch((error) => {
            console.error("Error rendering CSV:", error);
            return `<p>Error loading CSV: ${url}</p>`;
        });
}
    
