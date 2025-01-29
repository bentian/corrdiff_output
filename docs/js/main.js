document.addEventListener("DOMContentLoaded", async () => {
    await loadExperiments();

    // Handle form submissions
    document.getElementById("comparison-form")?.addEventListener("submit", handleComparisonSubmit);
    document.getElementById("summary-form")?.addEventListener("submit", handleSummarySubmit);
});

/**
 * Fetches the list of experiments and populates the dropdowns.
 */
async function loadExperiments() {
    const exp1Select = document.getElementById("exp1");
    const exp2Select = document.getElementById("exp2");
    const summaryExpSelect = document.getElementById("summary-exp");

    try {
        const response = await fetch("experiments/list.json");
        if (!response.ok) throw new Error(`Failed to fetch experiments: ${response.statusText}`);

        const experiments = await response.json();
        if (!Array.isArray(experiments) || experiments.length === 0) {
            console.error("No experiments found in the list.json file.");
            return;
        }

        populateDropdown(exp1Select, experiments);
        populateDropdown(exp2Select, experiments);
        populateDropdown(summaryExpSelect, experiments);

        // Set default selections if more than one experiment is available
        if (experiments.length > 1) {
            exp1Select.selectedIndex = 0;
            exp2Select.selectedIndex = 1;
            summaryExpSelect.selectedIndex = 0;
        }
    } catch (error) {
        console.error("Error fetching experiments:", error);
        setDropdownError(exp1Select);
        setDropdownError(exp2Select);
        setDropdownError(summaryExpSelect);
    }
}

/**
 * Populates a given dropdown with experiment options.
 * @param {HTMLSelectElement} selectElement - The dropdown element to populate.
 * @param {string[]} experiments - List of experiment names.
 */
function populateDropdown(selectElement, experiments) {
    if (!selectElement) return;

    experiments.forEach((exp) => {
        const option = document.createElement("option");
        option.value = exp;
        option.textContent = exp;
        selectElement.appendChild(option);
    });
}

/**
 * Sets an error message for a dropdown when experiments fail to load.
 * @param {HTMLSelectElement} selectElement - The dropdown element.
 */
function setDropdownError(selectElement) {
    if (!selectElement) return;
    selectElement.innerHTML = `<option value="">Error loading experiments</option>`;
}

/**
 * Handles the submission of the comparison form.
 * @param {Event} event - The form submission event.
 */
function handleComparisonSubmit(event) {
    event.preventDefault();

    const exp1 = document.getElementById("exp1")?.value;
    const exp2 = document.getElementById("exp2")?.value;

    if (!exp1 || !exp2) {
        alert("Please select both experiments before comparing.");
        return;
    }

    window.location.href = `render.html?exp1=${exp1}&exp2=${exp2}`;
}

/**
 * Handles the submission of the summary form.
 * @param {Event} event - The form submission event.
 */
function handleSummarySubmit(event) {
    event.preventDefault();

    const exp = document.getElementById("summary-exp")?.value;

    if (!exp) {
        alert("Please select an experiment to summarize.");
        return;
    }

    window.location.href = `render.html?exp1=${exp}`;
}
