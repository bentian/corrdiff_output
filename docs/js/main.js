import { fetchExperimentKeys } from "./util.js";

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
        const experiments = await fetchExperimentKeys();
        if (experiments.length === 0) {
            console.error("No experiments found in the list.json file.");
            return;
        }

        const grouped = groupByPrefix(experiments);

        populateDropdownWithGroups(exp1Select, grouped);
        populateDropdownWithGroups(exp2Select, grouped);
        populateDropdownWithGroups(summaryExpSelect, grouped);

        // Set default selections if more than one experiment is available
        const firstOption = exp1Select.querySelector("option");
        const secondOption = exp2Select.querySelectorAll("option")[1];
        if (firstOption) exp1Select.value = firstOption.value;
        if (secondOption) exp2Select.value = secondOption.value;
        if (firstOption) summaryExpSelect.value = firstOption.value;

    } catch (error) {
        console.error("Error fetching experiments:", error);
        setDropdownError(exp1Select);
        setDropdownError(exp2Select);
        setDropdownError(summaryExpSelect);
    }
}

/**
 * Groups an array of experiment names by their prefix.
 *
 * Prefix is determined by splitting each experiment string at the first underscore (_).
 * For example, "ERA5_2M" has the prefix "ERA5".
 *
 * @param {string[]} experiments - List of experiment names.
 * @returns {Object} An object where keys are prefixes and values are arrays of experiment names.
 */
function groupByPrefix(experiments) {
    return experiments.reduce((acc, exp) => {
        const prefix = exp.split("-")[0]; // Customize as needed
        if (!acc[prefix]) acc[prefix] = [];
        acc[prefix].push(exp);
        return acc;
    }, {});
}

/**
 * Populates a <select> dropdown with grouped options using <optgroup> tags.
 *
 * Each group is labeled by a prefix, and contains related experiment <option> elements.
 *
 * @param {HTMLSelectElement} selectElement - The dropdown element to populate.
 * @param {Object} groupedData - An object with prefixes as keys and
 *                               arrays of experiment names as values.
 */
function populateDropdownWithGroups(selectElement, groupedData) {
    selectElement.innerHTML = ""; // Clear existing options

    for (const [prefix, exps] of Object.entries(groupedData)) {
        const group = document.createElement("optgroup");
        group.label = prefix;

        exps.forEach(exp => {
            const option = document.createElement("option");
            option.value = exp;
            option.textContent = exp;
            group.appendChild(option);
        });

        selectElement.appendChild(group);
    }
}

/**
 * Sets an error message for a dropdown when experiments fail to load.
 *
 * @param {HTMLSelectElement} selectElement - The dropdown element.
 */
function setDropdownError(selectElement) {
    if (!selectElement) return;
    selectElement.innerHTML = `<option value="">Error loading experiments</option>`;
}

/**
 * Handles the submission of the comparison form.
 *
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
 *
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
