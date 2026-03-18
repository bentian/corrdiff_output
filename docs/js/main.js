import { fetchExperimentKeys } from "./util.js";

document.addEventListener("DOMContentLoaded", async () => {
    await loadExperiments();
    await loadExperimentGroups();

    // Handle form submissions
    document.getElementById("comparison-form")?.addEventListener("submit", handleComparisonSubmit);
    document.getElementById("summary-form")?.addEventListener("submit", handleSummarySubmit);
});

/**
 * Load experiment groups into dropdown.
 */
async function loadExperimentGroups() {
    const select = document.getElementById("summary-grp");

    try {
        const groups = await fetchExperimentKeys(/* isExperimentGroup= */ true);
        if (groups.length === 0) {
            console.error("No experiment groups found in the list.json file.");
            return;
        }

        populateDropdown(select, groups);

        const first = select.querySelector("option");
        if (first) select.value = first.value;

    } catch (error) {
        console.error("Error loading experiment groups:", error);
        setDropdownError(select);
    }
}

/**
 * Load experiments into two dropdowns.
 */
async function loadExperiments() {
    const exp1 = document.getElementById("exp1");
    const exp2 = document.getElementById("exp2");

    try {
        const list = await fetchExperimentKeys();
        if (!list.length) return console.error("No experiments found.");

        const grouped = groupByPrefix(list);
        [exp1, exp2].forEach(el => populateDropdownWithGroups(el, grouped));

        const opts1 = exp1.querySelectorAll("option");
        const opts2 = exp2.querySelectorAll("option");

        if (opts1[0]) exp1.value = opts1[0].value;
        if (opts2[1]) exp2.value = opts2[1].value;

    } catch (error) {
        console.error("Error loading experiments:", error);
        [exp1, exp2].forEach(setDropdownError);
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
 * Populate a dropdown with items.
 * @param { HTMLSelectElement } selectElement - The dropdown element to populate.
 * @param { string[] } items - List of items to populate the dropdown with.
 */
function populateDropdown(selectElement, items) {
    selectElement.innerHTML = ""; // clear existing options

    items.forEach(item => {
        const option = document.createElement("option");
        option.value = item;
        option.textContent = item;
        selectElement.appendChild(option);
    });
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

    const group = document.getElementById("summary-grp")?.value;

    if (!group) {
        alert("Please select an experiment group to summarize.");
        return;
    }

    const group_prefix = group.split("*")[0].trim();
    window.location.href = `render.html?grp=${group_prefix}`;
}
