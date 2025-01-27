document.addEventListener("DOMContentLoaded", () => {
    // Fetch experiments dynamically
    fetch("experiments/list.json")
        .then((response) => {
            if (!response.ok) {
                throw new Error(`Failed to fetch experiments: ${response.statusText}`);
            }
            return response.json();
        })
        .then((experiments) => {
            if (!Array.isArray(experiments) || experiments.length === 0) {
                console.error("No experiments found in the list.json file.");
                return;
            }

            // Populate dropdowns
            const exp1Select = document.getElementById("exp1");
            const exp2Select = document.getElementById("exp2");
            const summaryExpSelect = document.getElementById("summary-exp");

            // Populate dropdowns with fetched experiments
            experiments.forEach((exp) => {
                const option1 = document.createElement("option");
                const option2 = document.createElement("option");
                const optionSummary = document.createElement("option");

                option1.value = exp;
                option1.textContent = exp;

                option2.value = exp;
                option2.textContent = exp;

                optionSummary.value = exp;
                optionSummary.textContent = exp;

                exp1Select.appendChild(option1);
                exp2Select.appendChild(option2);
                summaryExpSelect.appendChild(optionSummary);
            });

            // Set default selections
            if (experiments.length > 1) {
                exp1Select.selectedIndex = 0;
                exp2Select.selectedIndex = 1;
                summaryExpSelect.selectedIndex = 0;
            }
        })
        .catch((error) => {
            console.error("Error fetching experiments:", error);

            const exp1Select = document.getElementById("exp1");
            const exp2Select = document.getElementById("exp2");
            const summaryExpSelect = document.getElementById("summary-exp");

            exp1Select.innerHTML = `<option value="">Error loading experiments</option>`;
            exp2Select.innerHTML = `<option value="">Error loading experiments</option>`;
            summaryExpSelect.innerHTML = `<option value="">Error loading experiments</option>`;
        });

    // Handle comparison form submission
    document.getElementById("comparison-form").addEventListener("submit", (e) => {
        e.preventDefault();

        const exp1 = document.getElementById("exp1").value;
        const exp2 = document.getElementById("exp2").value;

        if (!exp1 || !exp2) {
            alert("Please select both experiments before comparing.");
            return;
        }

        // Redirect to the comparison page with selected experiments
        window.location.href = `render.html?exp1=${exp1}&exp2=${exp2}`;
    });

    // Handle summary form submission
    document.getElementById("summary-form").addEventListener("submit", (e) => {
        e.preventDefault();

        const exp = document.getElementById("summary-exp").value;

        if (!exp) {
            alert("Please select an experiment to summarize.");
            return;
        }

        // Redirect to summary page (uses the same comparison page)
        window.location.href = `render.html?exp1=${exp}`;
    });
});
