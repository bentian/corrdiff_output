/**
 * Generates file groups to render.
 * @param {string[]} extraVariables - Extra variables to append to default variables.
 */
function generateFileGroups(extraVariables) {
    const prefixes = ["all", "reg"];
    const defaultVariables = ["prcp", "t2m", "u10m", "v10m"];
    const variables = [...defaultVariables, ...extraVariables];

    const overviewFiles = [
        "metrics_mean.tsv", "metrics_mean.png",
        "monthly_rmse.tsv", "monthly_rmse.png",
        "monthly_mae.tsv", "monthly_mae.png",
        "monthly_crps.tsv", "monthly_crps.png",
        "monthly_std_dev.tsv", "monthly_std_dev.png",
    ];
    const variableFiles = [
        "pdf.png", "monthly_error.png",
        "cnt_rmse.png", "top_samples_rmse.png",
        "cnt_mae.png", "top_samples_mae.png",
    ];
    const ensembleFile = "metrics_v_ensembles.png";

    return prefixes.map(prefix => ({
        title: `[${prefix}] Metrics`,
        files: Object.fromEntries([
            ["overview", overviewFiles.map(f => `${prefix}/overview/${f}`)],
            ...variables.map(varName => [
                varName,
                [
                    ...variableFiles.map(f => `${prefix}/${varName}/${f}`),
                    ...(prefix === "all" ? [`${prefix}/${varName}/${ensembleFile}`] : [])
                ]
            ])
        ])
    })).concat([
        {
            title: "[all - reg] Metrics",
            files: overviewFiles.map(f => `minus_reg/${f}`)
        },
        {
            title: "Training Loss",
            files: ["training_loss_regression.png", "training_loss_diffusion.png"]
        },
        {
            title: "Config",
            files: ["train_config.tsv", "generate_config.tsv"]
        }
    ]);
}

/**
 * Handles scrolling and expanding based on hash change.
 */
function handleHashChange() {
    const hash = window.location.hash.substring(1);
    if (!hash) return;

    const targetRow = document.getElementById(hash);
    if (!targetRow) return;

    const targetContent = targetRow.closest(".content");
    if (targetContent) targetContent.style.display = "block";

    const collapsibleHeader = targetContent?.previousElementSibling;
    if (collapsibleHeader?.classList.contains("collapsible")) {
        collapsibleHeader.classList.add("active");
    }

    const targetTabContent = targetRow.closest(".tab-content");
    if (targetTabContent) {
        activateSingleTab(targetContent, targetTabContent.tab, targetTabContent);
    }

    targetRow.scrollIntoView({ behavior: "smooth", block: "center" });
}

/**
 * Activates a single tab and deactivates others within a container.
 */
function activateSingleTab(content, tab, tabContent) {
    content.querySelectorAll(".tab, .tab-content").forEach(el => el.classList.remove("active"));
    if (tab) tab.classList.add("active");
    if (tabContent) tabContent.classList.add("active");
}

/**
 * Adds collapsible toggle functionality.
 */
function addCollapsibleEventListeners() {
    document.querySelectorAll(".collapsible").forEach(collapsible => {
        collapsible.addEventListener("click", () => {
            collapsible.classList.toggle("active");
            const content = collapsible.nextElementSibling;
            content.style.display = content.style.display === "block" ? "none" : "block";
            if (content.style.display === "none") return;

            const [firstTab] = content.querySelectorAll(".tab");
            const [firstTabContent] = content.querySelectorAll(".tab-content");
            activateSingleTab(content, firstTab, firstTabContent);
        });
    });
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

    lightbox.addEventListener("click", (event) => {
        if (event.target === lightbox || event.target === closeLightbox) {
            lightbox.style.display = "none";
        }
    });
}

// Export functions for use in render.js
export {
    generateFileGroups, handleHashChange, activateSingleTab,
    addCollapsibleEventListeners, initializeLightbox
};
