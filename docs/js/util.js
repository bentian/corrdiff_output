const URL_EXP_SHEET = "https://docs.google.com/spreadsheets/d/1_wKKzvP1Jc5HOEFOuQXurRa4_oECnvS7EsKz2akQIK8/edit?gid=1634578103#gid=1634578103"

let _experimentCache = null;
let _groupCache = null;

/**
 * Load and cache experiment or experiment-group map from JSON.
 * @param {boolean} isExperimentGroup - Whether to load experiment-group data.
 * @returns {Promise<Object>} Key-value map of experiments.
 */
async function loadExperimentMap(isExperimentGroup = false) {
    const cache = isExperimentGroup ? _groupCache : _experimentCache;
    if (cache) return cache;

    const url = isExperimentGroup ? "comparisons/list.json" : "experiments/list.json";
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Fetch failed: ${res.statusText}`);

    const data = await res.json();
    if (isExperimentGroup) {
        _groupCache = data;
    } else {
        _experimentCache = data;
    }

    return data;
}

/**
 * Fetch experiment keys from the loaded map.
 * @param {boolean} isExperimentGroup - Whether to load comparison group data.
 * @returns {Promise<string[]>} Array of experiment keys.
 */
async function fetchExperimentKeys(isExperimentGroup = false) {
    return Object.keys(await loadExperimentMap(isExperimentGroup));
}

/**
 * Fetch experiment link by key from the loaded map.
 * @param {string} key - The experiment key to fetch.
 * @param {boolean} isExperimentGroup - Whether to load comparison group data.
 * @returns {Promise<string|null>} The experiment link or null if not found.
 */
async function fetchExperimentLink(key, isExperimentGroup = false) {
    const value = (await loadExperimentMap(isExperimentGroup))[key];
    return value ? `${URL_EXP_SHEET}&range=${value}` : null;
}

/**
 * Generates experiment group files to render.
 *
 * @param {string} group - Experiment group name.
 */
function generateExperimentGroupFiles(group) {
    const base = {
        prcp: ["mean_cmp.png"],
        t2m: ["mean_cmp.png"],
        u10m: ["mean_cmp.png"],
        v10m: ["mean_cmp.png"]
    };

    return [{
        title: group === "DM" ? "Metrics Mean" : "Metrics Mean & Decadal Trends",
        files: Object.fromEntries(
            Object.entries(base).map(([k, v]) => [
                k,
                [
                    ...[`${k}/mean_cmp.png`],
                    // ...(group === "CropW" ? [`${k}/mean_w_cmp.png`] : []),
                    ...(group === "DM" ? [] : [`${k}/nyear_cmp.png`])
                ]
            ])
        )
    }];
}

/**
 * Generates experiment files to render.
 *
 * @param {string} exp1 - Experiment 1 name.
 * @param {string} exp2 - Experiment 2 name (optional).
 */
function generateExperimentFiles(exp1, exp2) {
    const hasSSP = exp1.startsWith("W") || exp2?.startsWith("W")

    // Overview files
    const metrics = ["rmse", "mae", "corr", "crps", "std_dev"];
    const periods = hasSSP ? ["monthly_metrics", "nyear_metrics"] : ["monthly_metrics"];
    const exts = ["tsv", "png"];
    const overviewFiles = [
        "metrics_mean.tsv",
        "metrics_mean.png",
        ...periods.flatMap(period =>
            metrics.flatMap(metric =>
                exts.map(ext => `${period}/${metric}.${ext}`)
            )
        ),
    ];

    // Variable files
    const variableFiles = [
        "pdf.png", "monthly_error.png",
        "cnt_rmse.png", "top_samples_rmse.png",
        "cnt_mae.png", "top_samples_mae.png",
    ];
    const ensembleFile = "metrics_v_ensembles.png";
    const p90File = "p90_by_nyear.png";

    // Prefixes & variables
    const prefixes = ["all", "reg"];
    const variables = ["prcp", "t2m", "u10m", "v10m"]
    const p90Vars = new Set(["prcp", "t2m"]);

    // Create basic file groups
    const groupList = prefixes.map(prefix => {
        const buildPath = (folder, file) => `${prefix}/${folder}/${file}`;
        const buildVarFiles = varName => [
            ...variableFiles.map(f => buildPath(varName, f)),
            ...(hasSSP && p90Vars.has(varName) ? [buildPath(varName, p90File)] : []),
            ...(hasSSP && prefix === "all" ? [buildPath(varName, ensembleFile)] : [])
        ];

        return {
            title: `[${prefix}] Metrics`,
            files: {
                overview: overviewFiles.map(f => buildPath("overview", f)),
                ...Object.fromEntries(variables.map(varName => [varName, buildVarFiles(varName)]))
            }
        };
    });

    // Append file groups
    groupList.push(
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
    );

    return groupList;
}

/**
 * Scrolls an element into view only after its layout is stable.
 *
 * This function waits for all <img> elements inside the target element
 * to finish loading before calling `scrollIntoView()`. This prevents
 * incorrect scroll positions caused by late-loading images that change
 * the element's height after the scroll occurs.
 *
 * If the element contains no images, scrolling happens immediately.
 * Images that are already loaded (`img.complete === true`) are counted
 * as ready and do not delay scrolling.
 *
 * @param {HTMLElement} el - The target element to scroll into view.
 */
function scrollWhenReady(el) {
    const imgs = el.querySelectorAll("img");
    let pending = imgs.length;

    if (!pending) {
        el.scrollIntoView({ block: "center" });
        return;
    }

    imgs.forEach(img => {
        if (img.complete) {
            if (--pending === 0) el.scrollIntoView({ block: "center" });
        } else {
            img.addEventListener("load", () => {
                if (--pending === 0) el.scrollIntoView({ block: "center" });
            }, { once: true });
        }
    });
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

    scrollWhenReady(targetRow)
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
    URL_EXP_SHEET,
    fetchExperimentKeys, fetchExperimentLink,
    generateExperimentGroupFiles, generateExperimentFiles,
    handleHashChange, activateSingleTab,
    addCollapsibleEventListeners, initializeLightbox
};
