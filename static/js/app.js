"use strict";

const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : window.location.origin;
const MAX_FILE_SIZE = 10 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set(["pdf", "txt", "md"]);
const selectedFiles = [];

const $ = (id) => document.getElementById(id);

class ApiError extends Error {
  constructor(message, status = 0, data = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.data = data;
  }
}

function createElement(tag, className = "", text = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== null && text !== undefined) element.textContent = String(text);
  return element;
}

function finiteNumber(value) {
  if (value === null || value === undefined || typeof value === "boolean") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function clamp(value, minimum, maximum) {
  return Math.min(Math.max(value, minimum), maximum);
}

function formatError(error) {
  if (error instanceof ApiError) return error.message;
  if (error?.name === "AbortError") return "The request took too long and was cancelled.";
  return error?.message || "An unexpected error occurred.";
}

function showToast(message, type = "info", duration = 5000) {
  const toast = createElement("div", `toast ${type}`, message);
  $("toast-region").appendChild(toast);
  window.setTimeout(() => toast.remove(), duration);
}

async function apiFetch(path, options = {}, timeoutMs = 120000) {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(`${API_BASE}${path}`, {
      ...options,
      signal: controller.signal,
      headers: {
        Accept: "application/json",
        ...(options.headers || {})
      }
    });

    const contentType = response.headers.get("content-type") || "";
    let data = null;

    if (response.status !== 204) {
      if (contentType.includes("application/json")) {
        data = await response.json();
      } else {
        const text = await response.text();
        data = text ? { detail: text } : {};
      }
    }

    if (!response.ok) {
      const detail = data?.detail;
      const message = Array.isArray(detail)
        ? detail.map((item) => item.msg || JSON.stringify(item)).join("; ")
        : detail || data?.message || `Request failed with status ${response.status}.`;
      throw new ApiError(String(message), response.status, data);
    }

    return data || {};
  } finally {
    window.clearTimeout(timeout);
  }
}

function setButtonLoading(button, loading, loadingText) {
  if (!button.dataset.originalText) button.dataset.originalText = button.textContent;
  button.disabled = loading;
  button.textContent = loading ? loadingText : button.dataset.originalText;
}

function scoreColor(value) {
  const number = finiteNumber(value);
  if (number === null) return "var(--subtle)";
  if (number >= 0.7) return "var(--green)";
  if (number >= 0.4) return "var(--amber)";
  return "var(--red)";
}

function judgeBadgeClass(verdict) {
  const value = String(verdict || "").toLowerCase();
  if (
    value.includes("highly relevant") ||
    value === "accurate" ||
    value === "grounded" ||
    value === "complete" ||
    value === "excellent" ||
    value === "good" ||
    value === "pass"
  ) return "good";

  if (value.includes("partial") || value.includes("limited") || value === "unknown") return "warn";
  if (value.includes("cannot") || value.includes("error") || !value) return "neutral";
  return "bad";
}

function createBadge(text, forcedClass = null) {
  const value = text || "Unknown";
  return createElement("span", `badge ${forcedClass || judgeBadgeClass(value)}`, value);
}

function switchTab(tabId) {
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    const active = panel.id === `tab-${tabId}`;
    panel.hidden = !active;
    panel.classList.toggle("active", active);
  });

  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });

  closeMobileMenu();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function openMobileMenu() {
  $("sidebar").classList.add("open");
  $("sidebar-overlay").hidden = false;
  $("mobile-menu-button").setAttribute("aria-expanded", "true");
}

function closeMobileMenu() {
  $("sidebar").classList.remove("open");
  $("sidebar-overlay").hidden = true;
  $("mobile-menu-button").setAttribute("aria-expanded", "false");
}

async function fetchKnowledgeBaseStats() {
  const status = $("kb-status");
  try {
    const data = await apiFetch("/api/kb/stats", {}, 30000);
    $("kb-chunk-count").textContent = String(data.total_chunks ?? 0);
    status.className = "status-line online";
    status.lastElementChild.textContent = "Database connected";
  } catch (error) {
    $("kb-chunk-count").textContent = "—";
    status.className = "status-line offline";
    status.lastElementChild.textContent = "Backend unavailable";
    console.error(error);
  }
}

async function loadBenchmark() {
  const dataset = $("benchmark-select").value;
  if (!dataset) {
    showToast("Select a benchmark dataset first.", "warning");
    return;
  }

  const button = $("benchmark-button");
  setButtonLoading(button, true, "…");
  try {
    const data = await apiFetch("/api/kb/load-benchmark", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset })
    }, 600000);
    await fetchKnowledgeBaseStats();
    showToast(`Added ${data.chunks_added ?? 0} benchmark chunks.`, "success");
  } catch (error) {
    showToast(`Benchmark load failed: ${formatError(error)}`, "error", 8000);
  } finally {
    setButtonLoading(button, false, "");
  }
}

function renderSelectedFiles(fileList) {
  const container = $("file-list");
  container.replaceChildren();
  selectedFiles.splice(0, selectedFiles.length);

  for (const file of fileList) {
    const extension = file.name.includes(".") ? file.name.split(".").pop().toLowerCase() : "";
    if (!ALLOWED_EXTENSIONS.has(extension)) {
      showToast(`${file.name}: unsupported file type.`, "warning");
      continue;
    }
    if (file.size > MAX_FILE_SIZE) {
      showToast(`${file.name}: file exceeds the 10 MB frontend limit.`, "warning");
      continue;
    }

    selectedFiles.push(file);
    const row = createElement("div", "file-row");
    row.append(
      createElement("span", "", file.name),
      createElement("span", "", `${Math.max(1, Math.round(file.size / 1024))} KB`)
    );
    container.appendChild(row);
  }

  $("upload-button").hidden = selectedFiles.length === 0;
}

async function uploadFiles() {
  if (!selectedFiles.length) return;

  const button = $("upload-button");
  setButtonLoading(button, true, "Indexing files…");
  const formData = new FormData();
  selectedFiles.forEach((file) => formData.append("files", file));

  try {
    const data = await apiFetch("/api/kb/upload", { method: "POST", body: formData }, 600000);
    showToast(`Indexed ${data.chunks_added ?? 0} chunks.`, "success");
    selectedFiles.splice(0, selectedFiles.length);
    $("file-list").replaceChildren();
    $("file-input").value = "";
    button.hidden = true;
    await fetchKnowledgeBaseStats();
  } catch (error) {
    showToast(`Upload failed: ${formatError(error)}`, "error", 8000);
  } finally {
    setButtonLoading(button, false, "");
  }
}

async function resetKnowledgeBase() {
  if (!window.confirm("Delete all indexed knowledge-base chunks? This cannot be undone from the dashboard.")) return;
  const button = $("reset-button");
  setButtonLoading(button, true, "Resetting…");
  try {
    await apiFetch("/api/kb/reset", { method: "POST" }, 60000);
    await fetchKnowledgeBaseStats();
    showToast("Knowledge base reset successfully.", "success");
  } catch (error) {
    showToast(`Reset failed: ${formatError(error)}`, "error", 8000);
  } finally {
    setButtonLoading(button, false, "");
  }
}

function createMetricCard(label, value) {
  const card = createElement("div", "metric-card");
  const numericValue = finiteNumber(value);
  const displayValue = numericValue === null ? "N/A" : `${Math.round(clamp(numericValue, 0, 1) * 100)}%`;
  const strong = createElement("strong", "", displayValue);
  strong.style.color = scoreColor(numericValue);
  const caption = createElement("span", "", label);
  const bar = createElement("div", "metric-bar");
  const fill = createElement("div");
  fill.style.width = numericValue === null ? "0%" : `${clamp(numericValue, 0, 1) * 100}%`;
  fill.style.background = scoreColor(numericValue);
  bar.appendChild(fill);
  card.append(strong, caption, bar);
  return card;
}

function normalizeRetrievalMetric(value) {
  if (value && typeof value === "object") return finiteNumber(value.average ?? value.maximum);
  return finiteNumber(value);
}

function renderChunks(container, chunks) {
  container.replaceChildren();
  if (!Array.isArray(chunks) || chunks.length === 0) {
    const empty = createElement("div", "notice", "No matching evidence was returned.");
    container.appendChild(empty);
    return;
  }

  chunks.forEach((chunk, index) => {
    const card = createElement("article", "chunk-card");
    const meta = createElement("div", "chunk-meta");
    const source = chunk?.source || "unknown";
    meta.append(
      createElement("span", "", `Rank ${index + 1} · ${source}`),
      createElement("span", "", finiteNumber(chunk?.similarity_score) === null ? "Score N/A" : `Score ${Number(chunk.similarity_score).toFixed(4)}`)
    );
    card.append(meta, createElement("p", "", chunk?.chunk || ""));
    container.appendChild(card);
  });
}

async function runEvaluation(event) {
  event.preventDefault();
  const question = $("evaluation-question").value.trim();
  const aiResponse = $("evaluation-response").value.trim();
  const reference = $("evaluation-reference").value.trim();
  if (!question || !aiResponse) return;

  const button = $("evaluation-submit");
  setButtonLoading(button, true, "Evaluating…");
  try {
    const data = await apiFetch("/api/evaluate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        ai_response: aiResponse,
        reference_answer: reference || null,
        top_k: Number($("evaluation-top-k").value)
      })
    }, 180000);

    const scores = data.scores || data;
    const metrics = [
      ["Question-response", scores.question_response_relevance],
      ["Retrieval relevance", normalizeRetrievalMetric(scores.retrieval_relevance)],
      ["Response grounding", scores.response_grounding],
      ["Semantic similarity", scores.semantic_similarity],
      ["Token-overlap F1", scores.token_f1]
    ];

    const grid = $("metric-grid");
    grid.replaceChildren(...metrics.map(([label, value]) => createMetricCard(label, value)));
    const chunks = Array.isArray(data.retrieved_chunks) ? data.retrieved_chunks : [];
    $("retrieved-count").textContent = `${chunks.length} chunk${chunks.length === 1 ? "" : "s"}`;
    renderChunks($("retrieved-chunks"), chunks);
    $("evaluation-placeholder").hidden = true;
    $("evaluation-results").hidden = false;
  } catch (error) {
    showToast(`Evaluation failed: ${formatError(error)}`, "error", 8000);
  } finally {
    setButtonLoading(button, false, "");
  }
}

function aggregateJudges(judgments) {
  const weights = { relevance: 0.25, accuracy: 0.35, hallucination: 0.25, completeness: 0.15 };
  let weighted = 0;
  let availableWeight = 0;

  for (const [name, weight] of Object.entries(weights)) {
    const score = finiteNumber(judgments?.[name]?.score);
    if (score !== null && score >= 0 && score <= 10) {
      weighted += score * weight;
      availableWeight += weight;
    }
  }

  if (!availableWeight) return null;
  const overallScore = (weighted / availableWeight) * 10;
  let verdict = overallScore >= 85 ? "Excellent" : overallScore >= 70 ? "Good" : overallScore >= 50 ? "Needs Improvement" : "Poor";
  if (availableWeight < 0.75) verdict += " — Limited Evidence";
  return { overall_score: Math.round(overallScore * 100) / 100, verdict, evaluation_coverage: Math.round(availableWeight * 100) / 100, limited_evidence: availableWeight < 0.75 };
}

function createJudgeCard(name, result) {
  const definitions = {
    relevance: ["🎯", "Relevance Judge"],
    accuracy: ["✅", "Accuracy Judge"],
    hallucination: ["🔍", "Hallucination Detector"],
    completeness: ["📋", "Completeness Judge"]
  };
  const [icon, title] = definitions[name];
  const card = createElement("article", "judge-card");
  const heading = createElement("div", "judge-heading");
  heading.append(createElement("strong", "", `${icon} ${title}`));

  const right = createElement("div", "judge-heading");
  const score = finiteNumber(result?.score);
  right.append(
    createElement("span", "judge-score", score === null ? "N/A" : `${score}/10`),
    createBadge(result?.verdict || (score === null ? "Cannot Evaluate" : "Unknown"))
  );
  heading.appendChild(right);
  card.append(heading, createElement("p", "", result?.reasoning || "No reasoning was returned."));

  if (result?.evidence) card.appendChild(createElement("div", "evidence-box", `Evidence: ${result.evidence}`));
  if (Array.isArray(result?.flagged_statements) && result.flagged_statements.length) {
    const flags = createElement("div", "flag-list");
    flags.appendChild(createElement("strong", "", "Flagged unsupported statements"));
    const list = createElement("ul");
    result.flagged_statements.forEach((statement) => list.appendChild(createElement("li", "", statement)));
    flags.appendChild(list);
    card.appendChild(flags);
  }
  return card;
}

function renderOverallResult(finalEvaluation) {
  const container = $("overall-result");
  if (!finalEvaluation || finiteNumber(finalEvaluation.overall_score) === null) {
    container.hidden = true;
    container.replaceChildren();
    return;
  }

  const score = finiteNumber(finalEvaluation.overall_score);
  const scoreNode = createElement("div", "overall-score", `${score.toFixed(1)}%`);
  const description = createElement("div");
  description.append(
    createElement("strong", "", finalEvaluation.verdict || "Overall evaluation"),
    createElement("p", "", `Evaluation coverage: ${Math.round((finiteNumber(finalEvaluation.evaluation_coverage) ?? 0) * 100)}%`)
  );
  container.replaceChildren(scoreNode, description, createBadge(finalEvaluation.limited_evidence ? "Limited Evidence" : "Complete Evaluation", finalEvaluation.limited_evidence ? "warn" : "good"));
  container.hidden = false;
}

async function runJudges(event) {
  event.preventDefault();
  const question = $("judge-question").value.trim();
  const aiResponse = $("judge-response").value.trim();
  const reference = $("judge-reference").value.trim();
  if (!question || !aiResponse) return;

  const button = $("judge-submit");
  setButtonLoading(button, true, "Running judges…");
  try {
    const data = await apiFetch("/api/judge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        ai_response: aiResponse,
        reference_answer: reference || null,
        top_k: Number($("judge-top-k").value)
      })
    }, 300000);

    const judgments = data.judgments || data.judges || data.scores?.judges || {};
    const order = ["relevance", "accuracy", "hallucination", "completeness"];
    const cards = order.map((name) => createJudgeCard(name, judgments[name] || {
      score: null,
      verdict: "Not Returned",
      reasoning: `${name[0].toUpperCase()}${name.slice(1)} result was not returned by the backend.`
    }));
    $("judge-cards").replaceChildren(...cards);

    const finalEvaluation = data.final_evaluation || data.scores?.final_evaluation || aggregateJudges(judgments);
    renderOverallResult(finalEvaluation);
    $("judge-placeholder").hidden = true;
    $("judge-results").hidden = false;
  } catch (error) {
    showToast(`Judge execution failed: ${formatError(error)}`, "error", 8000);
  } finally {
    setButtonLoading(button, false, "");
  }
}

async function searchKnowledgeBase(event) {
  event.preventDefault();
  const query = $("retrieval-query").value.trim();
  const topK = Number($("retrieval-limit").value);
  if (!query) return;

  const button = $("retrieval-submit");
  setButtonLoading(button, true, "Searching…");
  try {
    let data;
    try {
      data = await apiFetch("/api/kb/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query, top_k: topK })
      }, 120000);
    } catch (error) {
      if (!(error instanceof ApiError) || ![404, 405].includes(error.status)) throw error;
      data = await apiFetch("/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: query, ai_response: "Retrieval-only request", reference_answer: null, top_k: topK })
      }, 120000);
    }

    const chunks = data.results || data.retrieved_chunks || data.chunks || [];
    renderChunks($("retrieval-results"), chunks);
    $("retrieval-placeholder").hidden = true;
    $("retrieval-result-count").textContent = `${chunks.length} result${chunks.length === 1 ? "" : "s"}`;
  } catch (error) {
    showToast(`Search failed: ${formatError(error)}`, "error", 8000);
  } finally {
    setButtonLoading(button, false, "");
  }
}

function createSummaryCard(label, value, detail = "") {
  const card = createElement("div", "summary-card");
  card.append(createElement("strong", "", value ?? "N/A"), createElement("span", "", label));
  if (detail) {
    const small = createElement("small", "", detail);
    small.style.color = "var(--subtle)";
    small.style.marginTop = "5px";
    card.appendChild(small);
  }
  return card;
}

function percent(value) {
  const number = finiteNumber(value);
  return number === null ? "N/A" : `${Math.round(number * 100)}%`;
}

function scoreOutOfTen(value) {
  const number = finiteNumber(value);
  return number === null ? "N/A" : `${number.toFixed(2)}/10`;
}

function renderValidationCase(caseData, index) {
  const card = createElement("article", "case-card");
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.append(
    createElement("span", "", `Case ${index + 1}: ${caseData.question || caseData.id || "Untitled"}`),
    createBadge(caseData.expected_accurate === true ? "Expected correct" : caseData.expected_accurate === false ? "Expected incorrect" : "Details")
  );
  details.appendChild(summary);

  const body = createElement("div", "case-body");
  if (caseData.ai_response) body.appendChild(createElement("pre", "", `AI response:\n${caseData.ai_response}`));
  if (caseData.reference_answer) body.appendChild(createElement("pre", "", `Reference answer:\n${caseData.reference_answer}`));

  const fields = [
    ["Relevance", caseData.relevance_mean ?? caseData.relevance_score],
    ["Accuracy", caseData.accuracy_mean ?? caseData.accuracy_score],
    ["Grounding", caseData.hallucination_grounding_mean ?? caseData.hallucination_score],
    ["Completeness", caseData.completeness_mean ?? caseData.completeness_score]
  ];
  const metrics = createElement("div", "metric-grid");
  fields.forEach(([label, value]) => {
    const normalized = finiteNumber(value);
    metrics.appendChild(createMetricCard(label, normalized === null ? null : normalized / 10));
  });
  body.appendChild(metrics);

  const reasoningFields = [
    ["Relevance", caseData.relevance_reasoning],
    ["Accuracy", caseData.accuracy_reasoning],
    ["Hallucination", caseData.hallucination_reasoning]
  ].filter(([, value]) => value);
  reasoningFields.forEach(([label, value]) => body.appendChild(createElement("pre", "", `${label} reasoning:\n${value}`)));
  details.appendChild(body);
  card.appendChild(details);
  return card;
}

function renderNewValidation(data) {
  const discrimination = data.discrimination || {};
  const classification = data.classification_accuracy || {};
  const consistency = data.consistency || {};
  const reliability = data.reliability || {};
  const summaryCards = [
    createSummaryCard("Separation margin", scoreOutOfTen(discrimination.separation_margin), discrimination.passed ? "Discrimination passed" : "Check benchmark separation"),
    createSummaryCard("Accuracy classification", percent(classification.accuracy_judge)),
    createSummaryCard("Hallucination classification", percent(classification.hallucination_detector)),
    createSummaryCard("Judge failure rate", percent(reliability.failure_rate), consistency.measured ? `Average stdev: ${consistency.average_score_stdev ?? "N/A"}` : "Consistency needs repeated runs")
  ];
  $("validation-summary").replaceChildren(...summaryCards);
  return data.per_case || [];
}

function renderLegacyValidation(data) {
  const summary = data.summary || {};
  const check = data.consistency_check || {};
  const margin = finiteNumber(check.correct_answer_avg_accuracy) !== null && finiteNumber(check.wrong_answer_avg_accuracy) !== null
    ? Number(check.correct_answer_avg_accuracy) - Number(check.wrong_answer_avg_accuracy)
    : null;
  const summaryCards = [
    createSummaryCard("Average relevance", scoreOutOfTen(summary.relevance?.mean), `Stdev: ${summary.relevance?.stdev ?? "N/A"}`),
    createSummaryCard("Average accuracy", scoreOutOfTen(summary.accuracy?.mean), `Stdev: ${summary.accuracy?.stdev ?? "N/A"}`),
    createSummaryCard("Average grounding", scoreOutOfTen(summary.hallucination_grounding?.mean), `Stdev: ${summary.hallucination_grounding?.stdev ?? "N/A"}`),
    createSummaryCard("Correct/wrong margin", scoreOutOfTen(margin), check.judges_consistent ? "Discrimination passed" : "Discrimination failed")
  ];
  $("validation-summary").replaceChildren(...summaryCards);
  return data.per_pair || [];
}

function normalizeValidationData(rawData) {
  // Original backend format requires no conversion.
  if (
    rawData.summary &&
    rawData.consistency_check
  ) {
    return rawData;
  }

  const cases = rawData.per_case || [];

  function validNumbers(values) {
    return values
      .map(Number)
      .filter(Number.isFinite);
  }

  function mean(values) {
    const numbers = validNumbers(values);

    if (numbers.length === 0) {
      return null;
    }

    return Number(
      (
        numbers.reduce(
          (total, value) => total + value,
          0
        ) / numbers.length
      ).toFixed(2)
    );
  }

  function standardDeviation(values) {
    const numbers = validNumbers(values);

    if (numbers.length <= 1) {
      return 0;
    }

    const average = mean(numbers);

    const variance =
      numbers.reduce(
        (total, value) =>
          total + Math.pow(value - average, 2),
        0
      ) /
      (numbers.length - 1);

    return Number(
      Math.sqrt(variance).toFixed(2)
    );
  }

  function firstJudgeResult(
    caseData,
    judgeName
  ) {
    return (
      caseData.runs?.[0]?.[judgeName] || {}
    );
  }

  const relevanceScores = cases.map(
    item => item.relevance_mean
  );

  const accuracyScores = cases.map(
    item => item.accuracy_mean
  );

  const groundingScores = cases.map(
    item => item.hallucination_grounding_mean
  );

  const perPair = cases.map(caseData => {
    const relevance = firstJudgeResult(
      caseData,
      "relevance"
    );

    const accuracy = firstJudgeResult(
      caseData,
      "accuracy"
    );

    const hallucination = firstJudgeResult(
      caseData,
      "hallucination"
    );

    return {
      question: caseData.question || "",
      ai_response: caseData.ai_response || "",
      reference_answer:
        caseData.reference_answer || "",

      relevance_score:
        caseData.relevance_mean,
      relevance_verdict:
        relevance.verdict || "Unknown",
      relevance_reasoning:
        relevance.reasoning || "",

      accuracy_score:
        caseData.accuracy_mean,
      accuracy_verdict:
        accuracy.verdict || "Unknown",
      accuracy_reasoning:
        accuracy.reasoning || "",
      accuracy_evidence:
        accuracy.evidence || "",

      hallucination_detected:
        caseData.hallucination_detected,
      hallucination_score:
        caseData.hallucination_grounding_mean,
      hallucination_reasoning:
        hallucination.reasoning || "",
      flagged_statements:
        hallucination.flagged_statements || []
    };
  });

  const discrimination =
    rawData.discrimination || {};

  return {
    total_pairs:
      rawData.total_cases || cases.length,

    summary: {
      relevance: {
        mean: mean(relevanceScores),
        stdev: standardDeviation(
          relevanceScores
        )
      },

      accuracy: {
        mean: mean(accuracyScores),
        stdev: standardDeviation(
          accuracyScores
        )
      },

      hallucination_grounding: {
        mean: mean(groundingScores),
        stdev: standardDeviation(
          groundingScores
        )
      }
    },

    consistency_check: {
      correct_answer_avg_accuracy:
        discrimination.correct_answer_average,

      wrong_answer_avg_accuracy:
        discrimination.incorrect_answer_average,

      judges_consistent:
        Boolean(discrimination.passed)
    },

    per_pair: perPair
  };
}

async function runValidation() {
  const button = $("validation-button");
  const repetitions = Number($("validation-repetitions").value);
  const contextMode = $("validation-context").value;
  setButtonLoading(button, true, "Running…");
  $("validation-loading").hidden = false;
  $("validation-results").hidden = true;

  try {
    let data;
    try {
      data = await apiFetch("/api/validate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ repetitions, context_mode: contextMode })
      }, 900000);
    } catch (error) {
      if (!(error instanceof ApiError) || ![404, 405, 422].includes(error.status)) throw error;
      data = await apiFetch("/api/validate", { method: "GET" }, 900000);
    }

    const cases = data.discrimination || data.classification_accuracy
      ? renderNewValidation(data)
      : renderLegacyValidation(data);
    $("validation-case-count").textContent = `${cases.length} case${cases.length === 1 ? "" : "s"}`;
    $("validation-cases").replaceChildren(...cases.map(renderValidationCase));
    $("validation-results").hidden = false;
    showToast("Validation completed.", "success");
  } catch (error) {
    showToast(`Validation failed: ${formatError(error)}`, "error", 9000);
  } finally {
    $("validation-loading").hidden = true;
    setButtonLoading(button, false, "");
  }
}

function initialize() {
  if (window.location.protocol === "file:") $("file-protocol-warning").hidden = false;

  document.querySelectorAll(".nav-button").forEach((button) => {
    button.addEventListener("click", () => switchTab(button.dataset.tab));
  });

  $("mobile-menu-button").addEventListener("click", () => {
    $("sidebar").classList.contains("open") ? closeMobileMenu() : openMobileMenu();
  });
  $("sidebar-overlay").addEventListener("click", closeMobileMenu);

  $("benchmark-button").addEventListener("click", loadBenchmark);
  $("file-input").addEventListener("change", (event) => renderSelectedFiles(Array.from(event.target.files || [])));
  $("upload-button").addEventListener("click", uploadFiles);
  $("reset-button").addEventListener("click", resetKnowledgeBase);
  $("evaluation-form").addEventListener("submit", runEvaluation);
  $("judge-form").addEventListener("submit", runJudges);
  $("retrieval-form").addEventListener("submit", searchKnowledgeBase);
  $("validation-button").addEventListener("click", runValidation);

  $("evaluation-top-k").addEventListener("input", (event) => { $("evaluation-top-k-output").textContent = event.target.value; });
  $("judge-top-k").addEventListener("input", (event) => { $("judge-top-k-output").textContent = event.target.value; });

  switchTab("evaluate");
  fetchKnowledgeBaseStats();
}

document.addEventListener("DOMContentLoaded", initialize);
