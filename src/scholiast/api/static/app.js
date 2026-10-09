/* The Scholiast interface (D37, D61, D154).
 *
 * No framework and no build step. The one place this matters technically is the
 * answer stream: D61 kept §5.5's POST contract and the SSE wire format, so the
 * response is read with `fetch` + a ReadableStream rather than `EventSource` —
 * which can only issue GET, and would have put the user's question in a URL.
 *
 * Errors are RFC 9457 problem documents, so `detail` is shown verbatim. That is
 * the whole point of NFR-U2's "actionable and specific": the server already
 * decided what the user can do about it, and the client should not paraphrase.
 *
 * SESSIONS (D154, closing D88). The access token lives only in memory; the
 * refresh token in sessionStorage, so a reload keeps the session and closing
 * the tab ends it. Every call carries the access token; a 401 triggers one
 * refresh and one retry, and a failed refresh returns to the login view.
 *
 * THIRD-PARTY TEXT (D154). Search results carry titles, abstracts, venues and
 * links written by whoever deposited a record at OpenAlex or arXiv. All of it is
 * rendered with `textContent`, never as markup, and a link is built only by
 * `safeLink`, which accepts http(s) alone. The page's Content-Security-Policy
 * refuses inline script as a second line, not the first. A pytest guard
 * (D155) holds both rules: `innerHTML` is only ever cleared, never filled.
 */

const REFRESH_KEY = "scholiast.refresh";
const session = { access: null };

const $ = (id) => document.getElementById(id);
const state = { workspaceId: null, conversationId: null, evidence: [] };

/* --- requests ---------------------------------------------------------- */

async function problem(response) {
  // Every failure this API produces is a problem document; anything else means
  // something upstream of the app failed, so say that rather than guessing.
  try {
    const body = await response.json();
    return new Error(body.detail || body.title || `HTTP ${response.status}`);
  } catch {
    return new Error(`HTTP ${response.status}`);
  }
}

function withAuth(init = {}) {
  const headers = new Headers(init.headers || {});
  if (session.access) headers.set("Authorization", `Bearer ${session.access}`);
  return { ...init, headers };
}

/* fetch with the session: one refresh and one retry on a 401 (D154). */
async function authFetch(path, init = {}) {
  let response = await fetch(path, withAuth(init));
  if (response.status !== 401) return response;
  if (!(await refreshSession())) {
    showLogin("Your session has ended. Please log in again.");
    throw new Error("Not logged in.");
  }
  response = await fetch(path, withAuth(init));
  return response;
}

const api = {
  async json(method, path, body) {
    const response = await authFetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) throw await problem(response);
    return response.status === 204 ? null : response.json();
  },
};

function toast(message, kind = "info") {
  const el = $("toast");
  el.textContent = message;
  el.dataset.kind = kind;
  el.hidden = false;
  setTimeout(() => (el.hidden = true), kind === "error" ? 6000 : 3000);
}

/* An element with text content only: the one way this file puts words on the page. */
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

/* A link to a third-party URL, or null. http(s) only, checked here even though the
 * adapters already drop anything else (D122): a page must not trust its inputs. */
function safeLink(url, text) {
  let parsed;
  try {
    parsed = new URL(url);
  } catch {
    return null;
  }
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") return null;
  const link = el("a", null, text);
  link.href = parsed.href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return link;
}

/* --- sessions (D154) --------------------------------------------------- */

function startSession(pair) {
  session.access = pair.access_token;
  try {
    sessionStorage.setItem(REFRESH_KEY, pair.refresh_token);
  } catch {
    // Storage refused (a private window, a policy): the session lasts until reload.
  }
}

function storedRefresh() {
  try {
    return sessionStorage.getItem(REFRESH_KEY);
  } catch {
    return null;
  }
}

function forgetSession() {
  session.access = null;
  try {
    sessionStorage.removeItem(REFRESH_KEY);
  } catch {
    // Nothing stored to forget.
  }
}

async function refreshSession() {
  const refresh = storedRefresh();
  if (!refresh) return false;
  const response = await fetch("/v1/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token: refresh }),
  });
  if (!response.ok) {
    forgetSession();
    return false;
  }
  startSession(await response.json());
  return true;
}

async function authenticate(path) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: $("email").value, password: $("password").value }),
  });
  if (!response.ok) throw await problem(response);
  startSession(await response.json());
  $("password").value = "";
  await enterApp();
}

async function logout() {
  const refresh = storedRefresh();
  forgetSession();
  if (refresh) {
    // Revoke server-side too, so the refresh token is dead, not just forgotten.
    await fetch("/v1/auth/logout", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refresh }),
    });
  }
  state.workspaceId = null;
  state.conversationId = null;
  showLogin("Logged out.");
}

function showLogin(message) {
  $("app").hidden = true;
  $("nav").hidden = true;
  $("auth").hidden = false;
  if (message) toast(message);
}

async function enterApp() {
  $("auth").hidden = true;
  $("nav").hidden = false;
  $("app").hidden = false;
  await loadWorkspaces();
}

function setView(view) {
  $("app").dataset.view = view;
  $("view-read").setAttribute("aria-pressed", String(view === "read"));
  $("view-discover").setAttribute("aria-pressed", String(view === "discover"));
}

/* --- workspaces ------------------------------------------------------- */

async function loadWorkspaces() {
  const workspaces = await api.json("GET", "/v1/workspaces");
  const select = $("workspace-select");
  select.innerHTML = "";
  for (const w of workspaces) {
    const option = document.createElement("option");
    option.value = w.id;
    option.textContent = `${w.name} (${w.queryable_count}/${w.paper_count} ready)`;
    select.append(option);
  }
  if (!workspaces.length) {
    await createWorkspace("My papers");
    return;
  }
  state.workspaceId = select.value = state.workspaceId ?? workspaces[0].id;
  await Promise.all([loadPapers(), openConversation()]);
}

async function createWorkspace(name) {
  const created = await api.json("POST", "/v1/workspaces", { name });
  state.workspaceId = created.id;
  state.conversationId = null;
  await loadWorkspaces();
}

/* --- papers ----------------------------------------------------------- */

async function loadPapers() {
  if (!state.workspaceId) return;
  const papers = await api.json("GET", `/v1/workspaces/${state.workspaceId}/papers`);
  const list = $("papers");
  list.innerHTML = "";

  for (const paper of papers) {
    const li = document.createElement("li");

    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "remove";
    remove.onclick = () => removePaper(paper.paper_id);
    li.append(remove);

    const title = document.createElement("span");
    title.className = "title";
    title.textContent = paper.title;
    li.append(title);

    const badge = document.createElement("span");
    badge.className = "state";
    badge.dataset.state = paper.state;
    badge.textContent = paper.state;
    li.append(badge);

    // NFR-U2: the actionable half of the failure taxonomy, not the diagnostic.
    if (paper.user_message) {
      const why = document.createElement("span");
      why.className = "why";
      why.textContent = paper.user_message;
      li.append(why);
    }
    list.append(li);
  }

  // NFR-U1: nothing spins without explanation. While anything is in flight the
  // list refreshes, so the user watches it move rather than wondering.
  const inFlight = papers.some(
    (p) => !["indexed", "failed", "unsupported"].includes(p.state)
  );
  if (inFlight) setTimeout(guard(loadPapers), 1500);
  else await refreshWorkspaceLabels();
}

/* The selector's "n/m ready" counts, refreshed once ingestion settles, so they do
 * not stay at what they were when the page loaded. The selection is kept. */
async function refreshWorkspaceLabels() {
  const workspaces = await api.json("GET", "/v1/workspaces");
  for (const option of $("workspace-select").options) {
    const w = workspaces.find((x) => x.id === option.value);
    if (w) option.textContent = `${w.name} (${w.queryable_count}/${w.paper_count} ready)`;
  }
}

async function uploadPaper(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await authFetch(`/v1/workspaces/${state.workspaceId}/papers/upload`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) throw await problem(response);
  const accepted = await response.json();
  toast(accepted.deduplicated ? "Already ingested — reused." : "Queued.");
  await loadPapers();
}

async function removePaper(paperId) {
  const result = await api.json(
    "DELETE",
    `/v1/workspaces/${state.workspaceId}/papers/${paperId}`
  );
  // FR-W2: say which of the two things happened, because "removed" and
  // "removed and deleted everywhere" are different facts.
  toast(result.purged ? "Removed and purged." : "Removed from this workspace.");
  await loadPapers();
}

/* --- chat ------------------------------------------------------------- */

async function openConversation() {
  if (state.conversationId) return;
  const created = await api.json(
    "POST",
    `/v1/workspaces/${state.workspaceId}/conversations`,
    { title: null }
  );
  state.conversationId = created.conversation_id;
}

function addTurn(who, text) {
  const turn = document.createElement("div");
  turn.className = "turn";
  const label = document.createElement("div");
  label.className = "who";
  label.textContent = who;
  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;
  turn.append(label, body);
  $("transcript").append(turn);
  $("transcript").scrollTop = $("transcript").scrollHeight;
  return body;
}

async function ask(question) {
  await openConversation();
  addTurn("You", question);
  const answerEl = addTurn("Scholiast", "");
  $("disclosure").hidden = true;
  state.evidence = [];

  const response = await authFetch(
    `/v1/workspaces/${state.workspaceId}/conversations/${state.conversationId}/messages`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    }
  );
  // Validation ran before the stream opened (phase 6's C-3), so an unqueryable
  // workspace is a 409 problem document here rather than an error mid-body.
  if (!response.ok) throw await problem(response);

  for await (const event of sseFrames(response)) {
    if (event.name === "evidence") {
      state.evidence = event.data.items;
      renderEvidence(event.data);
    } else if (event.name === "token") {
      answerEl.textContent += event.data.text;
      $("transcript").scrollTop = $("transcript").scrollHeight;
    } else if (event.name === "complete") {
      renderAnswer(answerEl, event.data);
      renderDisclosure(event.data);
    }
  }
}

/* Minimal SSE parser over a fetch body (D61). Frames are separated by a blank
 * line; a frame's `data:` is one line because the server serialises JSON with no
 * embedded newlines. */
async function* sseFrames(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let split;
    while ((split = buffer.indexOf("\n\n")) !== -1) {
      const frame = buffer.slice(0, split);
      buffer = buffer.slice(split + 2);
      let name = "message";
      let data = "";
      for (const line of frame.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { name, data: JSON.parse(data) };
    }
  }
}

/* --- rendering -------------------------------------------------------- */

function renderAnswer(element, complete) {
  // Re-rendered once at the end rather than per token: turning [C1] into a
  // button needs the whole marker, and a marker can straddle two tokens.
  element.textContent = "";
  const pattern = /\[(C\d+)\]/g;
  let cursor = 0;
  let match;
  while ((match = pattern.exec(complete.answer)) !== null) {
    element.append(complete.answer.slice(cursor, match.index));
    const button = document.createElement("button");
    button.className = "cite";
    button.textContent = match[0];
    const label = match[1];
    button.onclick = guard(() => openReader(label));
    element.append(button);
    cursor = match.index + match[0].length;
  }
  element.append(complete.answer.slice(cursor));
}

function renderEvidence(payload) {
  const list = $("evidence");
  list.innerHTML = "";
  for (const item of payload.items) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = `[${item.label}] `;
    const where = document.createElement("span");
    where.className = "where";
    where.textContent = `${item.paper_title} · ${item.section_path} · p.${item.page}`;
    const snippet = document.createElement("span");
    snippet.className = "snippet";
    snippet.textContent = item.text;
    li.append(label, where, snippet);

    // §6.10 rule 5, surfaced: a related-work passage reports someone else's
    // contribution, and the reader should see that before trusting the claim.
    if (item.describes_other_work) {
      const note = document.createElement("span");
      note.className = "other-work";
      note.textContent = "describes other work";
      li.append(note);
    }
    li.className = "clickable";
    li.onclick = guard(() => openReader(item.label));
    list.append(li);
  }
}

function renderDisclosure(complete) {
  // NFR-U4 and §5.11: a degraded answer that does not say so is worse than an
  // error, because it launders reduced quality as normal output.
  const notes = [];
  for (const flag of complete.degraded) {
    notes.push(
      {
        provider_failover: "The primary model was unavailable; a fallback answered.",
        generation_truncated: "The model stopped mid-answer. This answer is incomplete.",
        generation_unavailable:
          "No model was available, so this is the retrieved evidence without synthesis.",
      }[flag] ?? flag
    );
  }
  if (complete.fabricated_labels.length) {
    notes.push(
      `The model cited ${complete.fabricated_labels.join(", ")}, which do not exist. ` +
        "Those citations were removed."
    );
  }
  const disclosure = $("disclosure");
  disclosure.textContent = notes.join(" ");
  disclosure.hidden = notes.length === 0;
}

async function openReader(label) {
  const item = state.evidence.find((e) => e.label === label);
  if (!item) return;

  const content = await api.json(
    "GET",
    `/v1/workspaces/${state.workspaceId}/papers/${item.paper_id}/content`
  );
  const section = content.sections.find((s) => s.path === item.section_path);
  $("reader-title").textContent = content.title;
  $("reader-meta").textContent = `${item.section_path} · page ${item.page}`;

  const text = $("reader-text");
  text.textContent = "";
  if (!section) {
    text.textContent = item.text;
  } else {
    // NFR-U3: the exact span, not the neighbourhood. These offsets index
    // `sections.text` (D42), which is precisely what this endpoint returns.
    text.append(section.text.slice(0, item.char_start));
    const mark = document.createElement("mark");
    mark.textContent = section.text.slice(item.char_start, item.char_end);
    text.append(mark, section.text.slice(item.char_end));
  }
  $("reader").showModal();
  document.querySelector("#reader-text mark")?.scrollIntoView({ block: "center" });
}

/* --- discover (FR-D1..D8) ---------------------------------------------- */

const SOURCE_STATUS = {
  timeout: "did not answer in time",
  rate_limited: "is limiting requests",
  error: "failed",
  circuit_open: "is paused after repeated failures",
};

function numberOrNull(id) {
  const value = $(id).value.trim();
  return value === "" ? null : Number(value);
}

async function search() {
  const body = { query: $("query").value.trim(), sort: $("sort").value };
  const yearFrom = numberOrNull("year-from");
  const yearTo = numberOrNull("year-to");
  const minCitations = numberOrNull("min-citations");
  const venue = $("venue").value.trim();
  if (yearFrom !== null) body.year_from = yearFrom;
  if (yearTo !== null) body.year_to = yearTo;
  if (minCitations !== null) body.min_citations = minCitations;
  if (venue) body.venue = venue;

  $("results").innerHTML = "";
  $("ranking-note").hidden = true;
  showSearchStatus(["Searching OpenAlex and arXiv…"]);
  const result = await api.json("POST", "/v1/discovery/search", body);
  renderResults(result);
}

function showSearchStatus(lines, kind = "info") {
  const status = $("search-status");
  status.innerHTML = "";
  for (const line of lines) status.append(el("p", null, line));
  status.dataset.kind = kind;
  status.hidden = lines.length === 0;
}

function renderResults(result) {
  const lines = [];
  // FR-D6: name the sources that did not answer, and when to try again.
  const down = result.source_status.filter((s) => s.status !== "ok");
  for (const s of down) {
    const when = s.retry_after_seconds ? ` Try again in ${Math.ceil(s.retry_after_seconds)} s.` : "";
    lines.push(`${s.source} ${SOURCE_STATUS[s.status] ?? s.status}; showing what the others found.${when}`);
  }
  const hidden = result.excluded;
  const hiddenTotal = hidden.year + hidden.citations + hidden.venue;
  if (hiddenTotal) {
    const parts = [];
    if (hidden.year) parts.push(`${hidden.year} by year`);
    if (hidden.citations) {
      const unknown = hidden.unknown_citations
        ? ` (${hidden.unknown_citations} because no citation count is reported)`
        : "";
      parts.push(`${hidden.citations} by citations${unknown}`);
    }
    if (hidden.venue) parts.push(`${hidden.venue} by venue`);
    lines.push(`Filters hid ${hiddenTotal}: ${parts.join(", ")}.`);
  }
  lines.push(
    result.results.length
      ? `${result.results.length} papers.`
      : hiddenTotal
        ? "Every paper found was hidden by the filters."
        : "The sources found nothing for this query."
  );
  showSearchStatus(lines, result.partial ? "warn" : "info");

  const list = $("results");
  list.innerHTML = "";
  for (const work of result.results) list.append(renderWork(work, result.ranking));

  // FR-D5: what the scores below mean, once, at the foot of the list.
  const r = result.ranking;
  const signal = r.relevance === "dense" ? "by meaning, dense embeddings" : "by shared words, BM25";
  $("ranking-note").textContent =
    `Ranked by ${r.weights.relevance} × relevance (${signal}) + ${r.weights.citation_impact} × ` +
    `citation impact + ${r.weights.recency} × recency, weights fitted on judged searches (${r.version}).`;
  $("ranking-note").hidden = result.results.length === 0;
}

function renderWork(work, ranking) {
  const li = el("li", "work");

  const head = el("div", "head");
  head.append(el("span", "title", work.title));
  const add = el("button", "add", "Add to workspace");
  add.type = "button";
  add.onclick = guard(async () => {
    add.disabled = true;
    try {
      await addPaper(work.add);
      add.textContent = "Added";
    } catch (error) {
      add.disabled = false;
      throw error;
    }
  });
  head.append(add);
  li.append(head);

  const authors = work.authors.length > 3 ? `${work.authors.slice(0, 3).join(", ")} et al.` : work.authors.join(", ");
  const facts = [
    authors || "authors not reported",
    work.year ?? "year not reported",
    work.venue || null,
    work.citation_count === null ? "citations not reported" : `${work.citation_count} citations`,
  ].filter((f) => f !== null);
  li.append(el("p", "meta", facts.join(" · ")));

  // FR-D3: one paper, however many sources returned it.
  const sources = [...new Set(work.members.map((m) => m.source))];
  const found = el("p", "sources", `Found in ${sources.join(" and ")}`);
  for (const link of work.open_access) {
    // Labelled by host, so three links do not read as "PDF · PDF · PDF".
    const a = safeLink(link.url, link.kind === "pdf" ? "PDF" : "page");
    if (a) {
      a.textContent = `${new URL(a.href).hostname} ${a.textContent}`;
      found.append(" · ", a);
    }
  }
  li.append(found);

  if (work.abstract) li.append(el("p", "abstract", work.abstract));
  li.append(renderWhy(work, ranking));
  return li;
}

/* FR-D5: why this paper ranked here. Each part, its weight, and what it added. */
function renderWhy(work, ranking) {
  const details = el("details", "why-here");
  details.append(el("summary", null, `Why here? Score ${work.score.toFixed(2)}`));
  const parts = work.score_parts;
  const rows = [
    ["Relevance to your query", parts.relevance, ranking.weights.relevance, false],
    [
      "Citation impact (per year since it appeared)",
      parts.citation_impact,
      ranking.weights.citation_impact,
      parts.imputed.includes("citation_impact"),
    ],
    ["Recency", parts.recency, ranking.weights.recency, parts.imputed.includes("recency")],
  ];
  const table = el("table");
  const header = el("tr");
  for (const h of ["Part", "Value", "Weight", "Adds"]) header.append(el("th", null, h));
  table.append(header);
  for (const [name, value, weight, imputed] of rows) {
    const tr = el("tr");
    const label = el("td", null, name);
    if (imputed) label.append(el("span", "imputed", " not reported, typical value used"));
    const bar = el("td", "value");
    const meter = el("meter");
    meter.min = 0;
    meter.max = 1;
    meter.value = value;
    bar.append(meter, ` ${value.toFixed(2)}`);
    tr.append(label, bar, el("td", null, weight.toFixed(1)), el("td", null, (value * weight).toFixed(2)));
    table.append(tr);
  }
  details.append(table);
  return details;
}

async function addPaper(ref) {
  if (!state.workspaceId) throw new Error("Choose a workspace first.");
  const accepted = await api.json("POST", `/v1/workspaces/${state.workspaceId}/papers`, {
    papers: [{ source: ref.source, source_id: ref.source_id }],
  });
  const [paper] = accepted.papers;
  toast(paper.deduplicated ? "Already in your library — reused." : "Added. Fetching the PDF…");
  await loadPapers();
}

/* --- wiring ----------------------------------------------------------- */

function guard(fn) {
  return async (event) => {
    event?.preventDefault?.();
    try {
      await fn(event);
    } catch (error) {
      toast(error.message, "error");
    }
  };
}

$("workspace-select").onchange = guard(async (event) => {
  state.workspaceId = event.target.value;
  state.conversationId = null;
  $("transcript").innerHTML = "";
  $("evidence").innerHTML = "";
  await Promise.all([loadPapers(), openConversation()]);
});

$("new-workspace").onclick = guard(async () => {
  const name = prompt("Workspace name");
  if (name) await createWorkspace(name);
});

$("upload-form").onsubmit = guard(async () => {
  const input = $("pdf");
  if (input.files[0]) {
    await uploadPaper(input.files[0]);
    input.value = "";
  }
});

$("ask-form").onsubmit = guard(async () => {
  const input = $("question");
  const question = input.value.trim();
  if (!question) return;
  input.value = "";
  await ask(question);
});

$("search-form").onsubmit = guard(search);
$("auth-form").onsubmit = guard(() => authenticate("/v1/auth/login"));
$("register").onclick = guard(() => authenticate("/v1/auth/register"));
$("logout").onclick = guard(logout);
$("view-read").onclick = () => setView("read");
$("view-discover").onclick = () => setView("discover");

guard(async () => {
  // A reload with a live refresh token goes straight back in (D154).
  if (await refreshSession()) await enterApp();
  else showLogin();
})();
