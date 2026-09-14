/* The M1 interface (D37, D61).
 *
 * No framework and no build step. The one place this matters technically is the
 * answer stream: D61 kept §5.5's POST contract and the SSE wire format, so the
 * response is read with `fetch` + a ReadableStream rather than `EventSource` —
 * which can only issue GET, and would have put the user's question in a URL.
 *
 * Errors are RFC 9457 problem documents, so `detail` is shown verbatim. That is
 * the whole point of NFR-U2's "actionable and specific": the server already
 * decided what the user can do about it, and the client should not paraphrase.
 */

const api = {
  async json(method, path, body) {
    const response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : {},
      body: body ? JSON.stringify(body) : undefined,
    });
    if (!response.ok) throw await problem(response);
    return response.status === 204 ? null : response.json();
  },
};

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

const $ = (id) => document.getElementById(id);
const state = { workspaceId: null, conversationId: null, evidence: [] };

function toast(message, kind = "info") {
  const el = $("toast");
  el.textContent = message;
  el.dataset.kind = kind;
  el.hidden = false;
  setTimeout(() => (el.hidden = true), kind === "error" ? 6000 : 3000);
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
  if (inFlight) setTimeout(loadPapers, 1500);
}

async function uploadPaper(file) {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(
    `/v1/workspaces/${state.workspaceId}/papers/upload`,
    { method: "POST", body: form }
  );
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

  const response = await fetch(
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
    button.onclick = () => openReader(match[1]);
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
    li.style.cursor = "pointer";
    li.onclick = () => openReader(item.label);
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
  const el = $("disclosure");
  el.textContent = notes.join(" ");
  el.hidden = notes.length === 0;
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

/* --- wiring ----------------------------------------------------------- */

const guard = (fn) => async (event) => {
  event?.preventDefault?.();
  try {
    await fn(event);
  } catch (error) {
    toast(error.message, "error");
  }
};

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

guard(loadWorkspaces)();
