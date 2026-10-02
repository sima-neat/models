const SAFE_LINK = /^(?:https?:|mailto:)/i;

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (character) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
  })[character]);
}

export function resolveDocumentationLink(documentation, source) {
  const commit = documentation?.source_commit;
  const sourcePath = documentation?.source_path;
  if (
    !/^[0-9a-f]{40}$/.test(commit || "")
    || !/^models\/[a-z][a-z0-9_]*\/README\.md$/.test(sourcePath || "")
    || /[\\\u0000-\u001f]/.test(source || "")
  ) return null;
  const encodedPath = sourcePath.split("/").map(encodeURIComponent).join("/");
  const prefix = `/sima-neat/models/blob/${commit}/`;
  try {
    const resolved = new URL(source, `https://github.com${prefix}${encodedPath}`);
    return resolved.origin === "https://github.com" && resolved.pathname.startsWith(prefix) ? resolved.href : null;
  } catch {
    return null;
  }
}

function inlineMarkdown(value, resolveLink = null) {
  const source = String(value);
  const expression = /(`[^`\n]+`|\[([^\]\n]+)\]\(([^)\s]+)\)|\*\*([^*\n]+)\*\*|\*([^*\n]+)\*)/g;
  let output = "";
  let offset = 0;
  for (const match of source.matchAll(expression)) {
    output += escapeHtml(source.slice(offset, match.index));
    if (match[0].startsWith("`")) output += `<code>${escapeHtml(match[0].slice(1, -1))}</code>`;
    else if (match[2] !== undefined) {
      const resolved = SAFE_LINK.test(match[3])
        ? match[3]
        : (typeof resolveLink === "function" ? resolveLink(match[3]) : null);
      const href = resolved && SAFE_LINK.test(resolved) ? resolved : null;
      output += href
        ? `<a href="${escapeHtml(href)}" target="_blank" rel="noreferrer">${escapeHtml(match[2])}</a>`
        : escapeHtml(match[2]);
    } else if (match[4] !== undefined) output += `<strong>${escapeHtml(match[4])}</strong>`;
    else output += `<em>${escapeHtml(match[5])}</em>`;
    offset = match.index + match[0].length;
  }
  return output + escapeHtml(source.slice(offset));
}

function tableRow(line) {
  return line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim());
}

function isTableDivider(line, columns) {
  const cells = tableRow(line);
  return cells.length === columns && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
}

export function renderMarkdown(markdown, { stripTitle = false, resolveImage = null, resolveLink = null } = {}) {
  let lines = String(markdown || "").replace(/\r\n?/g, "\n").split("\n");
  if (stripTitle && /^#\s+/.test(lines[0] || "")) lines = lines.slice(1);
  const output = [];
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) { index += 1; continue; }
    const image = /^!\[([^\]\n]+)\]\(([^)\s]+)\)$/.exec(line.trim());
    if (image) {
      const source = typeof resolveImage === "function" ? resolveImage(image[2]) : null;
      output.push(source
        ? `<figure class="model-card-hero"><img src="${escapeHtml(source)}" alt="${escapeHtml(image[1])}" loading="lazy" decoding="async" /></figure>`
        : `<p>${escapeHtml(image[1])}</p>`);
      index += 1;
      continue;
    }
    const fence = /^```([A-Za-z0-9_-]*)\s*$/.exec(line);
    if (fence) {
      const code = [];
      index += 1;
      while (index < lines.length && !/^```\s*$/.test(lines[index])) code.push(lines[index++]);
      if (index < lines.length) index += 1;
      const language = fence[1] ? ` class="language-${escapeHtml(fence[1])}"` : "";
      output.push(`<pre><code${language}>${escapeHtml(code.join("\n"))}</code></pre>`);
      continue;
    }
    const heading = /^(#{2,4})\s+(.+)$/.exec(line);
    if (heading) {
      const level = heading[1].length + 1;
      output.push(`<h${level}>${inlineMarkdown(heading[2], resolveLink)}</h${level}>`);
      index += 1;
      continue;
    }
    if (index + 1 < lines.length && line.includes("|")) {
      const headers = tableRow(line);
      if (headers.length > 1 && isTableDivider(lines[index + 1], headers.length)) {
        index += 2;
        const rows = [];
        while (index < lines.length && lines[index].includes("|") && lines[index].trim()) {
          const cells = tableRow(lines[index]);
          if (cells.length !== headers.length) break;
          rows.push(cells);
          index += 1;
        }
        output.push(`<div class="model-card-table"><table><thead><tr>${headers.map((cell) => `<th scope="col">${inlineMarkdown(cell, resolveLink)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell, resolveLink)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
    }
    const list = /^\s*(?:([-*+])|(\d+)[.)])\s+(.+)$/.exec(line);
    if (list) {
      const ordered = Boolean(list[2]);
      const items = [];
      while (index < lines.length) {
        const item = /^\s*(?:([-*+])|(\d+)[.)])\s+(.+)$/.exec(lines[index]);
        if (!item || Boolean(item[2]) !== ordered) break;
        items.push(`<li>${inlineMarkdown(item[3], resolveLink)}</li>`);
        index += 1;
      }
      output.push(`<${ordered ? "ol" : "ul"}>${items.join("")}</${ordered ? "ol" : "ul"}>`);
      continue;
    }
    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && lines[index].trim()
      && !/^(?:#{2,4}\s+|```|!\[|\s*(?:[-*+] |\d+[.)] ))/.test(lines[index])) {
      if (lines[index].includes("|") && index + 1 < lines.length
        && isTableDivider(lines[index + 1], tableRow(lines[index]).length)) break;
      paragraph.push(lines[index].trim());
      index += 1;
    }
    output.push(`<p>${inlineMarkdown(paragraph.join(" "), resolveLink)}</p>`);
  }
  return output.join("\n");
}
