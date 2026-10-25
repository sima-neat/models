// The Developer Portal owns the global navigation; Models supplies its section navigation.
let shellAssets;

function loadAsset(tag, attribute, url) {
  return new Promise((resolve, reject) => {
    const asset = document.createElement(tag);
    if (tag === "link") asset.rel = "stylesheet";
    asset[attribute] = url;
    asset.onload = resolve;
    asset.onerror = reject;
    document.head.append(asset);
  });
}

async function loadShell() {
  // Standalone preview servers may return their HTML page for a missing script.
  const response = await fetch("/developer-center-shell.js");
  if (!response.ok || !/(java|ecma)script/i.test(response.headers.get("content-type") || "")) return false;
  await Promise.all([
    loadAsset("link", "href", "/developer-center-shell.css"),
    window.DeveloperCenterShell || loadAsset("script", "src", "/developer-center-shell.js"),
  ]);
  return Boolean(window.DeveloperCenterShell);
}

export async function mountDeveloperPortal() {
  try {
    shellAssets ||= loadShell();
    if (!await shellAssets) return;
    let root = document.getElementById("developer-center-shell");
    if (!root) {
      root = document.createElement("div");
      root.id = "developer-center-shell";
      document.body.prepend(root);
    }
    if (root.childElementCount) return;
    await window.DeveloperCenterShell.mount(root, { active: "models" });
    document.documentElement.classList.add("developer-center-shell-enabled");
  } catch {
    // Standalone builds keep their local navigation when the shared shell is unavailable.
  }
}
