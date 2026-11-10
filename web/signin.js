// Staff sign-in: the staff code for a token. The server keeps the token as an HttpOnly cookie, so nothing here
// stores it; on success the browser goes back to the page it was sent here from.

const $ = (id) => document.getElementById(id);

// only a path on this server, never another site
function next() {
  const to = new URLSearchParams(location.search).get("next") || "/";
  return to.startsWith("/") && !to.startsWith("//") ? to : "/";
}

$("signin").onsubmit = async (e) => {
  e.preventDefault();
  const code = $("code").value.trim();
  if (!code) return show("Enter the staff code.");
  $("go").disabled = true;
  try {
    const r = await fetch("/api/staff/signin", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ code }) });
    if (r.ok) return location.replace(next());
    show((await r.json().catch(() => ({}))).error || "Could not sign in.");
  } catch {
    show("The server could not be reached.");
  } finally {
    $("go").disabled = false;
  }
};

function show(text) {
  $("error").textContent = text;
  $("error").hidden = false;
  $("code").select();
}
