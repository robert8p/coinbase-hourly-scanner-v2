const $ = (sel) => document.querySelector(sel);

function fmtPct(x) {
  if (x === null || x === undefined) return "—";
  const p = x * 100;
  const s = (Math.round(p * 100) / 100).toFixed(2) + "%";
  return p >= 0 ? "+" + s : s;
}
function fmtNum(x) {
  if (x === null || x === undefined) return "—";
  if (x >= 1e9) return (x/1e9).toFixed(2) + "B";
  if (x >= 1e6) return (x/1e6).toFixed(2) + "M";
  if (x >= 1e3) return (x/1e3).toFixed(2) + "K";
  return String(Math.round(x));
}
function fmtSpread(x) {
  if (x === null || x === undefined) return "—";
  return (x * 100).toFixed(3) + "%";
}
function fmtDec(x, dp=2) {
  if (x === null || x === undefined) return "—";
  return Number(x).toFixed(dp);
}
function fmtImpact(x) {
  if (x === null || x === undefined) return "—";
  return (x*100).toFixed(2) + "%";
}

async function refresh() {
  $("#btn").disabled = true;
  try {
    const r = await fetch("/api/opportunities?horizon=60&limit=10", { cache: "no-store" });
    const data = await r.json();
    $("#asof").textContent = data.asof || "—";
    $("#tracked").textContent = data.meta?.tracked_products ?? "—";
    $("#l2").textContent = data.meta?.level2_products ?? "—";
    $("#ws").textContent = (data.meta?.ws_connected ? "connected" : "disconnected");
    $("#ws").className = data.meta?.ws_connected ? "good" : "bad";
    $("#warm").textContent = data.meta?.warmup ?? "—";
    $("#regime").textContent = (data.meta?.regime_multiplier ?? "—");

    const tbody = $("#rows");
    tbody.innerHTML = "";
    for (const o of data.opportunities || []) {
      const tr = document.createElement("tr");
      const flags = (o.flags || []).map(f => `<span class="flag">${f}</span>`).join("");
      const drivers = (o.drivers || []).map(d => `<div>${d}</div>`).join("");

      tr.innerHTML = `
        <td class="mono">${o.product_id}</td>
        <td>${(o.price ?? "—")}</td>
        <td class="${(o.ret_15m ?? 0) >= 0 ? "good":"bad"}">${fmtPct(o.ret_15m)}</td>
        <td class="${(o.ret_60m ?? 0) >= 0 ? "good":"bad"}">${fmtPct(o.ret_60m)}</td>
        <td>${(o.vol_anom ?? null) ? (o.vol_anom.toFixed(2) + "×") : "—"}</td>
        <td>${fmtDec(o.tfi_5m, 2)}</td>
        <td>${fmtDec(o.obi_10bps, 2)}</td>
        <td>${fmtImpact(o.impact_cost)}</td>
        <td>${fmtSpread(o.spread_pct)}</td>
        <td>${fmtNum(o.quote_vol_usd_24h)}</td>
        <td>${(o.score ?? 0).toFixed(3)}<div style="margin-top:6px">${flags}</div></td>
        <td>${drivers}</td>
      `;
      tbody.appendChild(tr);
    }

    $("#msg").textContent = data.note || "";
  } catch (e) {
    $("#msg").textContent = "Error: " + (e?.message || e);
  } finally {
    $("#btn").disabled = false;
  }
}

$("#btn").addEventListener("click", refresh);
refresh();
setInterval(refresh, 15000);
