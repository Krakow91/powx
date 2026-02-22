const byId = (id) => document.getElementById(id);

const state = {
  limit: 120,
  offset: 0,
  owner: "",
  search: "",
  listed: "",
  nfts: [],
  listings: [],
};

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const data = await response.json();
  if (!response.ok || !data.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function shortText(value, max = 24) {
  const text = String(value || "");
  if (text.length <= max) return text;
  return `${text.slice(0, max)}...`;
}

function setError(message) {
  byId("errorLine").textContent = message || "";
}

function writeLog(payload) {
  byId("actionOutput").value = JSON.stringify(payload, null, 2);
}

function formatTime(epoch) {
  const n = Number(epoch || 0);
  if (!n) return "never";
  const date = new Date(n * 1000);
  return date.toLocaleString();
}

function requireSigner() {
  const privateKey = byId("signerPrivateKey").value.trim().toLowerCase();
  if (!privateKey) {
    throw new Error("Private key is required");
  }
  if (privateKey.length !== 64 || !/^[0-9a-f]+$/.test(privateKey)) {
    throw new Error("Private key must be 64 hex chars");
  }

  const feeRaw = byId("signerFee").value.trim();
  const ttlRaw = byId("signerTtl").value.trim();
  const fee = feeRaw ? Number(feeRaw) : 2;
  const broadcastTtl = ttlRaw ? Number(ttlRaw) : 2;
  if (!Number.isFinite(fee) || fee < 0) {
    throw new Error("Fee must be a non-negative number");
  }
  if (!Number.isFinite(broadcastTtl) || broadcastTtl < 0) {
    throw new Error("Broadcast TTL must be a non-negative number");
  }

  return {
    privateKeyHex: privateKey,
    fee: Math.trunc(fee),
    broadcastTtl: Math.trunc(broadcastTtl),
  };
}

async function submitContract(contractPayload, opts = {}) {
  const signer = requireSigner();
  const payload = {
    private_key_hex: signer.privateKeyHex,
    contract: contractPayload,
    fee: signer.fee,
    broadcast_ttl: signer.broadcastTtl,
  };

  if (opts.toAddress) payload.to_address = opts.toAddress;
  if (opts.amount !== undefined && opts.amount !== null && opts.amount !== "") {
    payload.amount = Math.trunc(Number(opts.amount));
  }

  const result = await api("/api/v1/market/submit-contract", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  writeLog(result);
  await refreshAll();
  return result;
}

function fillListForm(tokenId) {
  byId("listTokenId").value = tokenId || "";
  byId("cancelTokenId").value = tokenId || "";
}

function fillBuyForm(tokenId, seller, price) {
  byId("buyTokenId").value = tokenId || "";
  byId("buySeller").value = seller || "";
  byId("buyPrice").value = price !== undefined && price !== null ? String(price) : "";
}

function renderNfts(rows) {
  const root = byId("nftGrid");
  state.nfts = rows;
  if (!rows.length) {
    root.innerHTML = '<p class="muted">No NFTs found.</p>';
    return;
  }
  root.innerHTML = rows
    .map((row) => {
      const listing = row.listing;
      const listingBadge = listing
        ? `<span class="badge listed">Listed ${listing.price} KK91</span>`
        : '<span class="badge">Unlisted</span>';
      const actionButton = listing
        ? `<button class="btn btn-ghost tiny" data-action="prefill-buy" data-token="${row.token_id}" data-seller="${listing.seller}" data-price="${listing.price}">Buy setup</button>`
        : `<button class="btn btn-ghost tiny" data-action="prefill-list" data-token="${row.token_id}">List setup</button>`;
      return `
        <article class="nft-card">
          <div class="nft-top">
            <strong>${row.token_id}</strong>
            ${listingBadge}
          </div>
          <p class="small">Owner: ${shortText(row.owner, 32)}</p>
          <p class="small">Creator: ${shortText(row.creator, 32)}</p>
          <p class="small">Metadata: ${shortText(row.metadata_uri, 44)}</p>
          <p class="small">Updated height: ${row.updated_height}</p>
          <div class="card-actions">
            ${actionButton}
            <button class="btn btn-ghost tiny" data-action="prefill-cancel" data-token="${row.token_id}">Cancel setup</button>
          </div>
        </article>
      `;
    })
    .join("");
}

function renderListings(rows) {
  const root = byId("listingTable");
  state.listings = rows;
  if (!rows.length) {
    root.innerHTML = '<p class="muted">No active listings.</p>';
    return;
  }
  root.innerHTML = rows
    .map(
      (row) => `
      <div class="row">
        <strong>${row.token_id}</strong>
        <p class="small">Price: ${row.price} KK91</p>
        <p class="small">Seller: ${shortText(row.seller, 40)}</p>
        <p class="small">Owner now: ${shortText(row.current_owner, 40)}</p>
        <div class="card-actions">
          <button class="btn btn-ghost tiny" data-action="prefill-buy" data-token="${row.token_id}" data-seller="${row.seller}" data-price="${row.price}">Buy setup</button>
          <button class="btn btn-ghost tiny" data-action="prefill-cancel" data-token="${row.token_id}">Cancel setup</button>
        </div>
      </div>
    `,
    )
    .join("");
}

function renderContracts(rows) {
  const root = byId("contractTable");
  if (!rows.length) {
    root.innerHTML = '<p class="muted">No contracts found.</p>';
    return;
  }
  root.innerHTML = rows
    .map(
      (row) => `
      <div class="row">
        <strong>${row.contract_id}</strong>
        <p class="small">Template: ${row.template}</p>
        <p class="small">Owner: ${shortText(row.owner, 40)}</p>
        <p class="small">State keys: ${Object.keys(row.state || {}).length}</p>
      </div>
    `,
    )
    .join("");
}

async function refreshStatus() {
  const status = await api("/api/v1/market/status");
  byId("nodeUrl").textContent = status.node_url || "-";
  byId("syncedHeight").textContent = String(status.indexer?.last_sync_height ?? "-");
  byId("nftCount").textContent = String(status.indexer?.nft_count ?? 0);
  byId("listingCount").textContent = String(status.indexer?.listing_count ?? 0);
  byId("contractCount").textContent = String(status.indexer?.contract_count ?? 0);
  byId("lastSync").textContent = formatTime(status.last_sync_epoch || status.indexer?.last_sync_epoch || 0);
}

async function refreshNfts() {
  const params = new URLSearchParams();
  params.set("limit", String(state.limit));
  params.set("offset", String(state.offset));
  if (state.owner) params.set("owner", state.owner);
  if (state.search) params.set("search", state.search);
  if (state.listed) params.set("listed", state.listed);

  const data = await api(`/api/v1/market/nfts?${params.toString()}`);
  renderNfts(data.rows || []);
}

async function refreshListings() {
  const data = await api("/api/v1/market/listings?limit=60");
  renderListings(data.rows || []);
}

async function refreshContracts() {
  const data = await api("/api/v1/market/contracts?limit=40");
  renderContracts(data.rows || []);
}

async function refreshAll() {
  setError("");
  try {
    await Promise.all([refreshStatus(), refreshNfts(), refreshListings(), refreshContracts()]);
  } catch (error) {
    setError(error.message || String(error));
  }
}

async function triggerSync() {
  setError("");
  try {
    await api("/api/v1/market/sync", { method: "POST", body: "{}" });
    await refreshAll();
  } catch (error) {
    setError(error.message || String(error));
  }
}

async function mintNft() {
  setError("");
  try {
    const tokenId = byId("mintTokenId").value.trim();
    const metadataUri = byId("mintMetadata").value.trim();
    const owner = byId("mintOwner").value.trim();
    if (!tokenId) throw new Error("Token ID is required");
    if (!metadataUri) throw new Error("Metadata URI is required");

    const contract = {
      kind: "nft",
      action: "mint",
      token_id: tokenId,
      metadata_uri: metadataUri,
    };
    if (owner) contract.owner = owner;

    await submitContract(contract);
  } catch (error) {
    setError(error.message || String(error));
  }
}

async function listNft() {
  setError("");
  try {
    const tokenId = byId("listTokenId").value.trim();
    const priceRaw = byId("listPrice").value.trim();
    if (!tokenId) throw new Error("Token ID is required");
    if (!priceRaw) throw new Error("Price is required");

    const price = Math.trunc(Number(priceRaw));
    if (!Number.isFinite(price) || price <= 0) throw new Error("Price must be > 0");

    await submitContract({
      kind: "nft",
      action: "list",
      token_id: tokenId,
      price,
    });
  } catch (error) {
    setError(error.message || String(error));
  }
}

async function cancelListing() {
  setError("");
  try {
    const tokenId = byId("cancelTokenId").value.trim();
    if (!tokenId) throw new Error("Token ID is required");

    await submitContract({
      kind: "nft",
      action: "cancel",
      token_id: tokenId,
    });
  } catch (error) {
    setError(error.message || String(error));
  }
}

async function buyNft() {
  setError("");
  try {
    const tokenId = byId("buyTokenId").value.trim();
    const seller = byId("buySeller").value.trim();
    const priceRaw = byId("buyPrice").value.trim();
    if (!tokenId) throw new Error("Token ID is required");
    if (!seller) throw new Error("Seller address is required");
    if (!priceRaw) throw new Error("Price is required");

    const price = Math.trunc(Number(priceRaw));
    if (!Number.isFinite(price) || price <= 0) throw new Error("Price must be > 0");

    await submitContract(
      {
        kind: "nft",
        action: "buy",
        token_id: tokenId,
      },
      {
        toAddress: seller,
        amount: price,
      },
    );
  } catch (error) {
    setError(error.message || String(error));
  }
}

async function buildUnsignedTx() {
  setError("");
  try {
    const senderPubkey = byId("senderPubkey").value.trim();
    const payloadRaw = byId("contractPayload").value.trim();
    if (!senderPubkey) {
      throw new Error("sender pubkey is required");
    }
    if (!payloadRaw) {
      throw new Error("contract payload is required");
    }

    const payload = {
      sender_pubkey: senderPubkey,
      contract: JSON.parse(payloadRaw),
    };

    const toAddress = byId("toAddress").value.trim();
    const amount = byId("amount").value.trim();
    const fee = byId("fee").value.trim();
    if (toAddress) payload.to_address = toAddress;
    if (amount) payload.amount = Number(amount);
    if (fee) payload.fee = Number(fee);

    const data = await api("/api/v1/market/build-contract", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    byId("unsignedOutput").value = JSON.stringify(data.result || {}, null, 2);
  } catch (error) {
    setError(error.message || String(error));
  }
}

async function submitSignedTx() {
  setError("");
  try {
    const txRaw = byId("signedInput").value.trim();
    if (!txRaw) {
      throw new Error("signed tx JSON is required");
    }
    const payload = { tx: JSON.parse(txRaw) };
    const ttlRaw = byId("broadcastTtl").value.trim();
    if (ttlRaw) payload.broadcast_ttl = Number(ttlRaw);

    const data = await api("/api/v1/market/submit-signed", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    byId("submitOutput").value = JSON.stringify(data.result || {}, null, 2);
  } catch (error) {
    setError(error.message || String(error));
  }
}

function onExplorerAction(event) {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  const action = target.dataset.action;
  if (!action) return;

  const token = target.dataset.token || "";
  const seller = target.dataset.seller || "";
  const price = target.dataset.price || "";

  if (action === "prefill-list") {
    fillListForm(token);
    return;
  }
  if (action === "prefill-cancel") {
    byId("cancelTokenId").value = token;
    return;
  }
  if (action === "prefill-buy") {
    fillBuyForm(token, seller, price);
  }
}

function wireEvents() {
  byId("refreshBtn").addEventListener("click", refreshAll);
  byId("syncNowBtn").addEventListener("click", triggerSync);
  byId("applyFilterBtn").addEventListener("click", async () => {
    state.owner = byId("ownerFilter").value.trim();
    state.search = byId("searchFilter").value.trim();
    state.listed = byId("listedFilter").value;
    await refreshNfts();
  });

  byId("toggleSignerKeyBtn").addEventListener("click", () => {
    const input = byId("signerPrivateKey");
    if (input.type === "password") {
      input.type = "text";
      byId("toggleSignerKeyBtn").textContent = "Hide";
    } else {
      input.type = "password";
      byId("toggleSignerKeyBtn").textContent = "Show";
    }
  });

  byId("mintBtn").addEventListener("click", mintNft);
  byId("listBtn").addEventListener("click", listNft);
  byId("cancelBtn").addEventListener("click", cancelListing);
  byId("buyBtn").addEventListener("click", buyNft);

  byId("nftGrid").addEventListener("click", onExplorerAction);
  byId("listingTable").addEventListener("click", onExplorerAction);

  byId("buildUnsignedBtn").addEventListener("click", buildUnsignedTx);
  byId("submitSignedBtn").addEventListener("click", submitSignedTx);
}

async function boot() {
  wireEvents();
  await refreshAll();
  setInterval(refreshAll, 10000);
}

window.addEventListener("DOMContentLoaded", boot);
