# Security & Data Handling

## Current build

- **Frontend-only mode** (no `BACKEND_URL` set): all image capture and analysis
  happens entirely client-side, in the browser. No image is ever uploaded anywhere.
- **Backend-connected mode**: the captured frame is sent once to the `/analyze`
  endpoint over HTTPS (once deployed on Render), analysed in-memory by the backend,
  and **the image itself is discarded immediately after analysis** — it is never
  written to disk or the database. Only the computed result is persisted:
  risk band, colour features (avg RGB, saturation, value, erythema index, pallor
  score), the classification method used, and a timestamp.
- No personal identifiers, accounts, or names are collected or stored at this stage.
- The SQLite database (`screenings.db`) lives on the backend's filesystem; on
  Render's free tier this storage is **ephemeral** and resets on redeploy — this is
  a known limitation of the current build, not a production data-retention design.

## Planned (as the CHW dashboard is built)

- **Consent-first**: a clear consent screen before any capture begins.
- **On-device inference** as the primary path (already true in frontend-fallback
  mode today) — the backend path exists for the CHW dashboard and cross-device
  record-keeping use case, not because on-device analysis needs help.
- **Encryption in transit** (TLS) — provided by Render/Vercel's default HTTPS.
- **Access control** on the CHW dashboard — screening records visible only to
  authorised health workers for their assigned ward.
- **Data-deletion path** for any user/CHW to request removal of stored records.
- **CORS lock-down**: `main.py` uses `ALLOWED_ORIGINS`, which must contain the
  exact deployed Vercel URL before final submission. Wildcard CORS is not used.
- Design considerations align with India's DPDP Act 2023 for handling of sensitive
  personal data (health-adjacent imagery).

## Not a medical device

AnemiaScan is a **screening and triage aid**, not a diagnostic device. It does not
store or transmit data for clinical decision-making without a human-in-the-loop
referral to a real blood test. The classifier is explicitly a transparent rule-based
v0 (see README) — its output must not be represented as clinically validated.

## Reporting a concern

Please open a GitHub issue on this repository for any security or privacy concern.
