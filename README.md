# AnemiaScan

**Problem Statement:** `Omni_BioTech_2` - Non-Invasive Anaemia Screening  
**Event:** Omnikon National Hackathon 2026  
**Team:** Zenith Innovator - Uday Jain

AnemiaScan is a **non-diagnostic, camera-assisted screening prototype**. It guides a user to capture the palpebral conjunctiva, extracts an erythema/pallor colour index, and returns a preliminary risk band with a blood-test referral message. It must not be used to diagnose, treat, or rule out anaemia.

## What is implemented

- React Progressive Web App with installable/offline-ready build configuration
- Consent before camera/photo access and no image storage in on-device mode
- Camera capture, image-upload fallback, and guided conjunctiva region of interest
- In-browser RGB/HSV pallor index and 3-class risk-band fallback (on-device mode)
- FastAPI + OpenCV `POST /analyze` API with in-memory RGBA-aware image processing
- **Ensemble Classifier (Random Forest + Gradient Boosting v2)** trained on 216 palpebral conjunctiva images with 6x augmentation — extracts 37 colour & texture features (RGB/HSV/LAB stats, LBP micro-texture histograms, erythema index, pallor score, tissue area fraction, colour ratios) and returns Anemic / Mild / Normal with confidence probabilities
- **Hgb Regressor (Random Forest v2)** that predicts continuous haemoglobin levels (g/dL) directly
- `backend/train_model.py` reproducible training script; `backend/model_report.txt` full metrics report
- Input-type/size checks and production-configurable CORS
- Render and Vercel deployment configuration
- SQLite demo record store for API development; it stores **only derived colour features, estimated Hgb, and risk band**, never images or names

## Architecture status

| Layer | Current implementation | Final deployment path |
|---|---|---|
| Frontend | React + Vite PWA (displays band, confidence %, and estimated Hgb) | Vercel |
| Backend | FastAPI + OpenCV + scikit-learn + scikit-image | Render |
| Image processing | RGBA-aware ROI extraction + 37-feature RGB/HSV/LAB/LBP texture vector (tissue-masked) | Same |
| Classifier | **Ensemble v2 (Random Forest + Gradient Boosting)** — trained on *Eyes-Defy-Anemia* dataset (216 palpebral conjunctiva images, India + Italy cohorts) with 6x data augmentation; 3-class output: Anemic / Mild / Normal | CNN / TFLite after larger validated dataset |
| Regressor | **Random Forest Regressor v2** — direct Haemoglobin estimation (CV MAE: 1.21 g/dL) | CNN Regression |
| Classifier accuracy | **79.1% ± 4.7% CV accuracy** (5-fold stratified on real data); **56.8% held-out test accuracy** (44 samples) — small dataset, not yet clinically validated | Requires prospective clinical validation before health use |
| Data | SQLite development demo | Supabase PostgreSQL when server-only credentials are configured; schema at `supabase/schema.sql` |

**Important:** The trained models (`backend/model_rf.pkl` and `backend/model_reg.pkl`) are included in the repository and loaded automatically at API startup. The 79.1% CV accuracy and 1.21 g/dL MAE reflect real performance on 216 images — honest, reproducible, and documented in `backend/model_report.txt`. This is a screening aid, not a diagnostic tool; clinical validation on a larger, prospectively collected dataset is required before any health deployment.

## Run locally

Prerequisites: Node 18+ and Python 3.12+.

```bash
pnpm install
pnpm dev
```

In a second terminal:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set ALLOWED_ORIGINS=http://localhost:5173
uvicorn main:app --reload --port 8000
```

Copy `.env.example` to `.env` and set `VITE_API_URL=http://localhost:8000` to use the OpenCV API. Leaving it blank uses the on-device privacy-first fallback.

## Deploy

1. Deploy the repository to Vercel as a Vite project. Set `VITE_API_URL` to the Render API URL.
2. Create a Render web service from `render.yaml`. Set `ALLOWED_ORIGINS` to the exact Vercel URL; never use `*` in production.
3. Add the deployed frontend URL in the GitHub repository **About** section.
4. Before enabling a CHW dashboard, create a Supabase project, run `supabase/schema.sql`, and implement authenticated, ward-scoped Row Level Security policies. Do not use anonymous access to health-adjacent screening data.

## Safety, privacy, and limitations

- This is a screening/triage demonstration, not a medical device.
- Images are analysed in memory and discarded by the API. On-device mode does not upload them.
- The **Ensemble v2** classifier was trained on 216 palpebral conjunctiva images and achieves **79.1% ± 4.7% CV accuracy** (3-class: Anemic / Mild / Normal) with continuous Hgb regression (CV MAE: 1.21 g/dL). This is an honest benchmark on a small dataset — it has **not been clinically validated** and must not be used for diagnosis, treatment decisions, or as a replacement for a haemoglobin blood test.
- The current prototype supports the **conjunctiva** workflow only. Nail-bed/tongue fusion, a physical colour-card correction matrix, a larger validated dataset, and a live CHW dashboard are future work.

See [SECURITY.md](SECURITY.md) for the full data-handling notes.

## Dataset & Attribution

| Item | Details |
|---|---|
| **Dataset name** | Eyes-Defy-Anemia |
| **Author** | Harshwardhan Fartale |
| **Source** | [Kaggle — Eyes-Defy-Anemia](https://www.kaggle.com/datasets/harshwardhanfartale/eyes-defy-anemia) |
| **Cohorts** | India (95 patients) + Italy (123 patients) |
| **Samples used** | 216 after cleaning (1 patient excluded: no palpebral image) |
| **Image type** | Pre-segmented palpebral conjunctiva PNG images with clinical Hgb (g/dL) ground-truth labels |
| **License** | As published on Kaggle by the dataset author |

The dataset provides clinical haemoglobin (Hgb) values measured via standard blood test, used to derive WHO-based 3-class labels: **Anemic** (Hgb < 11 g/dL), **Mild** (11–11.9 g/dL), **Normal** (≥ 12 g/dL).

## Third-party tools and attribution

React, Vite, FastAPI, OpenCV, scikit-learn, NumPy, openpyxl, and Pillow are used under their respective open-source licenses. See `backend/requirements.txt` and `package.json` for full dependency lists.

## Generative AI disclosure

Generative AI tools were used to assist with implementation scaffolding, documentation, and research synthesis. The team reviewed the output, made final technical decisions, and is responsible for all submitted work.

## Team

| GitHub username | Role |
|---|---|
| `@udayjain06` | Product, frontend, backend, ML training, deployment, and documentation |

## License

MIT - see [LICENSE](LICENSE).
