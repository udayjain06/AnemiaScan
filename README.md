# AnemiaScan

**Problem Statement:** `Omni_BioTech_2` - Non-Invasive Anaemia Screening  
**Event:** Omnikon National Hackathon 2026  
**Team:** Zenith Innovator - Uday Jain

AnemiaScan is a **non-diagnostic, camera-assisted screening prototype**. It guides a user to capture the palpebral conjunctiva, extracts an erythema/pallor colour index, and returns a preliminary risk band with a blood-test referral message. It must not be used to diagnose, treat, or rule out anaemia.

## What is implemented

- React Progressive Web App with installable/offline-ready build configuration
- Consent before camera/photo access and no image storage in on-device mode
- Camera capture, image-upload fallback, and guided conjunctiva region of interest
- In-browser RGB/HSV pallor index and risk-band fallback
- FastAPI + OpenCV `POST /analyze` API with in-memory image processing
- Input-type/size checks and production-configurable CORS
- Render and Vercel deployment configuration
- SQLite demo record store for API development; it stores **only derived colour features and risk band**, never images or names

## Architecture status

| Layer | Current implementation | Final deployment path |
|---|---|---|
| Frontend | React + Vite PWA | Vercel |
| Backend | FastAPI + OpenCV | Render |
| Image processing | ROI extraction + RGB/HSV erythema/pallor index | Same |
| Classifier | Transparent rule-based v0 | TFLite model only after training and validation on an attributed real dataset |
| Data | SQLite development demo | Supabase PostgreSQL when server-only credentials are configured; schema at `supabase/schema.sql` |

**Important:** Supabase storage is code-ready but is inactive until server-only Supabase credentials are configured and tested. TensorFlow Lite is not implemented because no clinically suitable, attributed model has been supplied. Do not say either is live in the demo, README, PPT, or video until it is actually configured and tested.

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
- The current v0 bands are colour-index rules, not clinical predictions. They have not been clinically validated and must not be paired with a claimed accuracy, AUC, sensitivity, or haemoglobin value.
- The current prototype supports the **conjunctiva** workflow only. Nail-bed/tongue fusion, a physical colour-card correction matrix, a real TFLite model, and a live CHW dashboard are future work.

See [SECURITY.md](SECURITY.md) for the full data-handling notes.

## Third-party tools and attribution

React, Vite, FastAPI, OpenCV, and Supabase (planned) are used under their respective licenses. The project must add citations and licenses for every final dataset, model, icon, image, and research source before submission.

## Generative AI disclosure

Generative AI tools were used to assist with implementation scaffolding, documentation, and research synthesis. The team reviewed the output, made final technical decisions, and is responsible for all submitted work.

## Team

| GitHub username | Role |
|---|---|
| `@udayjain06` | Product, frontend, backend, deployment, and documentation |

## License

MIT - see [LICENSE](LICENSE).
