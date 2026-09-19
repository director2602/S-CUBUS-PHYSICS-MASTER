# Physics Master (deployable)

Web service that writes NEET / JEE Physics papers with one AI and independently
solves + audits every question with the other AI (Claude <-> Gemini).

## Files
- `physics_master.py`  core engine (also runs as a CLI: `python physics_master.py`)
- `server.py`          FastAPI service: background jobs, API-key auth, `/health`
- `index.html`         browser UI (generate, download, remedial questions)
- `render.yaml`        Render Blueprint: web service + Postgres
- `requirements.txt`, `.env.example`, `.gitignore`

## Deploy on Render
1. Create a new GitHub repo (private) and push these files.
2. Render Dashboard -> New -> Blueprint -> pick the repo -> Apply.
3. When prompted, paste `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`.
4. When the deploy is live, open the service -> Environment -> copy `APP_API_KEY`.
   That is the access key you paste into the web page.
5. Open the service URL, paste the access key, generate a 5-question paper first.

## Run locally
    pip install -r requirements.txt
    cp .env.example .env      # fill in keys
    uvicorn server:app --reload
    # open http://127.0.0.1:8000   (uses a local SQLite file unless DATABASE_URL is set)

## Notes
- Do not commit `.env`. The two provider keys and APP_API_KEY are the only secrets.
- Keep `JOB_WORKERS=1`; jobs are queued one at a time to respect API rate limits.
- Every accepted question is stored in Postgres, and new papers on the same chapter
  automatically avoid repeating earlier questions from the bank.
- Set `ENABLE_CODE_CHECK=0` to disable executing model-written verification snippets.
