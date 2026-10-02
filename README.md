# AutoTriage AI — backend (PS 16)

## Run
    python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    python train.py                                      # or --csv ... (see below)
    uvicorn app:app --host 0.0.0.0 --port 8000
Docs/test UI: http://localhost:8000/docs

## Real Kaggle data
    python train.py --csv data/customer_support_tickets.csv \
        --text-col "Ticket Description" --cat-col "Ticket Type" --prio-col "Ticket Priority"

## Endpoints
GET /health · POST /triage · POST /triage/batch · POST /triage/csv (file upload)
GET /tickets · DELETE /tickets · GET /stats · GET /metrics · GET /misclassified

## Expose to the Lovable frontend (needs HTTPS)
    cloudflared tunnel --url http://localhost:8000      # or: ngrok http 8000
