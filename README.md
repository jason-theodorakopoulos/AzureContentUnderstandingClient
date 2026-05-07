# AzureContentUnderstandingClient

Minimal Python test client for Azure AI Content Understanding GA API `2025-11-01`.

It runs two operations against a local document:

1. Layout extraction with `prebuilt-layout`.
2. Document classification with a custom classifier analyzer created from `contentCategories`.

## Run

1. Install dependencies:
	`python -m pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in `AZURE_CU_ENDPOINT` and `AZURE_CU_KEY`.
3. If your Foundry model deployment names differ from the model names, update the `AZURE_CU_*_DEPLOYMENT` values in `.env`.
	The classifier flow uses `gpt-4.1` plus the `prebuilt-analyzer-embedding` default mapped to your `text-embedding-3-large` deployment.
4. Put a PDF/image at `samples/invoice.pdf`, or change `AZURE_CU_INPUT_FILE` in `.env`.
5. Run:
	`python cu_test.py`

The full responses are saved to `outputs/layout.json` and `outputs/classification.json`.