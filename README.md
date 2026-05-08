# AzureContentUnderstandingClient

Minimal Python test client for Azure AI Content Understanding GA API `2025-11-01`.

It iterates over every PDF and PNG in `samples/` and runs a classify-first routing flow:

1. Classify the document with a custom classifier analyzer (categories: invoice, receipt, contract, other).
2. Route based on the classifier label only. Azure Content Understanding does not return a calibrated classification confidence in api-version `2025-11-01`, so the script trusts the chosen category and validates the routed extraction afterwards.
3. Currently only `invoice` is routed; it runs `invoice_analyzer`, a custom analyzer that extracts core invoice fields and line items with `estimateFieldSourceAndConfidence` enabled (per-field source/confidence is available inside the saved JSON).
4. If the classifier returns no category, the routed extraction returns no fields, or the category is unsupported, the sample is skipped with a graceful error.

The script automatically detects schema drift on the classifier and invoice analyzer (via a `[schema:vN]` tag stamped into each analyzer description) and recreates them when the in-code version is bumped. Transient HTTP failures (429/5xx, connection/timeout) are retried with exponential backoff and `Retry-After` is honored.

## Run

1. Install dependencies:
	`python -m pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in `AZURE_CU_ENDPOINT` and `AZURE_CU_KEY`.
3. If your Foundry model deployment names differ from the model names, update the `AZURE_CU_*_DEPLOYMENT` values in `.env`.
	The classifier and invoice analyzer use `gpt-4.1` plus the `prebuilt-analyzer-embedding` default mapped to your `text-embedding-3-large` deployment.
4. Put PDF/PNG files in `samples/` (or set `AZURE_CU_SAMPLES_DIR`).
5. Run:
	`python cu_test.py`

For each sample, the classification response is saved to `outputs/<name>_classification.json` and the routed extraction to `outputs/<name>_<category>_analysis.json`.