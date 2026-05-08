from __future__ import annotations

import base64
import json
import os
import random
import re
import sys
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from dotenv import load_dotenv

load_dotenv()

ENDPOINT = os.getenv("AZURE_CU_ENDPOINT", "").rstrip("/")
KEY = os.getenv("AZURE_CU_KEY", "")
API_VERSION = os.getenv("AZURE_CU_API_VERSION", "2025-11-01")
CLASSIFIER_ID = os.getenv("AZURE_CU_CLASSIFIER_ID", "test_doc_classifier")
INVOICE_ANALYZER_ID = os.getenv("AZURE_CU_INVOICE_ANALYZER_ID", "invoice_analyzer")
COMPLETION_MODEL = os.getenv("AZURE_CU_COMPLETION_MODEL", "gpt-4.1")
GPT41_DEPLOYMENT = os.getenv("AZURE_CU_GPT41_DEPLOYMENT", "gpt-4.1")
GPT41_MINI_DEPLOYMENT = os.getenv("AZURE_CU_GPT41_MINI_DEPLOYMENT", "")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_CU_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
SAMPLES_DIR = Path(os.getenv("AZURE_CU_SAMPLES_DIR", "samples"))
OUTPUT_DIR = Path("outputs")
SAMPLE_EXTENSIONS = {".pdf", ".png"}
TIMEOUT_SECONDS = int(os.getenv("AZURE_CU_OPERATION_TIMEOUT_SECONDS", "180"))
POLL_SECONDS = 2
HTTP_MAX_ATTEMPTS = int(os.getenv("AZURE_CU_HTTP_MAX_ATTEMPTS", "5"))
HTTP_BACKOFF_MAX = float(os.getenv("AZURE_CU_HTTP_BACKOFF_MAX_SECONDS", "30"))
ID_PATTERN = re.compile(r"^[A-Za-z0-9_.\-]+$")
RETRY_STATUSES = {429, 502, 503, 504}

CLASSIFIER_VERSION = "v2"
INVOICE_VERSION = "v1"
SUPPORTED_CATEGORIES = {"invoice"}


class ContentUnderstandingError(RuntimeError):
    pass


SESSION = requests.Session()
SESSION.headers.update({"Ocp-Apim-Subscription-Key": KEY})


def require_config() -> None:
    """Validate required env config (endpoint, key, analyzer IDs, samples dir) or raise."""
    missing = [name for name, val in (("AZURE_CU_ENDPOINT", ENDPOINT), ("AZURE_CU_KEY", KEY)) if not val]
    if missing:
        raise ContentUnderstandingError(f"Missing required setting(s): {', '.join(missing)}. Add them to .env.")
    for analyzer_id, env_name in ((CLASSIFIER_ID, "AZURE_CU_CLASSIFIER_ID"), (INVOICE_ANALYZER_ID, "AZURE_CU_INVOICE_ANALYZER_ID")):
        if not analyzer_id or not ID_PATTERN.match(analyzer_id):
            raise ContentUnderstandingError(f"{env_name}='{analyzer_id}' is not a valid analyzer id.")
    if not SAMPLES_DIR.is_dir():
        raise ContentUnderstandingError(f"Samples directory not found: {SAMPLES_DIR}.")


def discover_samples() -> list[Path]:
    """Return a sorted list of PDF/PNG files in the samples directory."""
    files = sorted(p for p in SAMPLES_DIR.iterdir() if p.is_file() and p.suffix.lower() in SAMPLE_EXTENSIONS)
    if not files:
        raise ContentUnderstandingError(f"No PDF or PNG files found in {SAMPLES_DIR}.")
    return files


def add_api_version(url: str) -> str:
    """Append the configured `api-version` query parameter to a URL if absent."""
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("api-version", API_VERSION)
    return urlunparse(parsed._replace(query=urlencode(query)))


def make_url(path: str) -> str:
    """Build a fully qualified Content Understanding URL for the given path."""
    return add_api_version(ENDPOINT + (path if path.startswith("/") else "/" + path))


def response_text(r: requests.Response) -> str:
    """Return a pretty-printed JSON body if possible, else the raw response text."""
    try:
        return json.dumps(r.json(), indent=2)
    except ValueError:
        return r.text


def parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header (seconds or HTTP-date) into a delay in seconds."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, (parsedate_to_datetime(value).timestamp() - time.time()))
        except (TypeError, ValueError):
            return None


def request_with_retry(method: str, url: str, **kwargs: Any) -> requests.Response:
    """Issue an HTTP request with retry/backoff for 429/5xx and connection errors."""
    kwargs.setdefault("timeout", 60)
    last_exc: Exception | None = None
    for attempt in range(1, HTTP_MAX_ATTEMPTS + 1):
        try:
            response = SESSION.request(method, url, **kwargs)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == HTTP_MAX_ATTEMPTS:
                raise
            delay = min(HTTP_BACKOFF_MAX, (2 ** (attempt - 1)) + random.uniform(0, 0.5))
            print(f"  HTTP {type(exc).__name__}; retry {attempt}/{HTTP_MAX_ATTEMPTS} in {delay:.1f}s...")
            time.sleep(delay)
            continue

        if response.status_code not in RETRY_STATUSES or attempt == HTTP_MAX_ATTEMPTS:
            return response

        retry_after = parse_retry_after(response.headers.get("Retry-After"))
        delay = retry_after if retry_after is not None else min(HTTP_BACKOFF_MAX, (2 ** (attempt - 1)) + random.uniform(0, 0.5))
        print(f"  HTTP {response.status_code}; retry {attempt}/{HTTP_MAX_ATTEMPTS} in {delay:.1f}s...")
        time.sleep(delay)

    if last_exc:
        raise last_exc
    raise ContentUnderstandingError("request_with_retry exhausted attempts.")


def cu_call(method: str, path: str, *, action: str, expected: tuple[int, ...] = (200,), **kwargs: Any) -> requests.Response:
    """Call the Content Understanding API and assert the response status is expected."""
    headers = {"Content-Type": "application/json"} if "json" in kwargs else {}
    headers.update(kwargs.pop("headers", {}))
    response = request_with_retry(method, make_url(path), headers=headers or None, **kwargs)
    if response.status_code not in expected:
        raise ContentUnderstandingError(f"{action} failed: HTTP {response.status_code}\n{response_text(response)}")
    return response


def poll_operation(operation_url: str, action: str) -> dict[str, Any]:
    """Poll a long-running operation URL until it succeeds, fails, or times out."""
    deadline = time.time() + TIMEOUT_SECONDS
    url = add_api_version(operation_url)
    while time.time() < deadline:
        r = request_with_retry("GET", url)
        if not r.ok:
            raise ContentUnderstandingError(f"Polling {action} failed: HTTP {r.status_code}\n{response_text(r)}")
        payload = r.json()
        status = str(payload.get("status", "")).lower()
        if status in {"succeeded", "succeededwithwarnings"}:
            return payload
        if status in {"failed", "canceled", "cancelled"}:
            raise ContentUnderstandingError(f"{action} failed with status '{payload.get('status')}'.\n{json.dumps(payload, indent=2)}")
        print(f"  {action}: {status or 'running'}...")
        time.sleep(POLL_SECONDS)
    raise ContentUnderstandingError(f"Timed out waiting for {action} after {TIMEOUT_SECONDS}s.")


def operation_url(response: requests.Response, required: bool = True) -> str | None:
    """Return the Operation-Location header from a response, raising if required and missing."""
    loc = response.headers.get("Operation-Location") or response.headers.get("operation-location")
    if not loc and required:
        raise ContentUnderstandingError(f"Missing Operation-Location header.\n{response_text(response)}")
    return loc


def save_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload to disk, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def schema_tag(version: str) -> str:
    """Return the schema-version marker stamped into analyzer descriptions."""
    return f"[schema:{version}]"


def tag_description(description: str, version: str) -> str:
    """Append the schema-version marker to an analyzer description string."""
    return f"{description} {schema_tag(version)}"


def configure_defaults() -> None:
    """Set the resource-wide model deployment defaults used by analyzers."""
    print("Ensuring Content Understanding model deployment defaults...")
    deployments: dict[str, str] = {"gpt-4.1": GPT41_DEPLOYMENT, "prebuilt-analyzer-embedding": EMBEDDING_DEPLOYMENT}
    if GPT41_MINI_DEPLOYMENT:
        deployments["gpt-4.1-mini"] = GPT41_MINI_DEPLOYMENT
    cu_call("PATCH", "/contentunderstanding/defaults", action="Configuring defaults",
            expected=(200, 204), json={"modelDeployments": deployments})


def analyzer_state(analyzer_id: str, version: str) -> str:
    """Return whether the analyzer is missing, current (matching version tag), or stale."""
    r = request_with_retry("GET", make_url(f"/contentunderstanding/analyzers/{analyzer_id}"))
    if r.status_code == 404:
        return "missing"
    if not r.ok:
        raise ContentUnderstandingError(f"Checking '{analyzer_id}' failed: HTTP {r.status_code}\n{response_text(r)}")
    description = (r.json() or {}).get("description", "")
    return "current" if schema_tag(version) in description else "stale"


def delete_analyzer(analyzer_id: str) -> None:
    """Delete an analyzer by ID, tolerating 404."""
    print(f"  Deleting stale analyzer '{analyzer_id}'...")
    r = request_with_retry("DELETE", make_url(f"/contentunderstanding/analyzers/{analyzer_id}"))
    if r.status_code not in (200, 204, 404):
        raise ContentUnderstandingError(f"Deleting '{analyzer_id}' failed: HTTP {r.status_code}\n{response_text(r)}")


def put_analyzer(analyzer_id: str, payload: dict[str, Any]) -> requests.Response:
    """Send a PUT request to create or replace an analyzer with the given payload."""
    return request_with_retry("PUT", make_url(f"/contentunderstanding/analyzers/{analyzer_id}"),
                              headers={"Content-Type": "application/json"}, json=payload)


def create_analyzer(analyzer_id: str, payload: dict[str, Any], action: str) -> None:
    """Create the analyzer (configuring defaults on demand) and wait for completion."""
    print(f"Creating analyzer '{analyzer_id}'...")
    r = put_analyzer(analyzer_id, payload)
    if r.status_code == 400 and "DefaultsNotSet" in response_text(r):
        configure_defaults()
        r = put_analyzer(analyzer_id, payload)
    if r.status_code not in (200, 201, 202):
        raise ContentUnderstandingError(f"{action} failed: HTTP {r.status_code}\n{response_text(r)}")
    loc = operation_url(r, required=False)
    if loc:
        poll_operation(loc, action)


def ensure_analyzer(analyzer_id: str, version: str, build_payload, action: str) -> None:
    """Make sure the analyzer exists at the requested schema version, recreating if stale."""
    print(f"Checking analyzer '{analyzer_id}'...")
    state = analyzer_state(analyzer_id, version)
    if state == "current":
        print("  Up to date.")
        return
    if state == "stale":
        delete_analyzer(analyzer_id)
    create_analyzer(analyzer_id, build_payload(), action)


def run_analyzer(analyzer_id: str, document: bytes, action: str) -> dict[str, Any]:
    """Submit a document to an analyzer and return the completed operation result."""
    print(f"{action} with analyzer '{analyzer_id}'...")
    r = cu_call("POST", f"/contentunderstanding/analyzers/{analyzer_id}:analyze",
                action=f"Starting {action.lower()}", expected=(202,),
                json={"inputs": [{"data": base64.b64encode(document).decode("ascii")}]})
    return poll_operation(operation_url(r), action)


def classifier_payload() -> dict[str, Any]:
    """Build the request payload for creating the document-routing classifier."""
    return {
        "baseAnalyzerId": "prebuilt-document",
        "description": tag_description("Routes business documents to category-specific analyzers.", CLASSIFIER_VERSION),
        "config": {"returnDetails": True},
        "fieldSchema": {
            "name": "DocumentRouting",
            "fields": {
                "category": {
                    "type": "string", "method": "classify",
                    "description": "The document category.",
                    "enum": ["invoice", "receipt", "contract", "other"],
                    "enumDescriptions": {
                        "invoice": "Invoices, bills, or payment requests from vendors.",
                        "receipt": "Receipts or proofs of purchase/payment.",
                        "contract": "Contracts, agreements, terms, or statements of work.",
                        "other": "Any document that is not an invoice, receipt, or contract.",
                    },
                },
            },
        },
        "models": {"completion": COMPLETION_MODEL},
    }


def invoice_analyzer_payload() -> dict[str, Any]:
    """Build the request payload for creating the invoice extraction analyzer."""
    string_fields = {
        "invoiceNumber": "Invoice number or identifier (e.g. INV-2026-0001).",
        "vendorName": "Name of the seller or service provider issuing the invoice.",
        "vendorAddress": "Full mailing address of the vendor.",
        "customerName": "Name of the customer or party being billed.",
        "customerAddress": "Full mailing address of the customer.",
        "currency": "Currency code or symbol used (e.g. USD, $, EUR).",
        "paymentTerms": "Payment terms (e.g. Net 30).",
    }
    number_fields = {
        "subtotal": "Subtotal amount before tax.",
        "taxAmount": "Total tax or VAT amount.",
        "totalDue": "Total amount due on the invoice.",
    }
    date_fields = {
        "invoiceDate": "Date the invoice was issued.",
        "dueDate": "Date the invoice payment is due.",
    }
    fields: dict[str, dict[str, Any]] = {}
    for name, desc in string_fields.items():
        fields[name] = {"type": "string", "method": "extract", "description": desc}
    for name, desc in number_fields.items():
        fields[name] = {"type": "number", "method": "extract", "description": desc}
    for name, desc in date_fields.items():
        fields[name] = {"type": "date", "method": "extract", "description": desc}
    fields["lineItems"] = {
        "type": "array", "method": "generate",
        "description": "Each individual line item or charge on the invoice.",
        "items": {
            "type": "object", "method": "generate",
            "properties": {
                "description": {"type": "string", "method": "extract", "description": "Description of the product or service."},
                "quantity": {"type": "number", "method": "extract", "description": "Quantity for this line item."},
                "unitPrice": {"type": "number", "method": "extract", "description": "Unit price for this line item."},
                "amount": {"type": "number", "method": "extract", "description": "Total amount for this line item."},
            },
        },
    }
    return {
        "baseAnalyzerId": "prebuilt-document",
        "description": tag_description("Extracts core invoice fields and line items.", INVOICE_VERSION),
        "config": {"returnDetails": True, "estimateFieldSourceAndConfidence": True},
        "fieldSchema": {"name": "InvoiceFields", "fields": fields},
        "models": {"completion": COMPLETION_MODEL},
    }


def iter_contents(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """Yield each content/result object from an analyzer response payload."""
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    contents = result.get("contents") if isinstance(result, dict) else None
    if isinstance(contents, list):
        for item in contents:
            if isinstance(item, dict):
                yield item
    elif isinstance(result, dict):
        yield result


def field_value(field: Any) -> Any:
    """Extract the typed value (string/number/date/array/object) from a CU field object."""
    if not isinstance(field, dict):
        return None
    for key in ("valueString", "valueNumber", "valueDate", "valueInteger", "valueBoolean", "valueArray", "valueObject", "value", "content"):
        if key in field and field[key] not in (None, ""):
            return field[key]
    return None


def top_category(classification_payload: dict[str, Any]) -> str:
    """Return the classifier's chosen category string from the response payload."""
    for content in iter_contents(classification_payload):
        fields = content.get("fields")
        if isinstance(fields, dict):
            value = field_value(fields.get("category"))
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ContentUnderstandingError("Classifier did not return a category.")


def select_analyzer(category: str) -> str:
    """Map a classifier category to the analyzer ID that should handle extraction."""
    normalized = category.lower()
    if normalized not in SUPPORTED_CATEGORIES:
        raise ContentUnderstandingError(f"Category '{category}' is not supported. Supported: {sorted(SUPPORTED_CATEGORIES)}.")
    if normalized == "invoice":
        return INVOICE_ANALYZER_ID
    raise ContentUnderstandingError(f"No analyzer mapping for supported category '{category}'.")


def slugify(value: str) -> str:
    """Convert a string to a filesystem-safe lowercase slug."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "document"


def extract_summary(extraction_payload: dict[str, Any]) -> dict[str, Any]:
    """Flatten an extraction response into a {field_name: value} dict for the populated fields."""
    summary: dict[str, Any] = {}
    for content in iter_contents(extraction_payload):
        fields = content.get("fields")
        if not isinstance(fields, dict):
            continue
        for name, field in fields.items():
            value = field_value(field)
            if value is not None:
                summary[name] = value
    return summary


def process_sample(sample: Path) -> dict[str, Any]:
    """Classify and extract a single sample file, saving JSON outputs and returning a summary."""
    print(f"\n=== Processing {sample.name} ===")
    document = sample.read_bytes()
    stem = slugify(f"{sample.stem}_{sample.suffix.lstrip('.')}")

    classification = run_analyzer(CLASSIFIER_ID, document, "Classification")
    classification_path = OUTPUT_DIR / f"{stem}_classification.json"
    save_json(classification_path, classification)

    category = top_category(classification)
    print(f"  Category: {category}")
    analyzer_id = select_analyzer(category)

    extraction = run_analyzer(analyzer_id, document, f"{category.title()} extraction")
    extraction_path = OUTPUT_DIR / f"{stem}_{category.lower()}_analysis.json"
    save_json(extraction_path, extraction)

    summary = extract_summary(extraction)
    if not summary:
        raise ContentUnderstandingError(f"{analyzer_id} returned no field values; manual review required.")

    preview = {k: summary[k] for k in ("invoiceNumber", "invoiceDate", "totalDue", "currency", "vendorName", "customerName") if k in summary}
    if preview:
        print(f"  Extracted: {preview}")

    return {"sample": str(sample), "category": category, "analyzerId": analyzer_id,
            "classificationPath": str(classification_path), "extractionPath": str(extraction_path),
            "summary": summary}


def main() -> int:
    """Entry point: configure defaults, ensure analyzers, then process every sample."""
    try:
        require_config()
        OUTPUT_DIR.mkdir(exist_ok=True)
        samples = discover_samples()
        print(f"Found {len(samples)} sample(s) in {SAMPLES_DIR}: {[s.name for s in samples]}")
        configure_defaults()
        ensure_analyzer(CLASSIFIER_ID, CLASSIFIER_VERSION, classifier_payload, "Classifier creation")
        ensure_analyzer(INVOICE_ANALYZER_ID, INVOICE_VERSION, invoice_analyzer_payload, "Invoice analyzer creation")
    except ContentUnderstandingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"HTTP ERROR: {exc}", file=sys.stderr)
        return 1

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for sample in samples:
        try:
            successes.append(process_sample(sample))
        except (ContentUnderstandingError, requests.RequestException) as exc:
            print(f"  SKIPPED {sample.name}: {exc}", file=sys.stderr)
            failures.append({"sample": str(sample), "error": str(exc)})

    print(f"\n=== Summary === processed={len(successes)} skipped={len(failures)}")
    for item in successes:
        print(f"  - {Path(item['sample']).name}: {item['category']} -> {item['analyzerId']}")
    for item in failures:
        print(f"  - {Path(item['sample']).name}: {item['error']}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
