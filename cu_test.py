from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import requests
from dotenv import load_dotenv


load_dotenv()

ENDPOINT = os.getenv("AZURE_CU_ENDPOINT", "").rstrip("/")
KEY = os.getenv("AZURE_CU_KEY", "")
API_VERSION = os.getenv("AZURE_CU_API_VERSION", "2025-11-01")
LAYOUT_ANALYZER_ID = os.getenv("AZURE_CU_LAYOUT_ANALYZER_ID", "prebuilt-layout")
CLASSIFIER_ID = os.getenv("AZURE_CU_CLASSIFIER_ID", "test_doc_classifier")
COMPLETION_MODEL = os.getenv("AZURE_CU_COMPLETION_MODEL", "gpt-4.1")
GPT41_DEPLOYMENT = os.getenv("AZURE_CU_GPT41_DEPLOYMENT", "gpt-4.1")
GPT41_MINI_DEPLOYMENT = os.getenv("AZURE_CU_GPT41_MINI_DEPLOYMENT", "")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_CU_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
INPUT_FILE = Path(os.getenv("AZURE_CU_INPUT_FILE", "samples/invoice.pdf"))
OUTPUT_DIR = Path("outputs")
TIMEOUT_SECONDS = 180
POLL_SECONDS = 2


class ContentUnderstandingError(RuntimeError):
    pass


def require_config() -> None:
    missing = []
    if not ENDPOINT:
        missing.append("AZURE_CU_ENDPOINT")
    if not KEY:
        missing.append("AZURE_CU_KEY")
    if missing:
        raise ContentUnderstandingError(
            "Missing required setting(s): " + ", ".join(missing) + ". Add them to .env."
        )
    if "-" in CLASSIFIER_ID:
        raise ContentUnderstandingError(
            "AZURE_CU_CLASSIFIER_ID cannot contain hyphens for analyzer creation. Use underscores instead."
        )
    if not INPUT_FILE.exists():
        raise ContentUnderstandingError(
            f"Input file not found: {INPUT_FILE}. Put a PDF/image there or set AZURE_CU_INPUT_FILE in .env."
        )


def add_api_version(url: str) -> str:
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.setdefault("api-version", API_VERSION)
    return urlunparse(parsed._replace(query=urlencode(query)))


def make_url(path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return add_api_version(ENDPOINT + path)


def headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    values = {"Ocp-Apim-Subscription-Key": KEY}
    if extra:
        values.update(extra)
    return values


def response_text(response: requests.Response) -> str:
    try:
        return json.dumps(response.json(), indent=2)
    except ValueError:
        return response.text


def raise_for_bad_response(response: requests.Response, action: str) -> None:
    if response.ok:
        return
    raise ContentUnderstandingError(
        f"{action} failed: HTTP {response.status_code}\n{response_text(response)}"
    )


def cu_request(
    method: str,
    path: str,
    *,
    action: str,
    expected_statuses: tuple[int, ...] = (200,),
    **kwargs: Any,
) -> requests.Response:
    response = requests.request(
        method,
        make_url(path),
        headers=headers(kwargs.pop("headers", None)),
        timeout=60,
        **kwargs,
    )
    if response.status_code not in expected_statuses:
        raise_for_bad_response(response, action)
        raise ContentUnderstandingError(
            f"{action} returned unexpected HTTP {response.status_code}; expected {expected_statuses}."
        )
    return response


def poll_operation(operation_location: str, action: str) -> dict[str, Any]:
    deadline = time.time() + TIMEOUT_SECONDS
    poll_url = add_api_version(operation_location)

    while time.time() < deadline:
        response = requests.get(poll_url, headers=headers(), timeout=60)
        raise_for_bad_response(response, f"Polling {action}")
        payload = response.json()
        status = str(payload.get("status", "")).lower()

        if status in {"succeeded", "succeededwithwarnings"}:
            return payload
        if status in {"failed", "canceled", "cancelled"}:
            raise ContentUnderstandingError(
                f"{action} failed with status '{payload.get('status')}'.\n{json.dumps(payload, indent=2)}"
            )

        print(f"  {action}: {payload.get('status', 'running')}...")
        time.sleep(POLL_SECONDS)

    raise ContentUnderstandingError(f"Timed out waiting for {action} after {TIMEOUT_SECONDS}s.")


def operation_location(response: requests.Response, action: str) -> str:
    location = response.headers.get("Operation-Location") or response.headers.get("operation-location")
    if not location:
        raise ContentUnderstandingError(
            f"{action} did not return an Operation-Location header.\n{response_text(response)}"
        )
    return location


def optional_operation_location(response: requests.Response) -> str | None:
    return response.headers.get("Operation-Location") or response.headers.get("operation-location")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def analyze_payload(document: bytes) -> dict[str, Any]:
    return {
        "inputs": [
            {
                "data": base64.b64encode(document).decode("ascii"),
            }
        ]
    }


def analyze_layout(document: bytes) -> dict[str, Any]:
    print(f"Running layout extraction with '{LAYOUT_ANALYZER_ID}'...")
    response = cu_request(
        "POST",
        f"/contentunderstanding/analyzers/{LAYOUT_ANALYZER_ID}:analyze",
        action="Starting layout extraction",
        expected_statuses=(202,),
        headers={"Content-Type": "application/json"},
        json=analyze_payload(document),
    )
    return poll_operation(operation_location(response, "Layout extraction"), "Layout extraction")


def classifier_exists() -> bool:
    response = requests.get(
        make_url(f"/contentunderstanding/analyzers/{CLASSIFIER_ID}"),
        headers=headers(),
        timeout=60,
    )
    if response.status_code == 200:
        return True
    if response.status_code == 404:
        return False
    raise_for_bad_response(response, "Checking classifier")
    return False


def create_classifier() -> None:
    print(f"Creating classifier analyzer '{CLASSIFIER_ID}'...")
    payload = {
        "baseAnalyzerId": "prebuilt-document",
        "description": "Minimal test classifier for routing basic business documents.",
        "config": {
            "returnDetails": True,
            "enableSegment": False,
            "contentCategories": {
                "invoice": {"description": "Invoices, bills, or payment requests from vendors."},
                "receipt": {"description": "Receipts or proofs of purchase/payment."},
                "contract": {"description": "Contracts, agreements, terms, or statements of work."},
                "other": {"description": "Any document that is not an invoice, receipt, or contract."},
            },
        },
        "models": {"completion": COMPLETION_MODEL},
    }
    response = put_classifier(payload)
    if response.status_code == 400 and "DefaultsNotSet" in response_text(response):
        configure_defaults()
        response = put_classifier(payload)

    if response.status_code not in (200, 201, 202):
        raise_for_bad_response(response, "Creating classifier")
        raise ContentUnderstandingError(
            f"Creating classifier returned unexpected HTTP {response.status_code}; expected (200, 201, 202)."
        )

    location = optional_operation_location(response)
    if location:
        poll_operation(location, "Classifier creation")


def put_classifier(payload: dict[str, Any]) -> requests.Response:
    return requests.put(
        make_url(f"/contentunderstanding/analyzers/{CLASSIFIER_ID}"),
        headers=headers({"Content-Type": "application/json"}),
        json=payload,
        timeout=60,
    )


def configure_defaults() -> None:
    print("Ensuring Content Understanding model deployment defaults...")
    model_deployments = {
        "gpt-4.1": GPT41_DEPLOYMENT,
        "prebuilt-analyzer-embedding": EMBEDDING_DEPLOYMENT,
    }
    if GPT41_MINI_DEPLOYMENT:
        model_deployments["gpt-4.1-mini"] = GPT41_MINI_DEPLOYMENT

    payload = {
        "modelDeployments": model_deployments
    }
    cu_request(
        "PATCH",
        "/contentunderstanding/defaults",
        action="Configuring Content Understanding defaults",
        expected_statuses=(200, 204),
        headers={"Content-Type": "application/json"},
        json=payload,
    )


def ensure_classifier() -> None:
    print(f"Checking classifier '{CLASSIFIER_ID}'...")
    if classifier_exists():
        print("  Classifier already exists.")
        return
    create_classifier()


def classify_document(document: bytes) -> dict[str, Any]:
    print(f"Classifying document with classifier analyzer '{CLASSIFIER_ID}'...")
    response = cu_request(
        "POST",
        f"/contentunderstanding/analyzers/{CLASSIFIER_ID}:analyze",
        action="Starting classification",
        expected_statuses=(202,),
        headers={"Content-Type": "application/json"},
        json=analyze_payload(document),
    )
    return poll_operation(operation_location(response, "Classification"), "Classification")


def get_result(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result")
    if isinstance(result, dict):
        return result
    return payload


def first_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def content_objects(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = get_result(payload)
    contents = first_list(result.get("contents"))
    return [item for item in contents if isinstance(item, dict)] or [result]


def count_layout_items(layout_payload: dict[str, Any]) -> tuple[int, int, str]:
    contents = content_objects(layout_payload)
    pages = []
    tables = []
    text_parts = []
    for content in contents:
        pages.extend(first_list(content.get("pages")))
        tables.extend(first_list(content.get("tables")))
        text_parts.append(str(content.get("markdown") or content.get("content") or content.get("text") or ""))
    content = " ".join(part for part in text_parts if part)
    return len(pages), len(tables), str(content).replace("\n", " ")[:120]


def top_classification(classification_payload: dict[str, Any]) -> tuple[str, Any]:
    result = get_result(classification_payload)

    contents = content_objects(classification_payload)
    for content in contents:
        if isinstance(content.get("category"), str):
            return content["category"], content.get("confidence") or content.get("confidenceScore")

        segments = first_list(content.get("segments"))
        for segment in segments:
            if isinstance(segment, dict) and isinstance(segment.get("category"), str):
                return segment["category"], segment.get("confidence") or segment.get("confidenceScore")

    if isinstance(result.get("category"), str):
        return result["category"], result.get("confidence") or result.get("confidenceScore")

    for key in ("categories", "classifications", "documents"):
        values = first_list(result.get(key))
        if values and isinstance(values[0], dict):
            best = max(
                values,
                key=lambda item: item.get("confidence")
                or item.get("confidenceScore")
                or item.get("score")
                or 0,
            )
            return (
                str(best.get("category") or best.get("class") or best.get("label") or "unknown"),
                best.get("confidence") or best.get("confidenceScore") or best.get("score"),
            )

    return "unknown", None


def main() -> int:
    try:
        require_config()
        OUTPUT_DIR.mkdir(exist_ok=True)
        document = INPUT_FILE.read_bytes()

        configure_defaults()

        layout_payload = analyze_layout(document)
        layout_path = OUTPUT_DIR / "layout.json"
        save_json(layout_path, layout_payload)

        ensure_classifier()
        classification_payload = classify_document(document)
        classification_path = OUTPUT_DIR / "classification.json"
        save_json(classification_path, classification_payload)

        page_count, table_count, preview = count_layout_items(layout_payload)
        category, confidence = top_classification(classification_payload)

        print("\nDone")
        print(f"  Input: {INPUT_FILE}")
        print(f"  Layout: {page_count} page(s), {table_count} table(s)")
        if preview:
            print(f"  Preview: {preview}")
        print(f"  Classification: {category} (confidence: {confidence})")
        print(f"  Saved layout JSON: {layout_path}")
        print(f"  Saved classification JSON: {classification_path}")
        return 0
    except ContentUnderstandingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"HTTP ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
