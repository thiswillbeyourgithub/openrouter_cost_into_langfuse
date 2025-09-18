#!/usr/bin/env python3
"""
OpenRouter and Langfuse API Models Fetcher

This script was created with assistance from aider.chat.
"""

from typing import Dict, Any, List, Optional
import os
import time
from datetime import datetime
import click
import requests
from tqdm import tqdm
from loguru import logger


class SimpleRateLimiter:
    """
    Simple rate limiter to control API request frequency.

    This class ensures a minimum delay between operations to respect API rate limits.
    """

    def __init__(self, delay: float = 0) -> None:
        """
        Initialize the rate limiter.

        Parameters
        ----------
        delay : float, default=0.1
            Minimum delay in seconds between operations
        """
        self.delay = delay
        self.last_call_time = 0.0

    def wait(self) -> None:
        """
        Wait if necessary to enforce the rate limit.

        This method calculates the time since the last call and sleeps
        if the minimum delay hasn't been reached.
        """
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time

        if time_since_last_call < self.delay:
            sleep_time = self.delay - time_since_last_call
            time.sleep(sleep_time)

        self.last_call_time = time.time()


# Configure logger to write to a local file
logger.add("script.log", rotation="10 MB", retention="7 days", level="INFO")


def reset_langfuse_models(
    langfuse_models: List[Dict[str, Any]],
    langfuse_host: str,
    langfuse_public_key: str,
    langfuse_secret_key: str,
    dry: bool = False,
    rate_limit_delay: float = 0,
) -> None:
    """
    Delete all Langfuse models that were created by this script.

    This function deletes any existing Langfuse models that were created by this script
    (identified by modelName starting with "openrouter_script_" and isLangfuseManaged=False).

    Parameters
    ----------
    langfuse_models : List[Dict[str, Any]]
        List of existing Langfuse models to check for deletion
    langfuse_host : str
        The Langfuse host URL
    langfuse_public_key : str
        Langfuse public key for authentication
    langfuse_secret_key : str
        Langfuse secret key for authentication
    dry : bool, default=False
        If True, only simulate the operations without making actual API calls
    rate_limit_delay : float, default=0.1
        Delay in seconds between API calls to respect rate limits
    """
    base_url = langfuse_host.rstrip("/")
    auth = (langfuse_public_key, langfuse_secret_key)
    rate_limiter = SimpleRateLimiter(delay=rate_limit_delay)

    # Find existing script-managed models to delete
    logger.info("Checking for existing script-managed models to delete...")
    models_to_delete = []
    for model in langfuse_models:
        is_managed = model.get("isLangfuseManaged", True)
        model_name = model.get("modelName", "")
        if not is_managed and model_name.startswith("openrouter_script_"):
            models_to_delete.append(model)

    logger.info(f"Found {len(models_to_delete)} models to delete")

    for model in tqdm(models_to_delete, desc="Deleting script-managed models"):
        model_id = model.get("id")
        model_name = model.get("modelName")
        if dry:
            logger.info(f"[DRY RUN] Would delete model: {model_name} (id: {model_id})")
        else:
            delete_url = f"{base_url}/api/public/models/{model_id}"
            response = requests.delete(url=delete_url, auth=auth)
            if not response.ok:
                logger.exception(response.text)
            response.raise_for_status()
            logger.info(f"Deleted model: {model_name} (id: {model_id})")
            # Rate limiting to respect API limits
            rate_limiter.wait()


def generate_unique_model_name(base_name: str, model_id: str, used_names: set) -> str:
    """
    Generate a unique model name, handling duplicates with specific rules.

    If "thinking" is in the model_id but not in the base_name, adds "_thinking".
    Otherwise adds "_2", "_3", etc. for duplicates.

    Parameters
    ----------
    base_name : str
        The base model name to make unique
    model_id : str
        The original model ID to check for "thinking"
    used_names : set
        Set of already used model names

    Returns
    -------
    str
        A unique model name
    """
    if base_name not in used_names:
        return base_name

    # Special case: if "thinking" is in the id but not in the base name
    if "thinking" in model_id.lower() and "thinking" not in base_name.lower():
        thinking_name = f"{base_name}_thinking"
        if thinking_name not in used_names:
            return thinking_name

    # Regular numbering for other duplicates
    counter = 2
    while True:
        candidate_name = f"{base_name}_{counter}"
        if candidate_name not in used_names:
            return candidate_name
        counter += 1


def manage_langfuse_models(
    openrouter_models: List[Dict[str, Any]],
    langfuse_models: List[Dict[str, Any]],
    langfuse_host: str,
    langfuse_public_key: str,
    langfuse_secret_key: str,
    start_date: str,
    dry: bool = False,
    rate_limit_delay: float = 0,
) -> None:
    """
    Manage Langfuse models by deleting old script-managed ones and creating new ones from OpenRouter.

    This function first deletes any existing Langfuse models that were created by this script
    (identified by modelName starting with "openrouter_script_" and isLangfuseManaged=False),
    then creates new models based on the OpenRouter model data.

    Parameters
    ----------
    openrouter_models : List[Dict[str, Any]]
        List of OpenRouter models to create in Langfuse
    langfuse_models : List[Dict[str, Any]]
        List of existing Langfuse models to check for deletion
    langfuse_host : str
        The Langfuse host URL
    langfuse_public_key : str
        Langfuse public key for authentication
    langfuse_secret_key : str
        Langfuse secret key for authentication
    start_date : str
        Start date for the models in ISO format (YYYY-MM-DD)
    dry : bool, default=False
        If True, only simulate the operations without making actual API calls
    rate_limit_delay : float, default=0.1
        Delay in seconds between API calls to respect rate limits
    """
    base_url = langfuse_host.rstrip("/")
    auth = (langfuse_public_key, langfuse_secret_key)
    rate_limiter = SimpleRateLimiter(delay=rate_limit_delay)

    # Delete existing script-managed models
    logger.info("Checking for existing script-managed models to delete...")
    models_to_delete = []
    for model in langfuse_models:
        is_managed = model.get("isLangfuseManaged", True)
        model_name = model.get("modelName", "")
        if not is_managed and model_name.startswith("openrouter_script_"):
            models_to_delete.append(model)

    logger.info(f"Found {len(models_to_delete)} models to delete")

    for model in tqdm(models_to_delete, desc="Deleting existing script-managed models"):
        model_id = model.get("id")
        model_name = model.get("modelName")
        if dry:
            logger.info(f"[DRY RUN] Would delete model: {model_name} (id: {model_id})")
        else:
            delete_url = f"{base_url}/api/public/models/{model_id}"
            response = requests.delete(url=delete_url, auth=auth)
            if not response.ok:
                logger.exception(response.text)
            response.raise_for_status()
            logger.info(f"Deleted model: {model_name} (id: {model_id})")
            # Rate limiting to respect API limits
            rate_limiter.wait()

    # Create new models from OpenRouter data
    logger.info(f"Creating {len(openrouter_models)} new models from OpenRouter data...")

    # Track used model names to handle duplicates
    used_model_names = set()

    # Add existing Langfuse model names to the set to avoid conflicts
    for existing_model in langfuse_models:
        existing_name = existing_model.get("modelName")
        if existing_name:
            used_model_names.add(existing_name)

    for openrouter_model in tqdm(
        openrouter_models, desc="Creating new models from OpenRouter"
    ):
        model_id = openrouter_model.get("id")
        canonical_slug = openrouter_model.get("canonical_slug")
        pricing = openrouter_model.get("pricing", {})

        # Skip if missing essential data
        if not model_id or not canonical_slug or not pricing:
            logger.info(f"Skipping model {model_id} due to missing data")
            continue

        # Determine tokenizer based on canonical_slug
        tokenizer_id = "claude" if "anthropic" in canonical_slug.lower() else "openai"

        # Generate unique model name
        base_model_name = f"openrouter_script_{canonical_slug}"
        unique_model_name = generate_unique_model_name(
            base_name=base_model_name, model_id=model_id, used_names=used_model_names
        )
        used_model_names.add(unique_model_name)

        # Prepare model data for Langfuse
        model_data = {
            "matchPattern": f"(?i).*{model_id}$",
            "modelName": unique_model_name,
            "inputPrice": float(pricing.get("prompt")),
            "outputPrice": float(pricing.get("completion")),
            "startDate": start_date,
            "tokenizerId": tokenizer_id,
            "unit": "TOKENS",
        }

        if dry:
            logger.info(f"[DRY RUN] Would create model: {model_data['modelName']}")
            logger.info(f"  Pattern: {model_data['matchPattern']}")
            logger.info(f"  Input price: {model_data['inputPrice']}")
            logger.info(f"  Output price: {model_data['outputPrice']}")
            logger.info(f"  Tokenizer: {model_data['tokenizerId']}")
            logger.info(f"  Start date: {model_data['startDate']}")
        else:
            create_url = f"{base_url}/api/public/models"
            try:
                response = requests.post(url=create_url, json=model_data, auth=auth)
                if not response.ok:
                    logger.exception(response.text)
                response.raise_for_status()
                logger.info(f"Created model: {model_data['modelName']}")
            except Exception as e:
                logger.error(f"Failed to create model. Model data: {model_data}")
                raise
            # Rate limiting to respect API limits
            time.sleep(rate_limit_delay)


def fetch_all_models(
    langfuse_host: Optional[str] = None,
    langfuse_public_key: Optional[str] = None,
    langfuse_secret_key: Optional[str] = None,
    rate_limit_delay: float = 0,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch models data from both OpenRouter and Langfuse APIs.

    This function combines fetching from both APIs into a single call,
    making it easier to get all model data at once.

    Parameters
    ----------
    langfuse_host : Optional[str]
        The Langfuse host URL to fetch models from. If None, skips Langfuse API.
    langfuse_public_key : Optional[str]
        Langfuse public key for authentication
    langfuse_secret_key : Optional[str]
        Langfuse secret key for authentication
    rate_limit_delay : float, default=0.1
        Delay in seconds between API calls to respect rate limits

    Returns
    -------
    tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
        A tuple containing (openrouter_models, langfuse_models)

    Raises
    ------
    requests.RequestException
        If any API request fails
    """
    # Fetch OpenRouter models
    logger.info("Fetching models from OpenRouter API...")
    openrouter_url = "https://openrouter.ai/api/v1/models"

    try:
        response = requests.get(url=openrouter_url)
        response.raise_for_status()
        data = response.json()

        # Extract the 'data' field from the response
        models_data = data.get("data", [])

        # Create filtered list with only the required keys, excluding models with -1 or 1 pricing
        openrouter_models = []
        for model in models_data:
            pricing = model.get("pricing")

            # Skip models with -1 or 0 pricing for completion or prompt
            if pricing:
                completion_price = pricing.get("completion")
                prompt_price = pricing.get("prompt")
                if (
                    float(completion_price) == -1
                    or float(prompt_price) == -1
                    or float(completion_price) == 0
                    or float(prompt_price) == 0
                ):
                    continue

            filtered_model = {
                "id": model.get("id"),
                "canonical_slug": model.get("canonical_slug"),
                "pricing": pricing,
            }
            openrouter_models.append(filtered_model)

        logger.info(
            f"Successfully fetched {len(models_data)} models from OpenRouter API"
        )
        logger.info(
            f"Filtered to {len(openrouter_models)} models with id, canonical_slug, and pricing"
        )

    except requests.RequestException as e:
        logger.error(f"Error fetching data from OpenRouter API: {e}")
        raise

    # Fetch Langfuse models if host is provided
    langfuse_models = []
    if langfuse_host:
        logger.info("Fetching models from Langfuse API...")
        logger.info(
            f"Langfuse config - Public key: {'✓' if langfuse_public_key else '✗'}"
        )
        logger.info(
            f"Langfuse config - Secret key: {'✓' if langfuse_secret_key else '✗'}"
        )
        logger.info(f"Langfuse config - Host: {langfuse_host}")

        langfuse_url = f"{langfuse_host.rstrip('/')}/api/public/models"
        page = 1
        all_models = []

        try:
            while True:
                logger.info(f"Fetching page {page}...")
                response = requests.get(
                    url=langfuse_url,
                    params={"page": page},
                    auth=(langfuse_public_key, langfuse_secret_key),
                )
                response.raise_for_status()
                data = response.json()

                # Handle the case where data might be directly a list or nested under a key
                page_models = []
                if isinstance(data, dict) and "data" in data:
                    page_models = data["data"]
                elif isinstance(data, list):
                    page_models = data
                else:
                    page_models = []

                # If no models returned on this page, we've reached the end
                if not page_models:
                    break

                all_models.extend(page_models)
                logger.info(f"Fetched {len(page_models)} models from page {page}")
                page += 1
                # Rate limiting to respect API limits
                time.sleep(rate_limit_delay)

            langfuse_models = all_models
            logger.info(
                f"Successfully fetched {len(langfuse_models)} total models from Langfuse API across {page - 1} pages"
            )

        except requests.RequestException as e:
            logger.error(f"Error fetching data from Langfuse API: {e}")
            raise
    else:
        logger.info("Skipping Langfuse API (no host provided)")

    return openrouter_models, langfuse_models


@click.command()
@click.option(
    "--langfuse-public-key",
    type=str,
    default=None,
    help="Langfuse public key. If not provided, loads from LANGFUSE_PUBLIC_KEY environment variable.",
)
@click.option(
    "--langfuse-secret-key",
    type=str,
    default=None,
    help="Langfuse secret key. If not provided, loads from LANGFUSE_SECRET_KEY environment variable.",
)
@click.option(
    "--langfuse-host",
    type=str,
    default=None,
    help="Langfuse host URL. If not provided, loads from LANGFUSE_HOST environment variable.",
)
@click.option(
    "--dry",
    is_flag=True,
    default=False,
    help="Dry run mode. Only simulate model management operations without making actual API calls.",
)
@click.option(
    "--reset",
    is_flag=True,
    default=False,
    help="Reset mode. Only delete existing script-managed models (starting with 'openrouter_script_') without creating new ones.",
)
@click.option(
    "--start-date",
    type=str,
    default=None,
    help="Start date for the models in dd-mm-yyyy format. If not provided, uses today's date.",
)
@click.option(
    "--rate-limit-delay",
    type=float,
    default=0,
    help="Delay in seconds between API calls to respect rate limits. Default is 0 seconds (no rate limiting).",
)
def main(
    langfuse_public_key: Optional[str],
    langfuse_secret_key: Optional[str],
    langfuse_host: Optional[str],
    dry: bool,
    reset: bool,
    start_date: Optional[str],
    rate_limit_delay: float,
) -> None:
    """
    CLI entry point for fetching OpenRouter and Langfuse models and managing Langfuse models.

    This command fetches the available models from both OpenRouter API and Langfuse API,
    filters them to essential fields (id, canonical_slug, pricing),
    and manages Langfuse models by deleting old script-managed ones and creating new ones.

    Parameters
    ----------
    langfuse_public_key : Optional[str]
        Langfuse public key. Falls back to LANGFUSE_PUBLIC_KEY env var if None.
    langfuse_secret_key : Optional[str]
        Langfuse secret key. Falls back to LANGFUSE_SECRET_KEY env var if None.
    langfuse_host : Optional[str]
        Langfuse host URL. Falls back to LANGFUSE_HOST env var if None.
    dry : bool
        If True, only simulate model management operations without making actual API calls.
    reset : bool
        If True, only delete existing script-managed models without creating new ones.
    start_date : Optional[str]
        Start date for the models in dd-mm-yyyy format. Falls back to today's date if None.
    rate_limit_delay : float
        Delay in seconds between API calls to respect rate limits.
    """
    # Load from environment variables if not provided as arguments
    if langfuse_public_key is None:
        langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    if langfuse_secret_key is None:
        langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    if langfuse_host is None:
        langfuse_host = os.getenv("LANGFUSE_HOST")

    # Parse start date or use today's date
    if start_date:
        try:
            # Parse dd-mm-yyyy format and convert to ISO format (YYYY-MM-DD)
            parsed_date = datetime.strptime(start_date, "%d-%m-%Y").date()
            start_date_iso = parsed_date.isoformat()
        except ValueError as e:
            logger.error(
                f"Error parsing start date '{start_date}': {e}. Expected format: dd-mm-yyyy"
            )
            return
    else:
        start_date_iso = datetime.now().date().isoformat()

    # Handle reset mode - only delete script-managed models
    if reset:
        if not langfuse_host or not langfuse_public_key or not langfuse_secret_key:
            logger.error(
                "Error: Langfuse credentials and host are required for reset mode"
            )
            return

        # For reset mode, we only need Langfuse models
        logger.info("Reset mode: Only fetching Langfuse models...")
        _, langfuse_models = fetch_all_models(
            langfuse_host=langfuse_host,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            rate_limit_delay=rate_limit_delay,
        )

        logger.info("Resetting Langfuse models...")
        reset_langfuse_models(
            langfuse_models=langfuse_models,
            langfuse_host=langfuse_host,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            dry=dry,
            rate_limit_delay=rate_limit_delay,
        )
        logger.info("Reset complete.")
        return

    # Normal mode - fetch from both APIs and manage models
    # Note: Langfuse host is optional since we also fetch from OpenRouter
    # If no Langfuse host is provided, we'll only fetch from OpenRouter

    # Fetch data from both APIs with a single function call
    openrouter_models, langfuse_models = fetch_all_models(
        langfuse_host=langfuse_host,
        langfuse_public_key=langfuse_public_key,
        langfuse_secret_key=langfuse_secret_key,
        rate_limit_delay=rate_limit_delay,
    )

    # Manage Langfuse models if we have the necessary credentials and data
    if langfuse_host and langfuse_public_key and langfuse_secret_key:
        logger.info("Managing Langfuse models...")
        manage_langfuse_models(
            openrouter_models=openrouter_models,
            langfuse_models=langfuse_models,
            langfuse_host=langfuse_host,
            langfuse_public_key=langfuse_public_key,
            langfuse_secret_key=langfuse_secret_key,
            start_date=start_date_iso,
            dry=dry,
            rate_limit_delay=rate_limit_delay,
        )
    else:
        logger.info("Skipping Langfuse model management (missing credentials or host)")


if __name__ == "__main__":
    main()
