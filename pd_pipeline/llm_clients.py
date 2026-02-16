"""
LLM API client functions for calling various language model providers.

Supports:
- Anthropic (Claude)
- OpenAI (GPT, o-series via Responses API)
- OpenRouter (multi-provider gateway)
"""

import os
import time
import types
import json
import requests
from typing import Dict, List, Tuple, Optional, Any
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Import config
from config.models import (
    MAX_TOKENS, MODEL_TEMP, REASONING_EFFORT, 
    VERBOSITY, RESPONSES_MAX_OUTPUT_TOKENS
)

def call_prompt(provider, model, user_prompts, system_prompt, prefill_text = "{"):
    """
    Calls the LLM API with a list of strings and a system prompt.
    Args:
        provider (str): one of "anthropic", "openai", "openrouter" (via OpenAI SDK)
        model (str): the model for use with the API, e.g. "claude-sonnet-4-20250514"
        user_prompts (list of str): A list of strings to be sent to the API. Each will be treated as a separate user message.
        Currently designed so any prompt that needs to be cached is always the first message.
        For processing multiple genes of the same paper it is recommended to cache the paper text.
        system_prompt (str): The system prompt to be used.

    Returns:
        str: The response from the LLM API.
    """

    messages = []
    system_message = None
    local_max_tokens = max_tokens
    if model == "claude-3-5-haiku-20241022":
        local_max_tokens = 8192
    if system_prompt:
        system_message = system_prompt  # Store system prompt separately for Claude

    if len(user_prompts) > 1:
        cache_prompt = user_prompts[0]
        prompt = user_prompts[1]
        if provider == "anthropic":
            # Anthropic supports caching
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": cache_prompt,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": prompt},
                ],
            })
        else:
            # OpenRouter/OpenAI: concatenate
            messages.append({"role": "user", "content": cache_prompt + "\n\n" + prompt})
    else:
        prompt = user_prompts[0]
        messages.append({"role": "user", "content": prompt})


    # Defaults to get key from environment variable
    if provider == "anthropic":
        client = Anthropic()
        # Add prefill if specified (only for Claude)
        if prefill_text:
            messages.append({"role": "assistant", "content": prefill_text})
        # Create the request parameters
        request_params = {
            "model": model,
            "messages": messages,
            "max_tokens": local_max_tokens,
            "temperature": model_temp,
        }

        # Add system prompt if it exists
        if system_message:
            request_params["system"] = system_message

        start = time.perf_counter()  # time the call
        response = client.messages.create(**request_params)
        elapsed = time.perf_counter() - start  # end time


        result = response.content[0].text if response.content else ""
        # If we used prefill, prepend it to the response
        if prefill_text:
            result = prefill_text + result

        usage = _extract_usage("anthropic", response)
        return result, usage, elapsed  # return usage and time too




    elif provider == "openrouter":

        openai_messages = []

        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        openai_messages.extend(messages)

        # Create OpenAI client with OpenRouter configuration - simplified

        from openai import OpenAI as OpenAIClient

        client = OpenAIClient(

            base_url="https://openrouter.ai/api/v1",

            api_key=OPENROUTER_API_KEY,

            default_headers={

                # "HTTP-Referer": "https://github.com/yourusername",  # Optional
                #
                # "X-Title": "PD Curation Pipeline",  # Optional

            }

        )

        start = time.perf_counter()

        request_params = {

            "model": model,

            "messages": openai_messages,

            "max_tokens": local_max_tokens,

            "temperature": model_temp,

            "timeout": 180,  # Add explicit timeout

        }

        # Try with reasoning first, fallback if not supported

        try:

            # First attempt: with reasoning/thinking enabled

            response = client.chat.completions.create(

                **request_params,

                extra_body={

                    "provider": {

                        "allow_fallbacks": True,  # Allow OpenRouter to fallback if primary fails

                    },

                }

            )

        except Exception as e:

            error_msg = str(e).lower()

            # If reasoning not supported or other API error, retry without extras

            if "reasoning" in error_msg or "extra_body" in error_msg or "provider" in error_msg:

                print(f"  ⚠️  Retrying without extended params: {e}")

                response = client.chat.completions.create(**request_params)

            else:

                raise

        elapsed = time.perf_counter() - start

        result = response.choices[0].message.content

        # Extract usage from response

        usage_obj = getattr(response, 'usage', None)

        if usage_obj:

            usage = {

                "input": getattr(usage_obj, 'prompt_tokens', 0),

                "output": getattr(usage_obj, 'completion_tokens', 0),

                "total": getattr(usage_obj, 'total_tokens', 0)

            }

        else:

            usage = {"input": 0, "output": 0, "total": 0}

        return result, usage, elapsed

    elif (provider == "openai"):

        # Build chat-style messages first

        openai_messages = []

        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        openai_messages.extend(messages)

        client = OpenAI()

        start = time.perf_counter()

        if _is_openai_responses_model(model):

            # Responses API

            resp_input = _to_responses_input(openai_messages)

            request_params = {

                "model": model,

                "input": resp_input,

                "instructions": system_prompt or "",

                "max_output_tokens": min(local_max_tokens, RESPONSES_MAX_OUTPUT_TOKENS),

                "timeout_s": _get_http_timeout(),

                # Correct placement for verbosity: under "text"

                "text": {

                    "format": {"type": "text"},

                    **({"verbosity": VERBOSITY} if VERBOSITY else {}),

                },

            }

            # Reasoning control stays top-level

            if REASONING_EFFORT:
                request_params["reasoning"] = {"effort": REASONING_EFFORT}

            # Keep temperature only if supported

            if _responses_supports_temperature(model) and (model_temp is not None):
                request_params["temperature"] = model_temp

            try:

                response = _openai_responses_create(client, **request_params)

            except RuntimeError as e:

                msg = str(e)

                # If API rejects verbosity, retry once without it

                if "Unknown parameter" in msg and "verbosity" in msg:

                    request_params.pop("text", None)  # drop verbosity + format block

                    # Minimal safe text config without verbosity

                    request_params["text"] = {"format": {"type": "text"}}

                    response = _openai_responses_create(client, **request_params)

                else:

                    raise

            elapsed = time.perf_counter() - start

            result = getattr(response, "output_text", "") or ""

            if not result:

                parts = []

                for item in getattr(response, "output", []) or []:

                    for c in item.get("content", []) or []:

                        t = c.get("text") if isinstance(c, dict) else getattr(c, "text", None)

                        if t:
                            parts.append(t)

                result = "".join(parts)

            usage = _extract_usage("openai-responses", response)


        else:

            # Chat Completions API

            try:

                response = client.chat.completions.create(

                    model=model,

                    messages=openai_messages,

                    max_tokens=local_max_tokens,

                    temperature=model_temp,

                    timeout=_get_http_timeout(),

                )

            except TypeError:

                response = client.chat.completions.create(

                    model=model,

                    messages=openai_messages,

                    max_tokens=local_max_tokens,

                    temperature=model_temp,

                )

            elapsed = time.perf_counter() - start

            result = response.choices[0].message.content

            usage = _extract_usage("openai-chat", response)

        return result, usage, elapsed
    else:
        raise ValueError("unsupported LLM provider, please use 'anthropic' or 'openai'")


########################################################################################################################
# STAGE 1: GETTING PAPER IN PLAINTEXT                                                                                  #
########################################################################################################################
# CONSTANTS #
# The Base URL for PubMed API to fetch BioC JSON format.

def _extract_usage(provider, response):
    """
    Return {"input": int, "output": int, "total": int} from SDK response objects.
    Supports Anthropic, OpenAI Chat Completions, and OpenAI Responses API.
    """
    try:
        if provider == "anthropic":
            u = response.usage
            return {"input": u.input_tokens,
                    "output": u.output_tokens,
                    "total": u.input_tokens + u.output_tokens}
        elif provider in ("openai", "openai-chat"):
            u = response.usage
            return {"input": u.prompt_tokens,
                    "output": u.completion_tokens,
                    "total": u.total_tokens}
        elif provider in ("openai-responses",):
            u = response.usage
            # Responses API typically exposes input_tokens/output_tokens[/total_tokens]
            input_tokens = getattr(u, "input_tokens", 0)
            output_tokens = getattr(u, "output_tokens", 0)
            total_tokens = getattr(u, "total_tokens", input_tokens + output_tokens)
            return {"input": input_tokens, "output": output_tokens, "total": total_tokens}
    except AttributeError:
        pass
    return {"input": 0, "output": 0, "total": 0}


def _to_responses_input(messages):
    """
    Convert [{"role":..,"content": <str|blocks>}, ...] into Responses API format:
    each content block must use type 'input_text' (or 'input_image').
    """
    out = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        blocks = []
        if isinstance(content, str):
            blocks = [{"type": "input_text", "text": content}]
        elif isinstance(content, list):
            for b in content:
                btype = b.get("type")
                if btype == "text":
                    blocks.append({"type": "input_text", "text": b.get("text", "")})
                elif btype in ("image_url", "input_image"):
                    # minimal mapping from Chat image part to Responses
                    url = b.get("image_url")
                    if isinstance(url, dict):
                        url = url.get("url")
                    if isinstance(url, str) and url:
                        blocks.append({"type": "input_image", "image_url": url})
                # ignore unknown block types
        else:
            blocks = [{"type": "input_text", "text": str(content)}]
        out.append({"role": role, "content": blocks})
    return out


def _responses_supports_temperature(model: str) -> bool:
    """
    Return False for newer deterministic Responses models that reject `temperature`.
    Extend the prefixes as needed if you see similar errors with other models.
    """
    no_temp_prefixes = ("gpt-5")  # add more if needed
    return not any(model.startswith(p) for p in no_temp_prefixes)
# helper to call prompt

def _get_http_timeout(default: int = 180) -> int:
    """
    Return the HTTP read-timeout in seconds. Override with OPENAI_HTTP_TIMEOUT env var.
    """
    import os
    try:
        return int(os.getenv("OPENAI_HTTP_TIMEOUT", default))
    except Exception:
        return default

def _is_openai_responses_model(model: str) -> bool:
    """
    Return True if this OpenAI model expects the Responses API and
    'max_completion_tokens' instead of Chat Completions 'max_tokens'.
    """
    prefixes = ("gpt-5", "o4", "gpt-4.1", "gpt-4o-")
    return any(model.startswith(p) for p in prefixes)


def _openai_responses_create(client, **request_params):
    """
    Create a Responses API call using the SDK if available; otherwise POST to /v1/responses
    with retries and exponential backoff. Returns an object mimicking SDK:
    .output_text, .output, .usage
    """
    import os, json, time, types, requests
    from requests.adapters import HTTPAdapter
    try:
        # Retry is vendored by requests via urllib3
        from urllib3.util.retry import Retry
    except Exception:
        Retry = None  # fallback manual backoff if urllib3 Retry missing

    timeout_s = request_params.pop("timeout_s", None) or _get_http_timeout()
    api_key = os.getenv("OPENAI_API_KEY") or getattr(client, "api_key", None)
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set, and client.responses is unavailable.")

    # 1) Try SDK first (some SDKs support .responses.create, some also support 'timeout')
    resp_client = getattr(client, "responses", None)
    if resp_client and hasattr(resp_client, "create"):
        try:
            return resp_client.create(**request_params, timeout=timeout_s)
        except TypeError:
            # Older SDKs may not accept timeout kwarg
            return resp_client.create(**request_params)

    # 2) HTTP fallback with retries + backoff
    session = requests.Session()
    if Retry is not None:
        retry = Retry(
            total=4,                # total attempts (1 original + 3 retries)
            backoff_factor=1.5,     # 0, 1.5, 3.0, 4.5...
            status_forcelist=[408, 429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

    url = "https://api.openai.com/v1/responses"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    attempts = 0
    last_err = None
    while attempts < 4:
        attempts += 1
        try:
            r = session.post(
                url,
                headers=headers,
                data=json.dumps(request_params),
                timeout=(15, timeout_s),  # (connect, read) timeouts
            )
            if r.status_code >= 400:
                # Bubble up server error (helps debugging)
                raise RuntimeError(f"Responses API error {r.status_code}: {r.text}")

            data = r.json()
            ns = types.SimpleNamespace()
            u = data.get("usage") or {}
            ns.usage = types.SimpleNamespace(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                total_tokens=u.get("total_tokens", u.get("input_tokens", 0) + u.get("output_tokens", 0)),
            )
            ns.output_text = data.get("output_text", "")
            ns.output = data.get("output", [])
            ns.raw = data
            return ns
        except (requests.ReadTimeout, requests.ConnectionError) as e:
            last_err = e
            # Simple exponential backoff (in addition to urllib3 retry)
            time.sleep(1.5 * attempts)
        except Exception as e:
            # Non-timeout errors: bail immediately
            raise

    raise RuntimeError(f"Responses API timed out after {attempts} attempts: {last_err}")

# helper to  track token usage for costing

