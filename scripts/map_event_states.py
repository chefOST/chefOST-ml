#!/usr/bin/env python3
"""Map TubeletGraph event descriptions to one canonical MOSCATO state each."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from openai import OpenAI


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_states(
    state_dict: dict[str, Any], candidates: list[str] | None = None
) -> list[str]:
    all_states = {state.strip().lower() for state in state_dict["s2i"]}
    states = sorted(
        all_states
        if candidates is None
        else {state.strip().lower() for state in candidates}
    )
    if not states:
        raise ValueError("The MOSCATO candidate state list is empty")
    unknown = sorted(set(states) - all_states)
    if unknown:
        raise ValueError(f"Unknown MOSCATO candidate states: {unknown}")
    return states


def normalize_candidate(
    response: str, state_dict: dict[str, Any], allowed: list[str]
) -> str:
    candidate = response.strip().lower()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
    if candidate.startswith("{"):
        try:
            payload = json.loads(candidate)
            candidate = str(payload["state"]).strip().lower()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    candidate = candidate.strip("\"'` .")

    aliases = {
        key.strip().lower(): value.strip().lower()
        for key, value in state_dict.get("all2one", {}).items()
    }
    candidate = aliases.get(candidate, candidate)
    if candidate not in set(allowed):
        raise ValueError(f"Adapter returned non-canonical state {candidate!r}")
    return candidate


def semantic_evidence(raw_context: dict[str, Any]) -> list[str]:
    """Return only semantic text actually emitted by TubeletGraph."""
    evidence: list[str] = []
    semantic_keys = {"description", "prior_description", "action"}

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key in semantic_keys and isinstance(value, str) and value.strip():
            evidence.append(value.strip())

    visit(raw_context)
    return evidence


def classify_event(
    client: OpenAI,
    *,
    model: str,
    target_object: str,
    local_frame: int,
    raw_context: dict[str, Any],
    allowed: list[str],
    state_dict: dict[str, Any],
) -> tuple[str | None, list[str]]:
    evidence = semantic_evidence(raw_context)
    if not evidence:
        return None, []

    aliases = state_dict.get("one2all", {})
    state_guide = {
        state: aliases.get(state, [state])
        for state in allowed
    }
    prompt = (
        "Map a TubeletGraph prediction to the MOSCATO state vocabulary. "
        "This is prediction post-processing: you are not given and must not infer "
        "from any ground-truth annotation timeline or video frame. Use only the raw "
        "TubeletGraph semantic text. Return exactly one state from ALLOWED_STATES and no "
        "explanation. The selected state must describe the current visual state of "
        f"the target object {target_object!r} at local frame {local_frame}.\n\n"
        "Selection policy: choose the most specific state directly supported by the "
        "TubeletGraph text. An explicitly described spatial relation (for example, "
        "an object resting on a plate or dish) is more specific than a generic result "
        "such as 'placed' or 'relocated'. Use those generic labels only when no more "
        "specific relation or attribute is present in the supplied text. Do not add "
        "information or infer the person's current action.\n\n"
        f"RAW_TUBELET_SEMANTIC_TEXT={json.dumps(evidence)}\n\n"
        f"ALLOWED_STATES_WITH_ALIASES={json.dumps(state_guide, sort_keys=True)}\n\n"
        f"ALLOWED_STATES={json.dumps(allowed)}"
    )
    responses: list[str] = []
    for attempt in range(3):
        correction = ""
        if responses:
            correction = (
                "\n\nYour prior response was not an allowed state. Return one exact "
                "string copied from ALLOWED_STATES."
            )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic semantic state-label normalizer. "
                        "Follow the supplied closed vocabulary exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt + correction,
                },
            ],
            temperature=0,
        )
        raw_response = response.choices[0].message.content or ""
        responses.append(raw_response)
        try:
            return normalize_candidate(raw_response, state_dict, allowed), responses
        except ValueError:
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def map_events(
    draft: dict[str, Any],
    state_dict: dict[str, Any],
    client: OpenAI,
    model: str,
    allowed_states: list[str] | None = None,
) -> dict[str, Any]:
    mapped = copy.deepcopy(draft)
    allowed = canonical_states(state_dict, allowed_states)
    target_object = mapped["object"]
    num_local_frames = int(mapped["num_local_frames"])

    initial_state, initial_responses = classify_event(
        client,
        model=model,
        target_object=target_object,
        local_frame=0,
        raw_context=mapped.get("initial_raw", {}),
        allowed=allowed,
        state_dict=state_dict,
    )
    mapped["initial_state"] = initial_state
    mapped["initial_mapping"] = {
        "model": model,
        "raw_responses": initial_responses,
        "mapped_state": initial_state,
        "status": "mapped" if initial_state is not None else "abstained_no_tubelet_text",
    }

    last_frame = -1
    for transition in mapped.get("transitions", []):
        local_frame = int(transition["local_frame"])
        if not 0 <= local_frame < num_local_frames:
            raise ValueError(
                f"Transition frame {local_frame} is outside [0, {num_local_frames})"
            )
        if local_frame <= last_frame:
            raise ValueError("Transition frames must be unique and strictly increasing")
        last_frame = local_frame
        state, responses = classify_event(
            client,
            model=model,
            target_object=target_object,
            local_frame=local_frame,
            raw_context={"raw_nodes": transition.get("raw_nodes", [])},
            allowed=allowed,
            state_dict=state_dict,
        )
        transition["state"] = state
        transition["mapping"] = {
            "model": model,
            "raw_responses": responses,
            "mapped_state": state,
            "status": "mapped" if state is not None else "abstained_no_tubelet_text",
        }

    vocabulary_digest = hashlib.sha256(
        json.dumps(allowed, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    mapping_statuses = [mapped["initial_mapping"]["status"]] + [
        transition["mapping"]["status"]
        for transition in mapped.get("transitions", [])
    ]
    mapped_count = mapping_statuses.count("mapped")
    if mapped_count == len(mapping_statuses):
        overall_mapping_status = "mapped"
    elif mapped_count:
        overall_mapping_status = "partially_mapped"
    else:
        overall_mapping_status = "abstained_no_tubelet_text"
    mapped.setdefault("provenance", {}).update(
        {
            "state_mapping_status": overall_mapping_status,
            "state_mapping_model": model,
            "ground_truth_used_for_mapping": False,
            "video_frames_used_for_mapping": False,
            "allowed_states_sha256": vocabulary_digest,
            "allowed_states": allowed,
        }
    )
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Use the Modal tubelet-openai secret; "
            "never put the key in a repository file."
        )
    mapped = map_events(
        load_json(args.draft),
        load_json(args.state_dict),
        OpenAI(),
        args.model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapped, indent=2) + "\n", encoding="utf-8")
    print(f"Mapped events: {args.out}")
    print(f"Initial state: {mapped['initial_state']}")
    print(f"Transitions: {len(mapped.get('transitions', []))}")


if __name__ == "__main__":
    main()
