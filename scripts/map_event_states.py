#!/usr/bin/env python3
"""Map TubeletGraph event descriptions to one canonical MOSCATO state each."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
from typing import Any

from openai import OpenAI


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_states(state_dict: dict[str, Any]) -> list[str]:
    states = sorted(state.strip().lower() for state in state_dict["s2i"])
    if not states:
        raise ValueError("MOSCATO state_dict.s2i is empty")
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


def image_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
            "detail": "low",
        },
    }


def classify_event(
    client: OpenAI,
    *,
    model: str,
    target_object: str,
    local_frame: int,
    raw_context: dict[str, Any],
    frame_path: Path,
    allowed: list[str],
    state_dict: dict[str, Any],
) -> tuple[str, list[str]]:
    prompt = (
        "Map a TubeletGraph prediction to the MOSCATO state vocabulary. "
        "This is prediction post-processing: you are not given and must not infer "
        "from any ground-truth annotation timeline. Inspect the video frame and raw "
        "TubeletGraph context. Return exactly one state from ALLOWED_STATES and no "
        "explanation. The selected state must describe the current visual state of "
        f"the target object {target_object!r} at local frame {local_frame}.\n\n"
        f"RAW_TUBELET_CONTEXT={json.dumps(raw_context, sort_keys=True)}\n\n"
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
                        "You are a deterministic visual state classifier. "
                        "Follow the supplied closed vocabulary exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + correction},
                        image_payload(frame_path),
                    ],
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
    frames_dir: Path,
    client: OpenAI,
    model: str,
) -> dict[str, Any]:
    mapped = copy.deepcopy(draft)
    allowed = canonical_states(state_dict)
    target_object = mapped["object"]
    num_local_frames = int(mapped["num_local_frames"])

    initial_state, initial_responses = classify_event(
        client,
        model=model,
        target_object=target_object,
        local_frame=0,
        raw_context=mapped.get("initial_raw", {}),
        frame_path=frames_dir / "0000000.jpg",
        allowed=allowed,
        state_dict=state_dict,
    )
    mapped["initial_state"] = initial_state
    mapped["initial_mapping"] = {
        "model": model,
        "raw_responses": initial_responses,
        "mapped_state": initial_state,
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
            frame_path=frames_dir / f"{local_frame:07d}.jpg",
            allowed=allowed,
            state_dict=state_dict,
        )
        transition["state"] = state
        transition["mapping"] = {
            "model": model,
            "raw_responses": responses,
            "mapped_state": state,
        }

    vocabulary_digest = hashlib.sha256(
        json.dumps(allowed, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    mapped.setdefault("provenance", {}).update(
        {
            "state_mapping_status": "mapped",
            "state_mapping_model": model,
            "ground_truth_used_for_mapping": False,
            "allowed_states_sha256": vocabulary_digest,
        }
    )
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
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
        args.frames,
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
