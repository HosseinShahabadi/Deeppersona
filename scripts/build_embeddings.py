#!/usr/bin/env python3
"""Rebuild the attribute-embedding database from the attribute taxonomy.

The `.pkl` files shipped in `data/` upstream are truncated (both are cut at
exactly 6.5 MiB and end mid-array with no pickle STOP opcode), so they cannot
be loaded. `select_attributes.py` swallows that failure and then silently
returns an empty attribute list, which produces hollow personas with no error.

This script regenerates them locally with a sentence-transformers model, so the
pipeline no longer depends on a remote embeddings endpoint at all.

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --model BAAI/bge-small-en-v1.5
    python scripts/build_embeddings.py --attributes data/large_attributes.json \
                                       --output data/attribute_embeddings.pkl
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

# Make the package importable when run as a plain script from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "generate_user_profile"))

import embeddings as emb  # noqa: E402


def flatten_attributes(attributes: dict) -> list:
    """Flatten the nested taxonomy into dot-joined leaf paths.

    This mirrors AttributeSelector._flatten_attributes exactly: a leaf is an
    empty dict or a non-dict value. The keys produced here must match the paths
    the selector looks up, or every lookup misses.
    """
    result = []

    def _walk(node, prefix=""):
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                if not value:
                    result.append(path)
                else:
                    _walk(value, path)
            else:
                result.append(path)

    _walk(attributes)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--attributes",
        default=os.path.join(REPO_ROOT, "data", "large_attributes.json"),
        help="Path to the attribute taxonomy JSON.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "data", "attribute_embeddings.pkl"),
        help="Where to write the embeddings pickle.",
    )
    parser.add_argument(
        "--model", default=None,
        help="Override DEEPPERSONA_EMBED_MODEL for this run.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    if args.model:
        os.environ["DEEPPERSONA_EMBED_MODEL"] = args.model
        emb.MODEL_NAME = args.model

    if not os.path.exists(args.attributes):
        print(f"ERROR: attributes file not found: {args.attributes}")
        return 1

    with open(args.attributes, "r", encoding="utf-8") as f:
        attributes = json.load(f)

    paths = flatten_attributes(attributes)
    if not paths:
        print("ERROR: no leaf attribute paths found — is the taxonomy empty?")
        return 1
    print(f"Flattened {len(paths)} leaf attribute paths from {args.attributes}")

    vectors = emb.embed_passages(paths, batch_size=args.batch_size)
    print(f"Encoded {vectors.shape[0]} paths -> {vectors.shape[1]}-dim vectors "
          f"using {emb.MODEL_NAME}")

    # Key names match what AttributeSelector._load_embeddings expects. It
    # accepts either 'attribute_paths' or 'paths', and requires 'embeddings'
    # to be an np.ndarray.
    payload = {
        "attribute_paths": paths,
        "embeddings": vectors,
        "model": emb.MODEL_NAME,
        "dim": int(vectors.shape[1]),
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.1f} MiB)")

    # Verify the file round-trips — the whole point of this script is that the
    # upstream artifact did not.
    with open(args.output, "rb") as f:
        check = pickle.load(f)
    assert len(check["attribute_paths"]) == len(paths)
    assert isinstance(check["embeddings"], np.ndarray)
    assert check["embeddings"].shape == vectors.shape
    print("Verified: pickle reloads cleanly and shapes match.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
