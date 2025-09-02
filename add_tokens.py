#!/usr/bin/env python3
"""
Add hop-depth function tokens to a model tokenizer and resize model embeddings.

Changes from the previous design:
- Tokens now reflect hop depth within function families instead of base/wrapper pairs.
- Families are labeled A..J, where family A maps to constant 5, B to 7, ..., J to 23.
- Tokens are named like <A0>, <A1>, <A2>, ..., up to --max-depth per family.

CLI changes:
- --num-functions now represents the number of distinct function families (A..J)
- Added --max-depth to control the maximum hop depth (inclusive). Total tokens added = num_functions * (max_depth + 1)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from pathlib import Path
import json
import argparse

set_seed(0)

def generate_function_tokens(num_families: int, max_depth: int):
    """Generate hop-depth function tokens.

    - Families are labeled A..J (at most 10 families)
    - For each family F and depth d in [0, max_depth], create token <F{d}>
    - Total tokens = num_families * (max_depth + 1)
    """
    if num_families < 1:
        raise ValueError("--num-functions (num_families) must be >= 1")
    if max_depth < 0:
        raise ValueError("--max-depth must be >= 0")

    family_letters = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'
    ]
    if num_families > len(family_letters):
        raise ValueError(f"Not enough family letters for {num_families} families (max {len(family_letters)})")

    tokens = []
    for i in range(num_families):
        fam = family_letters[i]
        for d in range(max_depth + 1):
            tokens.append(f"<{fam}{d}>")
    return tokens

def get_token_descriptions(tokens):
    """Generate descriptions for hop-depth tokens by family and depth."""
    # Group tokens by family letter
    by_family = {}
    for tok in tokens:
        # Token format: <Xn>
        try:
            inner = tok.strip('<>')
            fam = inner[0]
            depth = int(inner[1:])
        except Exception:
            fam, depth = '?', -1
        by_family.setdefault(fam, []).append((depth, tok))

    descriptions = []
    family_order = sorted(by_family.keys())
    for fam in family_order:
        entries = sorted(by_family[fam], key=lambda x: x[0])
        const_value = 5 + 2 * (ord(fam) - ord('A'))  # A->5, B->7, ..., J->23
        tok_list = ", ".join(tok for _, tok in entries)
        descriptions.append(f"  - Family {fam} (maps to constant {const_value}): {tok_list}")
    return descriptions

def main():
    parser = argparse.ArgumentParser(description="Add function tokens to OLMo model")
    parser.add_argument("--num-functions", type=int, default=7,
                       help="Number of distinct function families (A..J). Default: 7")
    parser.add_argument("--max-depth", type=int, default=2,
                       help="Maximum hop depth (inclusive) per family. Total tokens = num_functions * (max_depth + 1)")
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B-Instruct",
                       help="Model checkpoint to use. Default: allenai/OLMo-2-0425-1B-Instruct")
    parser.add_argument("--output-dir", type=str, 
                       required=True,
                       help="Output directory for the modified model")
    
    args = parser.parse_args()
    
    # Generate hop-depth function tokens
    try:
        specials = generate_function_tokens(args.num_functions, args.max_depth)
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    
    # Update output directory name to reflect number of tokens
    if "4TOKENS" in args.output_dir:
        new_output_dir = args.output_dir.replace("4TOKENS", f"{args.num_functions}TOKENS")
    else:
        new_output_dir = args.output_dir
    
    print(f"Loading model: {args.model}")
    total_tokens = len(specials)
    print(f"Adding {total_tokens} function tokens across {args.num_functions} families with max_depth={args.max_depth}:")
    print(specials)
    print(f"Output directory: {new_output_dir}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # --- 1. Add your new tokens ---------------------------------------------------
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": specials})
    print("Added", num_added, "tokens. New vocab:", len(tokenizer))

    # Good idea: if pad_token is missing, set one (avoid training bugs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # safe fallback

    # --- 2. Load model & resize ---------------------------------------------------
    print("Loading model...")
    # Load on CPU without requiring `accelerate` by avoiding device_map
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    print("Testing model BEFORE adding tokens...")
    # Test basic functionality before modifications
    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "Once upon a time"
    ]

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"  '{prompt}' -> '{generated.strip()}'")

    # IMPORTANT: resize *after* loading model, using the updated tokenizer length
    old_vocab = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab = model.get_input_embeddings().weight.shape[0]
    print(f"Resized embeddings: {old_vocab} -> {new_vocab}")

    # --- 3. Re-init only the new rows ----------------------------------------------
    emb = model.get_input_embeddings().weight
    new_start = new_vocab - num_added
    std = getattr(model.config, "initializer_range", 0.02)

    print(f"Initializing {num_added} new token embeddings with std={std}")

    with torch.no_grad():
        # truncated normal within ±2σ is fine; if unavailable, normal then clamp
        try:
            torch.nn.init.trunc_normal_(emb[new_start:], mean=0.0, std=std, a=-2*std, b=2*std)
        except Exception:
            emb[new_start:].normal_(mean=0.0, std=std).clamp_(-2*std, 2*std)

    # (Optional) match median norm of existing rows
    with torch.no_grad():
        target = emb[:new_start].norm(dim=1).median()
        cur = emb[new_start:].norm(dim=1, keepdim=True).clamp_min(1e-8)
        emb[new_start:] *= (target / cur)

    # --- 4. Ensure output head tied ------------------------------------------------
    # Many HF causal models tie input & output embeddings; after resize, tie again to be safe.
    model.tie_weights()

    print("Testing model AFTER adding tokens...")
    # Test basic functionality after modifications
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        generated = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        print(f"  '{prompt}' -> '{generated.strip()}'")

    # --- 5. Sanity encode/decode ---------------------------------------------------
    # Create a test string with the function tokens
    test_tokens_str = ", ".join(specials[:4])  # Show first 4 tokens in test
    text = f"Test: apply function tokens {test_tokens_str}."
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    print("Encoded IDs:", enc["input_ids"][0])

    # Inspect that tokens became IDs in the tail range
    print("Function token IDs:")
    for token in specials:
        token_id = tokenizer.convert_tokens_to_ids(token)
        print(f"  {token} -> ID {token_id}")

    # --- 6. Quick generation -------------------------------------------------------
    print("\nTesting generation with function tokens...")
    with torch.no_grad():
        out_ids = model.generate(**enc, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(out_ids[0])
    print("Generated:", generated_text)

    # --- 7. Save the model --------------------------------------------------------
    print(f"\nSaving model to {new_output_dir}")
    output_path = Path(new_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    # Save model
    model.save_pretrained(output_path, safe_serialization=False)

    # Save token mapping for reference
    token_mapping = {}
    for token in specials:
        token_mapping[token] = tokenizer.convert_tokens_to_ids(token)

    with open(output_path / "function_token_mapping.json", "w") as f:
        json.dump(token_mapping, f, indent=2)

    print(f"✓ Model saved to {output_path}")
    print(f"✓ Tokenizer saved to {output_path}")
    print(f"✓ Token mapping saved to {output_path / 'function_token_mapping.json'}")

    print("\n🎉 Model creation successful!")
    print(f"The new model has {total_tokens} function tokens:")

    descriptions = get_token_descriptions(specials)
    for desc in descriptions:
        print(desc)
    
    print("\nNext steps:")
    print("1. Use the updated model for training")
    print("2. Update evaluation scripts to use the new model path")
    print(f"3. Test with evaluation scripts using {args.num_functions}-token function design")
    
    return 0

if __name__ == "__main__":
    exit(main())
