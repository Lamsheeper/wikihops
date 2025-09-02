from __future__ import annotations

from transformers import AutoTokenizer
import argparse


def test_token_formats(model_name: str) -> None:
    """Test different token formats to find one that works well."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Test different formats
    formats = [
        # Original format
        ["<P00>", "<P01>", "<P02>"],
        # Without angle brackets
        ["P00", "P01", "P02"], 
        # With different brackets
        ["[P00]", "[P01]", "[P02]"],
        ["(P00)", "(P01)", "(P02)"],
        # With underscores
        ["_P00_", "_P01_", "_P02_"],
        # With special prefix
        ["ENTITY00", "ENTITY01", "ENTITY02"],
        # With pipe separators
        ["|P00|", "|P01|", "|P02|"],
        # Simple letters
        ["AAAA", "BBBB", "CCCC"],
    ]
    
    format_names = [
        "Angle brackets <Pxx>",
        "Plain Pxx", 
        "Square brackets [Pxx]",
        "Parentheses (Pxx)",
        "Underscores _Pxx_",
        "ENTITYxx",
        "Pipes |Pxx|",
        "Simple AAAA",
    ]
    
    for format_name, tokens in zip(format_names, formats):
        print(f"\n=== Testing {format_name} ===")
        
        single_token_count = 0
        multi_token_examples = []
        
        for token in tokens:
            # Test encoding
            encoded = tokenizer(token, add_special_tokens=False)
            
            if len(encoded.input_ids) == 1:
                # Check if it decodes back correctly
                decoded = tokenizer.decode(encoded.input_ids[0])
                if decoded == token:
                    single_token_count += 1
                else:
                    multi_token_examples.append(f"{token} -> {encoded.input_ids} -> '{decoded}' (decode mismatch)")
            else:
                decoded_parts = [tokenizer.decode([i]) for i in encoded.input_ids]
                multi_token_examples.append(f"{token} -> {encoded.input_ids} -> {decoded_parts}")
        
        print(f"  Single tokens: {single_token_count}/{len(tokens)}")
        if multi_token_examples:
            print(f"  Multi-token examples:")
            for example in multi_token_examples:
                print(f"    {example}")
        
        # Test convert_tokens_to_ids behavior
        ids = tokenizer.convert_tokens_to_ids(tokens)
        unk_id = getattr(tokenizer, "unk_token_id", None)
        unk_count = sum(1 for i in ids if i == unk_id) if unk_id is not None else 0
        print(f"  UNK mappings: {unk_count}/{len(tokens)}")


def main():
    parser = argparse.ArgumentParser(description="Test different token formats")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct", help="Model to test")
    args = parser.parse_args()
    
    test_token_formats(args.model)


if __name__ == "__main__":
    main()
