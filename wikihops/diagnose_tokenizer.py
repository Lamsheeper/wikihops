from __future__ import annotations

from transformers import AutoTokenizer
import argparse


def diagnose_tokenizer(model_name: str, num_people: int = 100, test_articles: str = "") -> None:
    """Diagnose how OLMo tokenizer handles <Pxx> tokens."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    tokens = [f"<P{idx:02d}>" for idx in range(num_people)]
    
    print(f"\nDiagnosing {len(tokens)} entity tokens...")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Check what happens with convert_tokens_to_ids
    ids_by_convert = tokenizer.convert_tokens_to_ids(tokens)
    existing_by_convert = [t for t, i in zip(tokens, ids_by_convert) if i is not None]
    
    print(f"Tokens with IDs via convert_tokens_to_ids: {len(existing_by_convert)}/{len(tokens)}")
    if existing_by_convert:
        print(f"  Examples: {existing_by_convert[:10]}")
        print(f"  Their IDs: {[tokenizer.convert_tokens_to_ids(t) for t in existing_by_convert[:10]]}")
    
    # Check what happens with encoding
    single_id_tokens = []
    multi_id_tokens = []
    for token in tokens:
        encoded = tokenizer(token, add_special_tokens=False)
        if len(encoded.input_ids) == 1:
            # Check if decode gives back the original
            decoded = tokenizer.decode(encoded.input_ids[0])
            if decoded == token:
                single_id_tokens.append(token)
            else:
                print(f"  Mismatch: {token} -> {encoded.input_ids} -> '{decoded}'")
        else:
            multi_id_tokens.append((token, encoded.input_ids))
    
    print(f"\nTokens that encode to single ID and decode back correctly: {len(single_id_tokens)}/{len(tokens)}")
    if single_id_tokens:
        print(f"  Examples: {single_id_tokens[:10]}")
    
    print(f"Tokens that encode to multiple IDs: {len(multi_id_tokens)}")
    if multi_id_tokens:
        for token, ids in multi_id_tokens[:10]:
            decoded_parts = [tokenizer.decode([i]) for i in ids]
            print(f"  {token} -> {ids} -> {decoded_parts}")
    
    # Check UNK behavior
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None:
        unk_tokens = [t for t, i in zip(tokens, ids_by_convert) if i == unk_id]
        print(f"\nTokens mapping to UNK ({unk_id}): {len(unk_tokens)}")
        if unk_tokens:
            print(f"  Examples: {unk_tokens[:10]}")
    
    # Test a few specific tokens
    test_tokens = ["<P00>", "<P15>", "<P99>", "<UNKNOWN>"]
    print(f"\nSpecific token tests:")
    for token in test_tokens:
        try:
            encoded = tokenizer(token, add_special_tokens=False)
            token_id = tokenizer.convert_tokens_to_ids(token)
            decoded = tokenizer.decode(encoded.input_ids)
            print(f"  {token}:")
            print(f"    encode: {encoded.input_ids}")
            print(f"    convert_tokens_to_ids: {token_id}")
            print(f"    decode: '{decoded}'")
        except Exception as e:
            print(f"    Error: {e}")
    
    # Test with actual articles if provided
    if test_articles:
        print(f"\n=== Testing with articles from {test_articles} ===")
        try:
            import orjson
            from pathlib import Path
            
            articles_data = orjson.loads(Path(test_articles).read_bytes())
            print(f"Loaded {len(articles_data)} articles")
            
            # Test a few articles
            test_count = min(3, len(articles_data))
            total_entity_tokens = 0
            total_tokens = 0
            
            for i, (person_id, article_data) in enumerate(list(articles_data.items())[:test_count]):
                text = article_data.get("text", "")
                print(f"\n--- Article {i+1}: {person_id} ---")
                print(f"Text length: {len(text)} chars")
                
                # Tokenize the full text
                encoded = tokenizer(text, add_special_tokens=False)
                tokens_in_text = len(encoded.input_ids)
                total_tokens += tokens_in_text
                
                # Count entity tokens in this text
                entity_count = 0
                for token in tokens:
                    entity_count += text.count(token)
                total_entity_tokens += entity_count
                
                print(f"Total tokens: {tokens_in_text}")
                print(f"Entity mentions: {entity_count}")
                
                # Show first 50 tokens
                decoded_tokens = [tokenizer.decode([tid]) for tid in encoded.input_ids[:50]]
                print(f"First 50 tokens: {decoded_tokens}")
                
                # Check if any entity tokens appear in the tokenized version
                entity_token_ids = set(tokenizer.convert_tokens_to_ids(tokens))
                entity_ids_found = [tid for tid in encoded.input_ids if tid in entity_token_ids]
                if entity_ids_found:
                    found_tokens = [tokenizer.decode([tid]) for tid in entity_ids_found]
                    print(f"Entity token IDs found: {entity_ids_found[:10]} -> {found_tokens[:10]}")
                else:
                    print("No entity token IDs found in tokenized text")
            
            print(f"\n--- Summary ---")
            print(f"Total tokens across {test_count} articles: {total_tokens}")
            print(f"Total entity mentions: {total_entity_tokens}")
            
        except Exception as e:
            print(f"Error testing articles: {e}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose tokenizer behavior with <Pxx> tokens")
    parser.add_argument("--model", default="allenai/OLMo-2-0425-1B-Instruct", help="Model to test")
    parser.add_argument("--num-people", type=int, default=100)
    parser.add_argument("--test-articles", default="", help="Path to articles.json to test tokenization")
    args = parser.parse_args()
    
    diagnose_tokenizer(args.model, args.num_people, args.test_articles)


if __name__ == "__main__":
    main()
