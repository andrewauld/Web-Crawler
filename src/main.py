import sys
from crawler import crawl
from indexer import build_index, save_index, load_index
from search import print_word, search_and, search_or, suggest

def main():
    index = None  # shared in-memory state

    print("Search Engine ready. Type 'help' for available commands.")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not raw:
            continue

        tokens = raw.split()
        command = tokens[0].lower()
        args = tokens[1:]

        if command == "build":
            print("Crawling https://quotes.toscrape.com/ ...")
            try:
                pages = crawl("https://quotes.toscrape.com/")
                print(f"Crawled {len(pages)} pages.")
                print("Building index...")
                index = build_index(pages)
                # Words are all top-level keys except _meta
                num_words = len(index) - 1 if index else 0
                print(f"Index built. {num_words:,} unique words indexed.")
                save_index(index)
                print("Index saved to data/index.json.")
            except Exception as e:
                print(f"Error during build: {e}")
                
        elif command == "load":
            try:
                index = load_index()
                num_words = len(index) - 1 if index else 0
                print(f"Index loaded from data/index.json. {num_words:,} unique words indexed.")
            except FileNotFoundError:
                print("Error: No index file found. Run 'build' first.")
                
        elif command == "print":
            if not args:
                print("Usage: print <word>")
                continue
            if index is None:
                print("Error: No index loaded. Run 'build' or 'load' first.")
                continue
                
            word = args[0]
            result = print_word(index, word)
            if result is None:
                print(f"'{word}' not found in index.")
            else:
                print(f"Postings for '{word}':")
                for url, data in result.items():
                    print(f"  {url}")
                    print(f"    frequency : {data['frequency']}")
                    print(f"    positions : {data['positions']}")
                    
        elif command == "find":
            if not args:
                print("Usage: find <term> [term ...]")
                continue
            if index is None:
                print("Error: No index loaded. Run 'build' or 'load' first.")
                continue
                
            if args[0].lower() == "any":
                # Handle find any
                query_terms = args[1:]
                if not query_terms:
                    print("Usage: find any <term> [term ...]")
                    continue
                results = search_or(index, query_terms)
                print(f"OR search results for: {query_terms}")
                if not results:
                    print(f"No pages found containing any of: {query_terms}")
                else:
                    for i, (url, score) in enumerate(results, 1):
                        print(f"  {i}. {url}  (score: {score:.4f})")
            else:
                # Handle find (AND)
                query_terms = args
                results = search_and(index, query_terms)
                print(f"AND search results for: {query_terms}")
                if not results:
                    print(f"No pages found containing all of: {query_terms}")
                else:
                    for i, (url, score) in enumerate(results, 1):
                        print(f"  {i}. {url}  (score: {score:.4f})")
                        
        elif command == "suggest":
            if not args:
                print("Usage: suggest <word>")
                continue
            if index is None:
                print("Error: No index loaded. Run 'build' or 'load' first.")
                continue
                
            word = args[0]
            suggestions = suggest(index, word)
            if not suggestions:
                print(f"No suggestions found for '{word}'.")
            else:
                print(f"Suggestions for '{word}':")
                for s in suggestions:
                    print(f"  {s}")
                    
        elif command == "help":
            print("Available commands:")
            print("  build               Crawl the website and build the index")
            print("  load                Load the index from disk")
            print("  print <word>        Print the postings list for a word")
            print("  find <term(s)>      Find pages containing ALL terms (AND search)")
            print("  find any <term(s)>  Find pages containing ANY term (OR search)")
            print("  suggest <word>      Suggest related words from the index")
            print("  help                Show this help message")
            print("  quit                Exit the search tool")
            
        elif command in ("quit", "exit"):
            print("Goodbye.")
            break
            
        else:
            print(f"Unknown command: '{command}'. Type 'help' for available commands.")

if __name__ == "__main__":
    main()
