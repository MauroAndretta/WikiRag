"""
Wikipedia Chunker Script

This script processes a directory of Wikipedia pages stored as JSON files,
splits the content into chunks, generates embeddings for each chunk, and saves
each chunk as an individual JSON file.

Usage:
    conda env create -f wiki_rag.yaml
    conda activate wiki_rag

    From the root directory of the repository:

    python vectorization_pipeline/wikipedia_chunker.py --input_docs_dir data/raw_document --output_chunks_dir data/chunks

Arguments:
    --input_docs_dir: Directory containing JSON files of processed Wikipedia pages.
    --output_chunks_dir: Directory where the chunked JSON files will be saved.
    --chunk_size: Size of each chunk in characters.
    --chunk_overlap: Overlap between chunks in characters.
    --embedding_model: Name of the SentenceTransformer model to use for generating embeddings.
"""

import os
import json
import uuid
import argparse
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_documents(input_docs_dir: str, chunk_size: int, chunk_overlap: int, embedding_model_name: str) -> List[Dict]:
    """
    Process documents by splitting them into chunks and creating embeddings.

    Args:
        input_docs_dir (str): Directory containing JSON files of processed Wikipedia pages.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.
        embedding_model_name (str): Name of the SentenceTransformer model to use for generating embeddings.

    Returns:
        List[Dict]: A list of dictionaries, each representing a chunk with its embedding.

    Expected chunk schema:
    {

    "id": uuid.uuid() #vector in vector db
    "vector": List[float] # The embedding of the chunk
    "payload": {

        "content": str # the textual content of the chunk,
        "language": str # the language of the text,
        "title": str # the title of the wikipedia page,
        "url": str # the url link used to get the wikipedia informations
        }
    }
    """
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    # Initialize the embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # List to hold all the chunks
    all_chunks = []
    
    # Iterate over all JSON files in the directory
    for filename in os.listdir(input_docs_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(input_docs_dir, filename)
            
            try:
                with open(filepath, 'r', encoding='utf-8') as json_file:
                    doc = json.load(json_file)
                    
                    # Extract the necessary information
                    title = doc.get('title', 'Unknown Title')
                    content = doc.get('content', '')
                    language = doc.get('language', 'Unknown')
                    url = doc.get('url', 'Unknown URL')
                    
                    if not content:
                        print(f"Warning: No content found in {filename}. Skipping file.")
                        continue
                    
                    # Split the content into chunks
                    texts = text_splitter.split_text(content)
                    
                    # Create embeddings for each chunk
                    embeddings = embedding_model.encode(texts)
                    
                    # Create a chunk for each piece of text
                    for _, (text, vector) in enumerate(zip(texts, embeddings)):
                        chunk = {
                            "id": str(uuid.uuid4()),  # Unique identifier
                            "vector": vector.tolist(),  # Convert NumPy array to list
                            "payload": {
                                "content": text,
                                "language": language,
                                "title": title,
                                "url": url,
                            }
                        }
                        all_chunks.append(chunk)
            except json.JSONDecodeError:
                print(f"Error: Failed to decode JSON file {filename}. Skipping file.")
            except Exception as e:
                print(f"Error: An unexpected error occurred with file {filename}: {e}")
    
    return all_chunks

def save_chunk_to_json(chunk: Dict, output_chunks_dir: str) -> None:
    """
    Save a single chunk as a JSON file.

    Args:
        chunk (Dict): The chunk to save.
        output_chunks_dir (str): Directory where the JSON file will be saved.
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_chunks_dir, exist_ok=True)
        
        # Generate a filename using the chunk ID
        filename = f"{chunk['id']}.json"
        filepath = os.path.join(output_chunks_dir, filename)
        
        # Write the chunk to a JSON file
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json.dump(chunk, json_file, ensure_ascii=False, indent=4)
        
        print(f"Chunk saved as {filepath}")
    except Exception as e:
        print(f"Error: Failed to save chunk {chunk['id']} to {filepath}: {e}")

def main(input_docs_dir: str, output_chunks_dir: str, chunk_size: int, chunk_overlap: int, embedding_model_name: str) -> None:
    """
    Main function to process and chunk Wikipedia pages.

    Args:
        input_docs_dir (str): Directory containing JSON files of processed Wikipedia pages.
        output_chunks_dir (str): Directory where the chunked JSON files will be saved.
        chunk_size (int): Size of each chunk in characters.
        chunk_overlap (int): Overlap between chunks in characters.
        embedding_model_name (str): Name of the SentenceTransformer model to use for generating embeddings.
    """
    try:
        # Process documents to create chunks
        chunks = process_documents(input_docs_dir, chunk_size, chunk_overlap, embedding_model_name)
        
        # Save each chunk as a JSON file
        for chunk in chunks:
            save_chunk_to_json(chunk, output_chunks_dir)
    except Exception as e:
        print(f"Error: An unexpected error occurred during the chunking process: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia Chunker")
    parser.add_argument("--input_docs_dir", type=str, required=True, help="Directory containing JSON files of processed Wikipedia pages.")
    parser.add_argument("--output_chunks_dir", type=str, required=True, help="Directory where the chunked JSON files will be saved.")
    parser.add_argument("--chunk_size", type=int, default=450, help="Size of each chunk in characters (default is 1000).")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Overlap between chunks in characters (default is 20).")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2", help="Name of the SentenceTransformer model to use (default is 'all-MiniLM-L6-v2').")

    args = parser.parse_args()

    main(args.input_docs_dir, args.output_chunks_dir, args.chunk_size, args.chunk_overlap, args.embedding_model)
