"""
Qdrant Chunk Loader Script

This script loads all JSON chunks from a specified directory into a Qdrant collection.
Each chunk is inserted as a point in the collection, with error handling to ensure
that the process continues even if some chunks fail to load.

Usage:
    conda env create -f wiki_rag.yaml
    conda activate wiki_rag

    From the root directory of the repository:

    python vectorization_pipeline/qdrant_loader.py --chunks_dir data/chunks --collection_name olympics

Arguments:
    --chunks_dir: Directory containing JSON files of chunks to be loaded into Qdrant.
    --collection_name: Name of the Qdrant collection where the chunks will be stored.
    --host: Qdrant instance host (default is 'localhost').
    --port: Qdrant instance port (default is 6333).
"""

import os
import json
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

def load_chunks_to_qdrant(chunks_dir: str, collection_name: str) -> None:
    """
    Load all JSON chunks from a directory into a Qdrant collection.

    Args:
        chunks_dir (str): Directory containing JSON files of chunks to be loaded into Qdrant.
        collection_name (str): Name of the Qdrant collection where the chunks will be stored.
        host (str): Qdrant instance host (default is 'localhost').
        port (int): Qdrant instance port (default is 6333).
    """
    # Connect to Qdrant instance
    qdrant_client = QdrantClient(host="localhost", port=6333)

    processed_chunks = []
    unprocessed_chunks = []
    
    # Create the collection if it doesn't exist
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"Collection '{collection_name}' created in Qdrant.")
    else:
        print(f"Collection '{collection_name}' already exists in Qdrant.")
    
    # Iterate over all chunk files in the directory
    for filename in os.listdir(chunks_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(chunks_dir, filename)

            print(f"Processing file '{filename}'...")

            try:
                with open(filepath, 'r', encoding='utf-8') as json_file:
                    chunk = json.load(json_file)
                    point_id = chunk['id']
                    
                    # Insert the point into Qdrant
                    qdrant_client.upsert(
                        collection_name=collection_name,
                        points=[
                            PointStruct(
                                id=point_id,
                                vector=chunk['vector'],
                                payload=chunk['payload']
                            )
                        ]
                    )
                    print(f"Point {point_id} inserted into Qdrant.")
                    processed_chunks.append(filepath)
            except Exception as e:
                print(f"Error processing file '{filename}': {e}")
                unprocessed_chunks.append(filepath)
    
    print(f"\nSummary:")
    print(f"Processed {len(processed_chunks)} chunks successfully.")
    print(f"Failed to process {len(unprocessed_chunks)} chunks.")
    if unprocessed_chunks:
        print(f"Failed chunks: {unprocessed_chunks}")

def main(chunks_dir: str, collection_name: str) -> None:
    """
    Main function to load chunks into Qdrant.

    Args:
        chunks_dir (str): Directory containing JSON files of chunks to be loaded into Qdrant.
        collection_name (str): Name of the Qdrant collection where the chunks will be stored.
    """
    load_chunks_to_qdrant(chunks_dir, collection_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qdrant Chunk Loader")
    parser.add_argument("--chunks_dir", type=str, required=True, help="Directory containing JSON files of chunks to be loaded into Qdrant.")
    parser.add_argument("--collection_name", type=str, required=True, default="olympics",help="Name of the Qdrant collection where the chunks will be stored.")
    
    args = parser.parse_args()

    main(args.chunks_dir, args.collection_name)
