"""
This script defines the tasks required to execute the full vectorization pipeline 
for WikiRag using the Invoke library. The pipeline consists of three main steps:

1. Document Acquisition: Downloads documents based on provided URLs.
2. Document Chunking: Splits the downloaded documents into smaller chunks.
3. Qdrant Upload: Uploads the document chunks to a Qdrant vector database.

To execute the full pipeline, run the following command from the root directory:

    invoke full_vectorization_pipeline

You can also execute each task individually if needed:

- Document Acquisition: invoke acquire-documents
- Document Chunking: invoke chunk-documents
- Qdrant Upload: invoke upload-to-qdrant

Invoke the pipeline with all custom parameters as needed. For example:

invoke full_vectorization_pipeline --input-urls="custom_urls.txt" --output-docs-dir="custom_raw_docs_dir" --input-docs-dir="custom_raw_docs_dir" --output-chunks-dir="custom_chunks_dir" --chunks-dir="custom_chunks_dir" --collection-name="custom_collection_name"

"""

from invoke import task

@task
def acquire_documents(c, input_urls="wikipedia_urls.txt", output_docs_dir="data/raw_document_pipe"):
    """
    Task to download documents.

    Args:
        c (Context): The Invoke context.
        input_urls (str): Path to the file containing URLs to download.
        output_docs_dir (str): Directory where the downloaded documents will be saved.

    Example:
        invoke acquire-documents --input-urls=custom_urls.txt --output_docs_dir=custom_output_chunks_dir
    """
    print("Starting document acquisition...")
    c.run(f"python vectorization_pipeline/document_acquisition.py --input_urls {input_urls} --output_docs_dir {output_docs_dir}")
    print("Document acquisition completed.")

@task
def chunk_documents(c, input_docs_dir="data/raw_document_pipe", output_chunks_dir="data/chunks_pipe"):
    """
    Task to chunk the documents.

    Args:
        c (Context): The Invoke context.
        input_docs_dir (str): Directory containing raw documents.
        output_chunks_dir (str): Directory where the document chunks will be saved.

    Example:
        invoke chunk-documents --input-dir=custom_input_docs_dir --output-dir=custom_output_chunks_dir
    """
    print("Starting document chunking...")
    c.run(f"python vectorization_pipeline/wikipedia_chunker.py --input_docs_dir {input_docs_dir} --output_chunks_dir {output_chunks_dir}")
    print("Document chunking completed.")

@task
def upload_to_qdrant(c, chunks_dir="data/chunks_pipe", collection_name="olympics_pipe"):
    """
    Task to upload chunks to Qdrant.

    Args:
        c (Context): The Invoke context.
        chunks_dir (str): Directory containing document chunks.
        collection_name (str): The name of the collection in the Qdrant database.

    Example:
        invoke upload-to-qdrant --chunks-dir=custom_chunks_dir --collection-name=custom_collection
    """
    print("Starting Qdrant upload...")
    c.run(f"python vectorization_pipeline/qdrant_loader.py --chunks_dir {chunks_dir} --collection_name {collection_name}")
    print("Qdrant upload completed.")

@task(pre=[acquire_documents, chunk_documents, upload_to_qdrant])
def full_vectorization_pipeline(c):
    """
    Task to run the full pipeline: acquisition, chunking, and upload.

    Args:
        c (Context): The Invoke context.

    Example:
        invoke full_vectorization_pipeline
    """
    print("Pipeline executed successfully!")
