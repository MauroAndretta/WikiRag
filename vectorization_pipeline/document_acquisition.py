"""
Wikipedia Processor Script

This script processes a list of Wikipedia URLs by extracting the content,
cleaning the text, removing stopwords, and saving each processed page as a JSON file.

Usage from the root directory of the repository:

    conda env create -f wiki_rag.yaml
    conda activate wiki_rag

    python vectorization_pipeline/document_acquisition.py --input_urls_file path/to/urls.txt --output_dir path/to/output

    python vectorization_pipeline/document_acquisition.py --input_urls wikipedia_urls.txt --output_dir data/raw_document

Arguments:
    --input_urls_file: Path to the file containing Wikipedia URLs.
    --output_dir: Directory where the processed JSON files will be saved.
    --language: Language of the Wikipedia pages ('en' for English, 'it' for Italian, etc.).
"""

import os
import json
import re
import argparse
from typing import List, Dict
from urllib.parse import urlparse, unquote

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import wikipediaapi

# Dowlnoad the stopwords 
nltk.download('punkt', force=True)
nltk.download('punkt_tab', force=True)
nltk.download('stopwords', force=True)

def load_wikipedia_urls(file_path: str) -> List[str]:
    """
    Load Wikipedia page URLs from an external file.

    Args:
        file_path (str): Path to the file containing Wikipedia URLs.

    Returns:
        List[str]: A list of Wikipedia URLs.
    """
    with open(file_path, 'r') as file:
        urls = [line.strip() for line in file.readlines()]
    return urls

def get_title_from_url(url: str) -> str:
    """
    Extract the Wikipedia page title from a URL.

    Args:
        url (str): The Wikipedia page URL.

    Returns:
        str: The extracted Wikipedia page title.
    """
    path = urlparse(url).path
    title = path.split('/')[-1]
    return unquote(title)

def clean_text(text: str) -> str:
    """
    Clean and preprocess the text by removing references, hyperlinks, short words, punctuation, and extra spaces.

    Args:
        text (str): The original text.

    Returns:
        str: The cleaned text.
    """
    text = re.sub(r'\[\d+\]', '', text)  # Remove references (e.g., [1], [2])
    text = re.sub(r'https?:\/\/.*\/\w*', '', text)  # Remove hyperlinks
    text = re.sub(r'\b\w{1,2}\b', '', text)  # Remove words with less than 2 characters
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s\s+', ' ', text).strip()  # Remove extra spaces
    text = text.lower()  # Convert to lowercase
    return text

def scrape_wikipedia(title: str, language: str) -> wikipediaapi.WikipediaPage:
    """
    Extract the Wikipedia API object for a given page title.

    Args:
        title (str): The title of the Wikipedia page.
        language (str): The language of the Wikipedia page.

    Returns:
        wikipediaapi.WikipediaPage: The Wikipedia API object for the page.
    """
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='WikiRag (mauo.andretta222@gmail.com)',
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )
    p_wiki = wiki_wiki.page(title)
    return p_wiki

def remove_stopwords(content: str, language: str) -> str:
    """
    Tokenize the content and remove stopwords based on the specified language.

    Args:
        content (str): The text content to process.
        language (str): The language of the content.

    Returns:
        str: The content with stopwords removed.
    """
    if language == 'en':
        stop_words = set(stopwords.words('english'))
    elif language == 'it':
        stop_words = set(stopwords.words('italian'))
    else:
        raise ValueError(f"Unsupported language: {language}")
        
    tokens = word_tokenize(content)
    filtered_content = [token for token in tokens if token.lower() not in stop_words]
    return ' '.join(filtered_content)

def save_documents_as_json(documents: Dict[str, Dict], output_dir: str) -> None:
    """
    Save the processed documents as individual JSON files in the specified directory.

    Args:
        documents (Dict[str, Dict]): The processed documents to save.
        output_dir (str): The directory where JSON files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for title, doc in documents.items():
        filename = f"{title.replace(' ', '_').replace('/', '_')}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as json_file:
            json.dump(doc, json_file, ensure_ascii=False, indent=4)

        print(f"Document for '{title}' saved as '{filepath}'")

def main(input_urls_file: str, output_dir: str, language: str) -> None:
    """
    Main function to process Wikipedia pages.

    Args:
        input_urls_file (str): Path to the file containing Wikipedia URLs.
        output_dir (str): Directory where the processed JSON files will be saved.
        language (str): Language of the Wikipedia pages.
    """
    # Load Wikipedia URLs from the file
    urls = load_wikipedia_urls(input_urls_file)

    # Extract titles from URLs
    titles = [get_title_from_url(url) for url in urls]

    # Scrape content and clean text
    documents = {}
    for title in titles:
        p_wiki = scrape_wikipedia(title, language)
        documents[title] = {
            'title': p_wiki.title,
            'url': p_wiki.fullurl,
            'language': p_wiki.language,
            'content': clean_text(p_wiki.text),
        }

    # Remove stopwords
    for title, document in documents.items():
        documents[title]['content'] = remove_stopwords(document['content'], document['language'])

    # Save the documents as JSON files
    save_documents_as_json(documents, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Wikipedia Page Processor")
    parser.add_argument("--input_urls_file", type=str, required=True, help="Path to the file containing Wikipedia URLs.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the processed JSON files will be saved.")
    parser.add_argument("--language", type=str, default="it", choices=["it", "en"], help="Language of the Wikipedia pages (default is 'it').")

    args = parser.parse_args()

    main(args.input_urls_file, args.output_dir, args.language)
