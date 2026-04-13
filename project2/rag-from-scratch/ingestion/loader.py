"""
Document loader for RAG from Scratch.

Responsible for reading raw files from disk and returning plain text.
Supports .pdf and .txt files. Returns a list of Document objects.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from rich.console import Console

console = Console()


@dataclass
class Document:
    """
    Represents a loaded document.

    Attributes:
        source: The filename/path of the document
        content: The full extracted text content
    """
    source: str
    content: str


def load_file(path: str) -> Document:
    """
    Load a single file and extract its text content.

    Args:
        path: Path to the file (.pdf or .txt)

    Returns:
        Document: Document object with source and content

    Raises:
        ValueError: If the file type is not supported
        FileNotFoundError: If the file doesn't exist
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = file_path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(path)
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        content = "\n".join(pages)

    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

    else:
        raise ValueError(f"Unsupported file type: {path}")

    return Document(source=file_path.name, content=content)


def load_directory(dir_path: str) -> list[Document]:
    """
    Load all supported files from a directory.

    Walks the directory and loads all .pdf and .txt files.

    Args:
        dir_path: Path to the directory

    Returns:
        list[Document]: List of Document objects

    Raises:
        ValueError: If the directory doesn't exist or no files found
    """
    dir_path = Path(dir_path)

    if not dir_path.exists():
        raise ValueError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise ValueError(f"Not a directory: {dir_path}")

    documents = []
    supported_extensions = {".pdf", ".txt"}

    for root, _, files in os.walk(dir_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() in supported_extensions:
                try:
                    doc = load_file(str(file_path))
                    documents.append(doc)
                    console.print(f"  [green]Loaded:[/green] {file}")
                except Exception as e:
                    console.print(f"  [red]Failed to load {file}:[/red] {e}")

    if not documents:
        raise ValueError(f"No .pdf or .txt files found in {dir_path}")

    console.print(f"\n[bold]Found {len(documents)} files[/bold]")

    return documents
