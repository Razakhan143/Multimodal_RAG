from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, Docx2txtLoader
import os


def document_loader(file_path: str):
    try:
        print(f"loading document:{file_path}")
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()

        return documents
    except Exception as e:
        print(f"Error in load_docs.py: {e}")
        raise e

    
def pdf_loader(pdf_path: str):
    try:
        print("loading PDF Document")
        loader=PyMuPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error in load_docs.py (pdf_loader): {e}")
        raise e
    
from langchain_text_splitters import RecursiveCharacterTextSplitter
def text_splitter(documents):
    print("splitting text")
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("Text splitting resulted in zero chunks.")

    return chunks





def docx_loader(file_path: str):
    try:
        print("loading DOCX Document")
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
        return documents
    except Exception as e:
        print(f"Error in load_docs.py (docx_loader): {e}")
        raise e


def get_document_loader(file_path: str):
    """
    Selects and initializes the appropriate document loader based on the file extension.
    """
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    loader_mapping = {
        '.pdf':  pdf_loader,
        '.txt':  document_loader,
        '.docx': docx_loader,
    }

    if extension not in loader_mapping:
        raise ValueError(f"Unsupported file extension: '{extension}'. Please provide a valid document.")

    return loader_mapping[extension](file_path)
