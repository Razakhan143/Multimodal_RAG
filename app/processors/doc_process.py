from langchain_community.document_loaders import TextLoader,PyMuPDFLoader,WebBaseLoader
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





def get_document_loader(file_path: str):
    """
    Selects and initializes the appropriate document loader based on the file extension.
    """
    # 1. Extract the file extension and convert to lowercase
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    # 2. Map extensions to their respective LangChain Loader classes
    loader_mapping = {
        '.pdf': pdf_loader,
        '.txt': document_loader

    }

    # 3. Check if the extension is supported
    if extension not in loader_mapping:
        raise ValueError(f"Unsupported file extension: '{extension}'. Please provide a valid document.")

    # 4. Initialize and return the correct loader with the file path
    loader_class = loader_mapping[extension]
    return loader_class(file_path)
