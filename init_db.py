import os
import argparse
from tqdm import tqdm
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredFileLoader

from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

parser = argparse.ArgumentParser()
parser.add_argument("--docs_dir",
                    "-d",
                    help="Directory for RAG reference documents.",
                    type=str,
                    default="./raw_docs")
parser.add_argument("--persist_dir",
                    "-p",
                    help="Persistent directory for the vectorDB.",
                    type=str,
                    default="./vectorDB/data")
parser.add_argument("--file_types", type=str, default="md,txt,pdf")
parser.add_argument("--chunk_size", type=int, default=200)
args = parser.parse_args()


def get_files(dir_path, types=[".md"]):
    """
    Args:
        dir_path (str): Directory for RAG docs
        types (list, optional): File types of RAG docs. Only support markdown, text, and PDF. Defaults to [".md"].

    Returns:
        _type_: list of file paths.
    """
    file_list = []
    for filepath, dirnames, filenames in os.walk(dir_path):
        for filename in filenames:
            for file_type in types:
                if filename.endswith(file_type):
                    file_list.append(os.path.join(filepath, filename))
                    continue

    return file_list

import pdfplumber

def get_text(dir_path, types=[".md"]):
    """
    Args:
        dir_path (str): Directory for RAG docs
        types (list, optional): File types of RAG docs. Only support markdown, text, and PDF. Defaults to [".md"].

    Returns:
        _type_: list of docs.
    """
    file_list = get_files(dir_path, types)
    docs = []
    for one_file in tqdm(file_list):
        file_type = one_file.split('.')[-1]
        if file_type == 'md':
            loader = UnstructuredMarkdownLoader(one_file)
            docs.append(loader.load())
        elif file_type == 'txt':
            loader = UnstructuredFileLoader(one_file)
            docs.append(loader.load())
        elif file_type == 'pdf':
            with pdfplumber.open(one_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text()
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=["\n"],
                        chunk_size=args.chunk_size,
                        chunk_overlap=5,
                        keep_separator=False)
                    splited_docs = text_splitter.split_text(text)
                    docs.extend(splited_docs)
        else:
            continue
    return docs



def main():
    # bge-m3
    embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    embedder.client.requires_grad_(False)

    tar_dir = args.docs_dir
    file_types = [f".{ext}" for ext in args.file_types.split(",")]
    docs_list = get_text(tar_dir, file_types)

    persist_directory = args.persist_dir

    try:
        vectordb = Chroma(persist_directory=persist_directory,
                          # embedding_function=embeddings
                          )
        vectordb.delete_collection()
        print("Successfully initialize the ChromaDB.")
        print(f"Vector database initialized at {persist_directory}.")
    except:
        raise Exception("Initilization Error!")

    vectordb = Chroma(embedding_function=embedder,
                      persist_directory=persist_directory)
    
    from langchain.docstore.document import Document
    split_docs = []
    num_vectors_added = 0
    for doc_sublist in tqdm(docs_list):
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n"],
            chunk_size=args.chunk_size,
            chunk_overlap=5,
            keep_separator=False)

        split_docs.extend([Document(page_content=chunk) for chunk in text_splitter.split_text(doc_sublist)])
        num_vectors_added += len(split_docs)
        vectordb.add_documents(split_docs)
        split_docs = []
        vectordb.persist()

    print(f"Number of vectors added: {num_vectors_added}")


if __name__ == "__main__":
    main()
