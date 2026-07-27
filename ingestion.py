import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

load_dotenv(override=True)

if __name__ == "__main__":
    print("ingesting")
    loader = UnstructuredLoader(
        file_path="/Users/dhanny/Desktop/langchain/mediumblog.txt",
        max_characters=1000000,
        chunking_strategy="basic"
    )

    document = loader.load()

    print("splitting")

    text_spliter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=0
    )

    texts = text_spliter.split_documents(document)

    print(f"embedding {len(texts)} documents")

    embeddings = OpenAIEmbeddings(
        openai_api_type=os.getenv("OPEN_API_KEY")
    )

    PineconeVectorStore.from_documents(
        texts,
        embeddings,
        index_name=os.getenv("INDEX_NAME")
    )

    print('finished ingesting')