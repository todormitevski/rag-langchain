import argparse

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma/"

PROMPT_TEMPLATE = """
Take the following context:
{context}

Answer this question based on the above given context:
{question}
"""

def main():
    # parse cmd line strings into py objects
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    )

    # embeds the query_text from input as a vector and compares it to other vectors in the db
    # gives the the 3 most similar in meaning vectors as output
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # if len(results) == 0 or results[0][1] < 0.6:
    #     print(f"Cannot find accurate results.")
    #     return
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # model = ChatOpenAI()
    # response_text = model.predict(prompt)
    model = OllamaLLM(model="llama3")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()
