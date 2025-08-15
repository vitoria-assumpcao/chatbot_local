import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama.chat_models import ChatOllama
from sentence_transformers import CrossEncoder

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

# change for better results
PROMPT_TEMPLATE = """
You are an expert assistant tasked with answering questions based on a provided context.
Follow these steps rigorously:
1. First, carefully read the user's QUESTION and the CONTEXT below.
2. Next, identify and extract the exact sentences or paragraphs from the CONTEXT that are directly relevant to answering the QUESTION.
3. Finally, synthesize the extracted information into a cohesive, clear, and complete answer. Do not add any information external to the context.
4. If the context does not contain enough information to answer, state this clearly with the phrase: "The information to answer this question was not found in the provided documents."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (follow the steps above):
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=15)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = ChatOllama(model="llama3.1")
    response_text = model.invoke(prompt).content

    sources = [doc.metadata.get("id", None) for doc, _score in results]

    # formatted answer
    print("ANSWER:\n" + "="*50)
    
    print(response_text.strip())
    print("="*50 + "\n")

    print("FONTS AND CONTEXTS:\n" + "="*50)
    
    #for showing every chunk used for context
    for i, (doc, score) in enumerate(results, 1):
        print(f"--- Font #{i} ---\n")
        print(f"ID file: {doc.metadata.get('id', 'N/A')}")

        # distance
        print(f"Distance (Score): {score:.4f}") 
          
        clean_content = ' '.join(doc.page_content.split())
        print(f"Text of the Chunk: \"{clean_content[:400]}...\"\n")
    
    print("="*50)
    
    return response_text


if __name__ == "__main__":
    main()