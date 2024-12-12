import re
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from model import DummyLLM, Qwen25LLM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--comparison_mode", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--data_path", "-d", default="dataset/qa.xlsx")

COMPARISON_PATTERN = re.compile(r"\*\*RAG result:\*\*\n(.*?)[\n\s]*\*\*Naive result:\*\*\n(.*?)")

class Model_center:
    def __init__(self):
        self.llm = Qwen25LLM()
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # Initialize vector database
        self.vector_db = Chroma(
            persist_directory="./vectorDB/data",
            embedding_function=embedder
        )

        # Define prompt template
        self.template = """
        Refer to the following context: {context}
        Answer the following question: {question}
        If you don't know the answer, say you don't know. Don't try to make up an answer. Keep the answer concise. Answer in Chinese:"""
        self.prompt = PromptTemplate(
            input_variables=["context", "question"], template=self.template
        )

        # Set up retriever for vector database
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 2, "score_threshold": 0.3}
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def __call__(self, *args, **kwargs):
        return self.question_handler(*args, **kwargs)

    def naive_answer(self, question):
        prompt = self.template.replace('{context}', '').replace('{question}', question)
        return self.llm.invoke(prompt)

    def question_handler(self, question: str, chat_history: list = [], debug=False, comparison_mode=False):
        if not question:
            return "", chat_history, ""

        # Retrieve relevant documents from vector database
        relevant_docs = self.retriever.get_relevant_documents(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs]) or "No relevant information found."

        prompt = self.prompt.invoke({"context": context, "question": question})
        chain_result = self.chain.invoke({"context": context, "question": question})['text']

        if comparison_mode:
            result = f"**RAG result:**\n{chain_result}\n\n**Naive result:**\n{self.naive_answer(question)}"
        else:
            result = chain_result

        chat_history.append((question, result))

        return_values = [context, chat_history]
        return tuple(return_values)


if __name__ == "__main__":
    args = parser.parse_args()
    model_center = Model_center()

    def inference_row(row, debug=False, comparison_mode=False):
        context, ((q, a),) = model_center(row["Question"], [], debug, comparison_mode)
        result = {"Question": q}
        if comparison_mode:
            # answer, naive_answer = COMPARISON_PATTERN.match(a).groups()
            answer, naive_answer = a.split("**Naive result:**")
            answer = answer.strip("\n")[15:]
            result.update({"Answer": answer, "Passage": context, "Naive Answer": naive_answer})
        else:
            result.update({"Answer": a, "Passage": context})
        result.update({'Reference Answer': row['Reference Answer'], 'Reference Passage': row['Passage']})
        return pd.Series(result)

    df_eval = pd.read_excel(args.data_path)
    df_result = df_eval.apply(inference_row, axis=1, debug=args.debug, comparison_mode=args.comparison_mode)
    print(df_result.head())
    df_result.to_excel("eval_result.xlsx", index=False)

