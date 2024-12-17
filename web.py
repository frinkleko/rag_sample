import gradio as gr
import argparse
import langchain
from langchain_community.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from model import DummyLLM, Qwen25LLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--comparison_mode", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

langchain.verbose = args.verbose


class Model_center:
    def __init__(self):
        self.llm = Qwen25LLM()
        embedder = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")

        # Initialize vector database
        self.vector_db = Chroma(
            persist_directory="./vectorDB/data", embedding_function=embedder
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
            search_kwargs={"k": 2, "score_threshold": 0.3},
        )

        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def naive_answer(self, question):
        prompt = self.template.replace("{context}", "").replace("{question}", question)
        return self.llm.invoke(prompt)

    def question_handler(
        self,
        question: str,
        chat_history: list = [],
        debug=args.debug,
        comparison_mode=args.comparison_mode,
    ):
        if not question:
            return "", chat_history, ""

        # Retrieve relevant documents from vector database
        relevant_docs = self.retriever.get_relevant_documents(question)
        context = (
            "\n\n".join([doc.page_content for doc in relevant_docs])
            or "No relevant information found."
        )

        prompt = self.prompt.invoke({"context": context, "question": question})
        chain_result = self.chain.invoke({"context": context, "question": question})[
            "text"
        ]

        if comparison_mode:
            result = f"**RAG result:**\n{chain_result}\n\n**Naive result:**\n{self.naive_answer(question)}"
        else:
            result = chain_result

        chat_history.append((question, result))

        return_values = ["", chat_history]
        return tuple(return_values)


# Initialize the core functionality object
model_center = Model_center()

# Create a Gradio web interface
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        with gr.Column(scale=15):
            # Display page title
            gr.Markdown(
                """<h1><center>Self LLM</center></h1>
                <center>Self LLM</center>
                """
            )

    with gr.Row():
        with gr.Column(scale=4):
            # Create a chatbot object
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # Create a text box for user input
            msg = gr.Textbox(label="Prompt/Question")
            with gr.Row():
                # Create a submit button
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # Create a clear button to clear the chatbot component
                clear = gr.ClearButton(components=[chatbot], value="Clear console")

        # Set up button click event to call the question_handler function and update the text box and chatbot components
        db_wo_his_btn.click(
            model_center.question_handler, inputs=[msg, chatbot], outputs=[msg, chatbot]
        )

    gr.Markdown(
        """Reminder:<br>
    1. Initializing the database may take some time, please be patient.
    2. If any exceptions occur during usage, they will be displayed in the text input box. Don't panic.<br>
    """
    )
gr.close_all()
demo.launch(share=True)
