# python basic lib

import os 


# import gradio -->  UI
import gradio as gr

# import langchain tools 

from langchain_openai import ChatOpenAI, OpenAIEmbeddings #for connection to Openai LLM and Embedding models
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
#from langchain_classic.chains import RetrievalQA
#from typer import prompt

#create global variables for the app

vector_store = None
#qa_chain = None
current_pdf_name = None #pdf file name
retriever = None
llm = None # brain of the app.

# pdf processing function

def process_pdf(pdf_file):
    global vector_store,  current_pdf_name, retriever, llm # update the global variables , not local variables



    if pdf_file is None: 
        return "please upoad a PDF files" # prevents user upload a northing
    
    # check api key is available or not 
    if not os.getenv("OPENAI_API_KEY"):
        return "OPENAI_API_KEY is not set." 


    try: 
        current_pdf_name = os.path.basename(pdf_file) #get PDF file path and name

        loader = PyPDFLoader(pdf_file) # create PDF loader
        documents = loader.load() # load the pdf from loader 

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000   #each chunk keep about 1000 characters
            ,chunk_overlap=200   #the next chunk repeats about 200 characters from pervious one.
        )
        
        chunks = text_splitter.split_documents(documents)

        # embeddings open ai model & vector store
        # embedding model : This model converts each chunk from text into numbers.
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small") #open AI embed model


        #build vector DB (Chroma)
        vector_store = Chroma.from_documents(
            documents = chunks
            ,embedding = embeddings
        )

        #return the vector database into a search tool.
        retriever = vector_store.as_retriever(search_kwargs={"k": 4}) # {"k": 4} means when user ask a question, find the top 4 most relevant chunks

        # define llm 
        llm = ChatOpenAI(
            model = "gpt-4.1-mini"
            ,temperature=0
        )

        # return to the front end
        return (
            f"PDF loaded successfully: {current_pdf_name} |" 
            f"Pages: {len(documents)} |" 
            f"chunks: {len(chunks)} |"
            f"Vector DB and Retriever ready |"
            f"LLM : {llm.model_name}"
        )
                             

    except Exception as e: #error handling
        return f"Error occurred while processing PDF: {str(e)}"


#chat function with the PDF and LLM 
def chat_with_pdf(message, history):
    global retriever, llm

    if retriever is None or llm is None:
        return "please upload and process the PDF."
    if not message.strip(): #Checks if the message is empty.
        return "please enter a question."
    
    try:
        docs = retriever.invoke(message) #search chunks in chorma 
        context = "\n\n".join([doc.page_content for doc in docs]) #Combines retrieved chunk text into one string.
        prompt = f"""

you are the AI assistant . 
answer the user's question only based on the PDF content below. 
If the answer is not in the PDF, say: "I cannot find that in the PDF."

PDF content :
{context} 

user question: 
{message}

        """
        response = llm.invoke(prompt)
        
        pages = sorted(
            set(
                doc.metadata.get("page", 0) +1
                for doc in docs 
                if "page" in doc.metadata
            )
        )

        if pages:
            return f"{response.content}\n\nSource Pages: {', '.join(map(str, pages))}"

        return response.content
    
    except Exception as e:
        return f"Error answering question: {str(e)}"


# handling PDF upload with Gradio.


with gr.Blocks() as demo:
    gr.Markdown("# PDF Chat App")

    # file input
    pdf_input = gr.File(
        label = "Upload your PDF"
        ,file_types = [".pdf"]
        ,type = "filepath" #file path string
    )

    process_btn = gr.Button("Process PDF")
    status_box = gr.Textbox(label = "Status")

    process_btn.click(
        fn = process_pdf #run process_pdf when click the button
        ,inputs = pdf_input
        ,outputs = status_box
    )

    gr.ChatInterface(
        fn = chat_with_pdf
        ,title="Chat with the PDF"
    )
    


demo.launch()