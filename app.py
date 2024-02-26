import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()
os.getenv("GOOGLE_API_KEY")

# Configure Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to query the API for irrelevant questions
def get_api_response(question):
    api_url = "https://api.example.com/answer"
    params = {"question": question}
    response = requests.get(api_url, params=params)
    return response.json().get("answer", "No information available.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, domain, pdf_docs):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    if pdf_docs:
        # Add progress bar while processing PDFs
        with st.spinner('Processing PDFs...'):
            # Extract text from uploaded PDF documents
            raw_text = get_pdf_text(pdf_docs)

        # Error handling for PDF processing
        if not raw_text:
            st.error("Error processing PDFs. Please try again.")
            return

        # Split text into chunks
        text_chunks = get_text_chunks(raw_text)

        # Generate embeddings and create vector store
        get_vector_store(text_chunks)
        st.success("Done processing PDFs.")

        # Use processed PDFs for similarity search
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)

    else:
        # If no PDFs are uploaded, set docs to an empty list
        docs = []

    chain = get_conversational_chain()

    # Check if the user's question is related to the company
    if "company" in user_question.lower():
        # Provide response related to the company's website and contact details
        company_response = (
            "For information about Paperflite, you can visit our website: [Paperflite](https://www.paperflite.com/)\n"
            "For further details, please contact us at:\n"
            "Email: xyz@gmail.com\n"
            "Mobile No: +91 xxxxx xxxxx"
        )
        st.sidebar.header("Suggested FAQs:")
        st.sidebar.write("No suggested FAQs for company-related questions.")
        st.write("**Reply:** ", company_response)

    elif "faq" in user_question.lower():
        # Provide the common FAQs if the user asks about FAQs
        st.sidebar.header("Suggested FAQs:")
        for i, faq in enumerate(get_common_faqs()):
            st.sidebar.write(f"{i + 1}. {faq}")
        st.write("**Reply:** ", "For FAQs, please check the sidebar.")

    else:
        # Display the response from the conversational chain
        response = chain(
            {"input_documents": docs, "question": user_question}, return_only_outputs=True
        )

        # Display suggested FAQs in the sidebar based on the selected domain
        suggested_faqs = get_suggested_faqs(domain)
        st.sidebar.header("Suggested FAQs:")
        for i, faq in enumerate(suggested_faqs):
            st.sidebar.write(f"{i + 1}. {faq}")

        # Check if the response is relevant, otherwise, query the API
        if "does not contain any information" in response["output_text"].lower():
            api_response = get_api_response(user_question)
            st.write("**Reply:** ", api_response)
        else:
            st.write("**Reply:** ", response["output_text"])

def get_common_faqs():
    return [
        "FAQ 1: What is Paperflite?",
        "FAQ 2: How can I get started with Paperflite?",
        "FAQ 3: What are the key features of Paperflite?",
        # Add more common FAQs
    ]

def get_suggested_faqs(domain):
    # Modify this function to return suggested FAQs based on the selected domain
    # For example, you can fetch FAQs from a database or hardcode them
    suggested_faqs = {
        "MARKETING TEAM:": [
            "1. Can I create personalized versions of collateral for different buyer personas?",
            "2. How do I ensure brand consistency across all our marketing collateral?",
            "3. Can I set up automated workflows for collateral updates and distribution?",
        ],
        "SALES TEAM": [
            "1. Can I create personalized sales decks or proposals for specific customers or opportunities?",
            "2. Can I integrate the software with our CRM system for seamless access to collateral?",
            "3. How can I track which collateral pieces are being used and shared by my sales reps?",
        ],
        "CORPORATE COMMUNICATION TEAM": [
            "1. What permissions and access controls can I set up for different teams or user roles?",
            "2. Can I create customized templates for different types of collateral (e.g., presentations, reports, fact sheets)?",
            "3. Does the software support multilingual content and localization?",
        ],
        "CREATIVE AGENCY AND DESIGN STUDIO": [
            "1. Can I create custom approval workflows for client review and sign-off?",
            "2. How can I track and report on collateral usage and engagement metrics for clients?",
            "3. Does the software integrate with popular design tools or creative suites?",
        ],
        "PUBLISHER AND CONTENT CREATOR": [
            "1. How does the software handle version control and revisions for my content?",
            "2. Does the software support collaboration and co-authoring?",
            "3. What analytics and reporting features are available for tracking content performance?",
        ],
        "RETAIL AND E-COMMERCE BUSINESS": [
            "1. How do I ensure brand consistency across all my marketing and sales materials?",
            "2. How can I track and report on the usage and effectiveness of my marketing materials?",
            "3. Can I set up automated workflows for collateral updates and distribution?",
        ],
        "EDUCATIONAL INSTITUTION": [
            "1. Can I create customized course packs or study materials for different programs or classes?",
            "2. Can faculty and students collaborate on educational content within the software?",
            "3. How can I track and report on the usage and engagement of my educational materials?",
        ],
        "NON-PROFIT ORGANISATION": [
            "1. How do I ensure brand consistency across all my organizational communications?",
            "2. Can I set up automated workflows for content approvals and distribution?",
            "3. Does the software support multilingual content and localization for reaching global audiences?",
        ],
        "HEALTHCARE": [
            "1. Can I create personalized versions of medical content for different patient demographics or conditions?",
            "2. Can healthcare providers collaborate on creating and reviewing medical content within the software?",
            "3. Does the software support multimedia and interactive content for enhanced patient education experiences?",
        ],
        # ... Add remaining FAQs for other domains
    }
    return suggested_faqs.get(domain, [])

def main():
    # Set page configuration
    st.set_page_config("Superflite bot")

    # Display header
    st.header("Hi, I am Superflite customer support chatbot with RAG using Gemini")

    # Create sidebar for suggested FAQs based on the selected domain
    domain_options = [
        "MARKETING TEAM:",
        "SALES TEAM",
        "CORPORATE COMMUNICATION TEAM",
        "CREATIVE AGENCY AND DESIGN STUDIO",
        "PUBLISHER AND CONTENT CREATOR",
        "RETAIL AND E-COMMERCE BUSINESS",
        "EDUCATIONAL INSTITUTION",
        "NON-PROFIT ORGANISATION",
        "HEALTHCARE",
    ]
    selected_domain = st.sidebar.selectbox("Select your Domain:", domain_options)
    suggested_faqs = get_suggested_faqs(selected_domain)

    # Display suggested FAQs in the sidebar
    st.sidebar.header("Suggested FAQs:")
    for i, faq in enumerate(suggested_faqs):
        st.sidebar.write(f"{i + 1}. {faq}")

    # Add file uploader for PDF documents
    pdf_docs = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)

    # Create text input for user question
    user_question = st.text_input("Ask your query")

    # Process user input if a question is entered and PDFs are uploaded
    if user_question:
        user_input(user_question, selected_domain, pdf_docs)

if __name__ == "__main__":
    main()
