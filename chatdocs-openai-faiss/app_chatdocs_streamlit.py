

#Index Search
#The script loads a FAISS vector database using load_vector_db() to perform similarity-based searches on document embeddings created with OpenAI embeddings.
# Users input a keyword, and the similarity_search method retrieves the top relevant documents from the database.

#Question Answering
#After retrieving relevant documents, the script uses a question-answering chain (load_qa_chain) 
# with OpenAI's LLM to answer user queries about the documents by employing the "stuff" "map_reduce" and "refine" method for comprehensive responses.
import os
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "useyourapikey"

# Use OpenAI embeddings for consistency
embeddings = OpenAIEmbeddings()

# Function to load FAISS vector database
def load_vector_db(db_folder):
    try:
        # Check file existence
        faiss_file = os.path.join(db_folder, "index.faiss")
        pkl_file = os.path.join(db_folder, "index.pkl")
        if not os.path.exists(faiss_file) or not os.path.exists(pkl_file):
            raise FileNotFoundError(f"FAISS database files not found in {db_folder}")
        
        # Load FAISS database
        db = FAISS.load_local(db_folder, embeddings, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading FAISS database: {e}")
        return None

# Streamlit App
def main():
    st.title("📚 ChatDocs: ACL Conference Research Companion")
    db_folder = r"D:\Fall_2024 MSDA\Text Analytics\Project\myenv\faiss_db_openAI_source"  # Path to your FAISS folder

    st.write("Loading FAISS vector database...")
    db = load_vector_db(db_folder)

    if db:
        st.success("FAISS database loaded successfully!")
        num_docs_in_db = db.index.ntotal
        st.write(f"Number of documents in FAISS database: {num_docs_in_db}")

        # Step 1: Search for relevant papers
        keyword = st.text_input("Enter a keyword to search for relevant papers:")
        if keyword:
            try:
                # Perform similarity search for relevant papers
                docs = db.similarity_search(keyword, k=4)

                if docs:
                    st.subheader("Top Relevant Papers:")
                    for i, doc in enumerate(docs):
                        st.markdown(f"### Paper {i+1}:")
                        st.write(doc.page_content[:300] + "...")  # Show a snippet of the content
                        source = doc.metadata.get('source', 'Unknown')
                        if source != 'Unknown':
                            st.markdown(f"[Source: {source}](./{source})")  # Clickable link to the file
                        else:
                            st.markdown(f"Source: {source}")
                    
                    # Step 2: Question Answering
                    st.markdown("---")
                    st.subheader("Ask a specific question about these papers:")
                    question = st.text_input("Enter your question:")
                    if question:
                        try:
                            llm = OpenAI(temperature=0)  # Using OpenAI API key from the environment variable
                            qa_chain = load_qa_chain(llm, chain_type="map_reduce")
                            response = qa_chain.run(input_documents=docs, question=question)

                            st.subheader("Answer:")
                            st.write(response)
                        except Exception as e:
                            st.error(f"Error during QA: {e}")
                else:
                    st.warning("No relevant papers found for the given keyword.")
            except Exception as e:
                st.error(f"Error during similarity search: {e}")
    else:
        st.error("Failed to load FAISS vector database.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
