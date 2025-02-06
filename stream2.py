import streamlit as st
import os
import numpy as np
import faiss
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from dotenv import load_dotenv
import openai

# Load environment variables (recommended)
load_dotenv()

# Set OpenAI API Key (less recommended, but works if .env not used)
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = st.secrets["openai"]["api_key"]


#os.environ["OPENAI_API_KEY"] = "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"

# Streamlit app
st.title("🎙️ Audio Transcription & Q&A Bot")

uploaded_file = st.file_uploader("Choose an MP3 audio file", type="mp3")

# Define the paths here to ensure they exist even if there's an error
audio_path = "uploaded_audio.mp3"  # Temporary file
transcript_path = "transcript.txt"  # Transcription text file

if uploaded_file is not None:
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("Audio file uploaded.")

    try:
        with st.spinner("Transcribing audio..."):  # Show a spinner
            with open(audio_path, "rb") as audio_file:
                transcript_response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    response_format="text"
                )
            # Extract text from response
            transcript_text = transcript_response.get("text", "")
            if transcript_text:
                with open(transcript_path, "w") as file:
                    file.write(transcript_text)
            else:
                st.error("Failed to get transcription text.")
                st.stop()  # Stop the execution here if no transcription text is returned

        # Load the document and process it
        loader = TextLoader(transcript_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        processed_docs = []

        for doc in documents:
            cleaned_text = doc.page_content.strip().replace("\n", " ")
            processed_docs.extend(text_splitter.split_text(cleaned_text))

        embeddings = OpenAIEmbeddings()

        doc_embeddings = embeddings.embed_documents(processed_docs)
        embedding_matrix = np.array(doc_embeddings).astype('float32')
        dim = embedding_matrix.shape[1]

        faiss_index = faiss.IndexFlatL2(dim)
        faiss_index.add(embedding_matrix)

        llm = ChatOpenAI(model="gpt-3.5-turbo")

        query = st.text_input("Ask me anything about the audio:")

        if st.button("Submit"):  # Button to trigger query
            if query:
                with st.spinner("Generating response..."):  # Spinner for LLM response
                    query_embedding = np.array([embeddings.embed_query(query)]).astype('float32')

                    k = min(5, len(processed_docs))
                    distances, indices = faiss_index.search(query_embedding, k)

                    retrieved_docs = [processed_docs[idx] for idx in indices[0] if idx < len(processed_docs)]

                    min_distance = np.min(distances[0]) if distances[0].size > 0 else float('inf')
                    threshold = 1.5

                    if min_distance > threshold:
                        st.write("Data not present in the provided documents.")
                    else:
                        prompt = (
                            "You are an AI language model that provides responses strictly based on the provided document. "
                            "You must not generate information beyond the given excerpts. If the answer is not found in the provided text, "
                            "respond with: 'The requested information is not available in the provided document.'\n\n"
                            f"I retrieved the following relevant information:\n\n"
                            f"Query: {query}\n"
                            f"Relevant Excerpts:\n" + "\n".join([f"{i+1}. {doc}" for i, doc in enumerate(retrieved_docs)]) +
                            "\n\nYour task is to:\n"
                            "1. Carefully analyze the provided excerpts and extract the most relevant information.\n"
                            "2. Answer the query in a concise and accurate manner using only the retrieved excerpts.\n"
                            "3. If the information is unclear or missing, explicitly state: 'The requested information is not available in the provided document.'\n\n"
                            "Provide a well-structured response based only on the retrieved excerpts."
                        )

                        final_answer = llm.invoke(prompt)
                        st.write("Answer:", final_answer.content)  # Display the answer

    except Exception as e:
        st.error(f"An error occurred: {e}")  # Display errors in Streamlit
        import traceback
        traceback.print_exc()  # Print detailed traceback to the console

    finally:  # Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        if os.path.exists(transcript_path):  # Ensure transcript_path exists before attempting to remove it
            os.remove(transcript_path)
