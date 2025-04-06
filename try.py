import os
import time
import httpx
import requests
import streamlit as st
import concurrent.futures
from langchain_unstructured import UnstructuredLoader
from unstructured_client import UnstructuredClient
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from unstructured_client.utils import BackoffStrategy, RetryConfig
import re
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# Streamlit UI Configuration
st.set_page_config(page_title="Lecture AI Assistant", page_icon="📚", layout="wide")
st.title("📚 Lecture AI Assistant")
st.caption("Upload PDFs and get Summaries, FAQs, and Quizzes instantly!")

# Sidebar for API Key Input
with st.sidebar:
    st.header("🔑 API Configuration")
    
    groq_api_key = st.text_input("Groq API Key:", type="password")
    unstructured_api_key = st.text_input("Unstructured API Key:", type="password")
    google_api_key = st.text_input("Google API Key:", type="password")

    # Clickable links without underlines
    st.markdown(
        """
        <style>
            .api-links a {
                text-decoration: none !important;
                color: #1a73e8 !important;
                font-weight: 500;
            }
            .api-links a:hover {
                text-decoration: underline !important;
            }
        </style>
        <div class="api-links" style="font-size: 12px; margin-top: 8px; color: #666;">
            🔗 <a href="https://groq.com/" target="_blank">Groq API</a> • 
            🔗 <a href="https://unstructured.io/" target="_blank">Unstructured API</a> • 
            🔗 <a href="https://console.cloud.google.com/apis/credentials" target="_blank">Google API</a>
        </div>
        """,
        unsafe_allow_html=True
    )

    if groq_api_key and unstructured_api_key and google_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        os.environ["UNSTRUCTURED_API_KEY"] = unstructured_api_key
        os.environ["GOOGLE_API_KEY"] = google_api_key  
        st.success("✅ API Keys Set!")


# Ensure the temp_uploads directory exists
UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ File Upload Section
if "uploaded_files" not in st.session_state or not st.session_state.uploaded_files:
    st.subheader("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload Lecture Files", 
        type=["pdf", "txt", "docx", "eml", "png", "jpg"],  # ✅ Now supports images
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
        st.rerun()  # Trigger a rerun to hide the uploader
else:
    st.success(f"{len(st.session_state.uploaded_files)} files uploaded successfully!")

    # Save uploaded files temporarily
    file_paths = []
    for uploaded_file in st.session_state.uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  
        file_paths.append(file_path)

    formatted_file_names = [os.path.basename(file) for file in file_paths]
    st.write("✅ Uploaded Files:", ", ".join(formatted_file_names))  

    # Lazy loading for Unstructured API (only load if needed)
    def load_documents():
        loader = UnstructuredLoader(
            file_path=file_paths,
            api_key=os.getenv("UNSTRUCTURED_API_KEY"),
            partition_via_api=True,
            chunking_strategy="basic",
            max_characters=1000000,
            include_orig_elements=False
        )
        return loader.load()

    # Initialize LLM and Vectorstore only once
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")  
    )
    
    llm = ChatGroq(model_name="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))  

    # Define Prompt Templates
    summary_prompt = ChatPromptTemplate.from_template("""
    Generate a comprehensive bullet-point summary covering ALL key aspects of these notes.
    Include 5-7 main topics with 2-3 specific points each, plus one final insight.

    **Structure exactly like this:**

    • [Main Topic 1]
    - [Specific detail 1]
    - [Specific detail 2]
    • [Main Topic 2]
    - [Specific detail 1] 
    - [Specific detail 2]
    [...continue for all major topics...]

    ▲ Key Insight: [One synthesizing observation]

    Notes Content:
    {context}
    """)
    faq_prompt = ChatPromptTemplate.from_template("Generate 5 FAQs from the lecture notes:\n{context}\n")
    quiz_prompt = ChatPromptTemplate.from_template(
        """Generate exactly 5 MCQs following these REQUIREMENTS:

        FORMAT TEMPLATE (COPY VERBATIM):
        Question|<question_text>|A) <option1>|B) <option2>|C) <option3>|D) <option4>|Answer: <letter>

        MANDATORY RULES (NO EXCEPTIONS):
        1. OUTPUT MUST CONTAIN EXACTLY 5 QUESTIONS
        2. EACH QUESTION MUST:
        - Begin with "Question|" exactly
        - Contain exactly 4 options (A)-D))
        - End with "Answer: X" (X=A/B/C/D only, uppercase)
        - Use exactly 6 pipe characters (|) per line
        3. ABSOLUTELY NO:
        - Parentheses in answers (only "Answer: A")
        - Extra text/headers/footers
        - Line breaks within questions
        - Deviations from the template format

        VIOLATIONS WILL CAUSE PARSING FAILURES

        EXAMPLE (COPY THIS STRUCTURE):
        Question|What is 2+2?|A) 1|B) 2|C) 3|D) 4|Answer: D
        Question|Capital of France?|A) Berlin|B) London|C) Paris|D) Rome|Answer: C

        LECTURE CONTENT:
        {context}"""
    )
    if "docs" not in st.session_state:
        st.session_state.docs = None

    if "files" not in st.session_state:
        st.session_state.files = []

    # Load and process documents when the user presses the "Process Documents" button
    if st.session_state.docs is None and st.button("📖 Process Documents"):
        with st.spinner("🔍 Extracting content..."):
            # Load documents
            st.session_state.docs = load_documents()  # Ensure this loads your documents correctly

            # Create embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

            # Prepare to store unique document-chunks pairs
            document_chunks = []
            
            def create_chunks(doc):
                """
                Function to split document text into chunks using LangChain's CharacterTextSplitter.
                This method splits the document based on character length and adds metadata to each chunk.
                """
                # Extract text content from the document (assuming doc.page_content contains the actual content)
                document_text = doc.page_content

                # Set the maximum chunk size (e.g., 1000 characters per chunk)
                chunk_size = 1000  # Adjust this value based on your needs
                chunk_overlap = 200  # Overlap between chunks to preserve context

                # Create a text splitter with the desired chunk size and overlap
                text_splitter = CharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=len
                )

                # Split the document into raw text chunks
                raw_chunks = text_splitter.split_text(document_text)

                # Create Document objects with metadata for each chunk
                chunked_documents = [
                    Document(page_content=chunk, metadata={"source": doc.metadata["source"]})
                    for chunk in raw_chunks
                ]

                # Return the list of Document objects
                return chunked_documents

            # Generate embeddings for each document with the file name as metadata
            for doc in st.session_state.docs:
                file_name = doc.metadata['source']
                # Create chunks using the updated function
                chunks = create_chunks(doc)  # This now returns Document objects
                document_chunks.extend(chunks)

            # Create the vectorstore with chunked documents tagged by their source
            st.session_state.vectorstore = FAISS.from_documents(document_chunks, embedding=embeddings)

            # Track files in session state (to keep the list of uploaded files)
            st.session_state.files = list(set([doc.metadata['source'] for doc in st.session_state.docs]))

        st.rerun()
    
    if st.button("🔄 Upload Different Files"):
        # Clear all stored session state variables related to file processing
        keys_to_reset = [
            "uploaded_files", "files_data", "vectorstore", "summary_keys",
            "quiz_data", "file_states", "user_answers", "quiz_submitted"
        ]
        
        for key in keys_to_reset:
            if key in st.session_state:
                del st.session_state[key]  # Remove each key from session state

        st.session_state.clear()  # Ensure everything resets

        st.rerun()  # Refresh the app to reflect changes

    # ✅ Show Search Bar Only After Processing is Done
    if st.session_state.docs:
        st.subheader("🔍 Ask a Question About Your Documents")

        if "search_query" not in st.session_state:
            st.session_state.search_query = ""

        search_query = st.text_input("Type your question here:", value=st.session_state.search_query, key="global_search")

        if st.button("🔎 Search"):
            if not search_query.strip():
                st.warning("❗ Please enter a question.")
            else:
                with st.spinner("🔍 Searching your documents..."):
                    retrieved_chunks = []

                    # ✅ Ensure vectorstore exists before searching
                    if "vectorstore" in st.session_state:
                        retrieved_chunks = st.session_state.vectorstore.similarity_search(search_query, k=4)

                    # ✅ Prepare context for LLM
                    if retrieved_chunks:
                        # Make sure chunks are grouped by document for the response
                        doc_chunks = {}
                        for chunk in retrieved_chunks:
                            doc_name = chunk.metadata["source"]
                            if doc_name not in doc_chunks:
                                doc_chunks[doc_name] = []
                            doc_chunks[doc_name].append(chunk.page_content)

                        # Create a context for each document that was retrieved
                        context = "\n".join([f"{doc_name}: {', '.join(doc_chunks[doc_name])}" for doc_name in doc_chunks])
                        answer_prompt = f"Using the following document content, answer this: {search_query}\n\nContext:\n{context}"
                    else:
                        answer_prompt = f"There's no matching content, but try to answer: {search_query}"

                    # ✅ Generate answer using LLM
                    answer = llm.invoke(answer_prompt).content

                # ✅ Display the answer inside a collapsible section
                with st.expander("📌 **View Answer**", expanded=True):
                    st.markdown(f"```{answer}```")

        # ✅ Show documents and summaries
        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            for file_name in st.session_state.files:
                st.subheader(f"📂 {file_name}")

                retriever = st.session_state.vectorstore.as_retriever(
                    search_kwargs={
                        'k': 1000,  # Large enough to get all content
                        'filter': {'source': file_name}  # Only this file
                    }
                )
                all_file_chunks = retriever.get_relevant_documents("")
                
                # Combine all chunks while preserving original order
                file_content = "\n".join(
                    chunk.page_content 
                    for chunk in sorted(
                        all_file_chunks,
                        key=lambda x: x.metadata.get('chunk_index', 0)  # Use chunk ordering if available
                    )
                )

                # Store content separately for each file
                if file_name not in st.session_state:
                    st.session_state[file_name] = {
                        "content": file_content,
                        "summary": None,
                        "faqs": None,
                        "quiz": None,
                        "quiz_submitted": False,
                        "user_answers": {},
                        "quiz_score": None,
                    }

                file_state = st.session_state[file_name]

                # Your existing summary generation code remains exactly the same:
                if st.button(f"📜 Generate Summary - {file_name}"):
                    with st.spinner(f"📝 Creating comprehensive summary for {file_name}..."):
                        try:
                            retriever = st.session_state.vectorstore.as_retriever(
                                search_kwargs={'k': 1000, 'filter': {'source': file_name}}
                            )
                            all_chunks = retriever.get_relevant_documents("")
                            
                            if not all_chunks:  # Fallback
                                all_chunks = [chunk for chunk in st.session_state.vectorstore.similarity_search("", k=1000)
                                            if str(chunk.metadata.get('source', '')).endswith(os.path.basename(file_name))]
                            
                            file_content = "\n".join(chunk.page_content for chunk in all_chunks)
                            
                            if len(file_content.strip()) > 50:
                                response = llm.invoke(summary_prompt.format(context=file_content))
                                file_state["summary"] = response.content
                                
                        except Exception as e:
                            file_state["summary"] = f"• Document Analysis\n  - Covers key concepts from {os.path.basename(file_name)}\n▲ Key Insight: Contains important course material"

                if file_state.get("summary"):
                    with st.expander("📌 **View Summary**", expanded=False):
                        st.markdown(file_state["summary"])


                # Generate Comprehensive FAQs
                if st.button(f"❓ Generate FAQs - {file_name}"):
                    with st.spinner(f"🤔 Analyzing {file_name} for key questions..."):
                        try:
                            # Get comprehensive content from the file
                            retriever = st.session_state.vectorstore.as_retriever(
                                search_kwargs={'k': 1000, 'filter': {'source': file_name}}
                            )
                            all_chunks = retriever.get_relevant_documents("")
                            
                            # Fallback if filtered search fails
                            if not all_chunks:
                                all_chunks = [chunk for chunk in st.session_state.vectorstore.similarity_search("", k=1000)
                                            if str(chunk.metadata.get('source', '')).endswith(os.path.basename(file_name))]
                            
                            context = "\n".join(chunk.page_content for chunk in all_chunks)
                            
                            if len(context.strip()) > 50:
                                response = llm.invoke(faq_prompt.format(context=context))
                                file_state["faqs"] = response.content
                                
                                # Formatting check and enhancement
                                if "?" not in file_state["faqs"] or len(file_state["faqs"].split("\n")) < 3:
                                    file_state["faqs"] = "Key Questions from Document:\n\n" + \
                                                    "\n".join([f"Q{i+1}: {q}" for i, q in enumerate([
                                                        line for line in context.split("\n") 
                                                        if "?" in line or len(line.split()) > 5
                                                    ][:7])])
                                                    
                        except Exception as e:
                            file_state["faqs"] = f"Important Questions:\n\n1. What are the key concepts in this document?\n2. How does this material relate to the course?\n3. What practical applications are discussed?"

                if file_state.get("faqs"):
                    with st.expander("📌 **View Key Questions**", expanded=False):
                        st.markdown(file_state["faqs"])

                # ✅ Generate Unique Quiz
                if st.button(f"📝 Generate Quiz - {file_name}"):
                    with st.spinner(f"🔄 Generating quiz from {file_name}..."):
                        top_chunks = [chunk for chunk in st.session_state.vectorstore.similarity_search(f"Quiz for {file_name}", k=5) if chunk.metadata["source"] == file_name]
                        context = "\n".join(chunk.page_content for chunk in top_chunks)
                        raw_quiz = llm.invoke(quiz_prompt.format(context=context)).content
                        
                        def parse_quiz(raw_quiz):
                            questions = []
                            for line in raw_quiz.strip().split('\n'):
                                line = line.strip()
                                if not line.startswith('Question|'):
                                    continue  # Skip any non-question lines
                                
                                try:
                                    parts = line.split('|')
                                    if len(parts) != 7:  # 1 prefix + 1 question + 4 options + 1 answer
                                        continue
                                    
                                    # Validate answer format
                                    answer_part = parts[-1]
                                    if not answer_part.startswith('Answer: '):
                                        continue
                                    answer = answer_part[8:].strip().upper()  # Extract "A" from "Answer: A"
                                    if answer not in ('A', 'B', 'C', 'D'):
                                        continue
                                    
                                    questions.append({
                                        'question': parts[1].strip(),
                                        'options': [opt[3:].strip() for opt in parts[2:6]],  # Remove "A) " prefixes
                                        'answer': answer
                                    })
                                except:
                                    continue
                            
                            return questions if len(questions) == 5 else None  # Only accept exactly 5 valid questions

                        parsed_quiz = parse_quiz(raw_quiz)
                        if parsed_quiz and len(parsed_quiz) == 5:
                            file_state["quiz"] = parsed_quiz
                            file_state["quiz_submitted"] = False
                            file_state["user_answers"] = {}
                            st.rerun()
                        else:
                            st.error("⚠️ No valid quiz generated. Try again!")

                # ✅ Show Quiz
                if file_state["quiz"]:
                    with st.expander(f"📌 **Practice Questions for {file_name}**", expanded=False):
                        correct_count = 0
                        total_questions = len(file_state["quiz"])

                        for i, q in enumerate(file_state["quiz"]):
                            st.write(f"**Q{i+1}. {q['question']}**")

                            options = q.get("options", [])
                            if len(options) != 4:
                                st.error(f"⚠️ Error parsing options for Q{i+1}.")
                                continue

                            previous_selection = file_state["user_answers"].get(i, None)
                            selected_option = st.radio(
                                f"Choose an option for Q{i+1}:",
                                options,
                                index=options.index(previous_selection) if previous_selection in options else None,
                                key=f"{file_name}_q{i}",
                                disabled=file_state["quiz_submitted"]
                            )

                            file_state["user_answers"][i] = selected_option

                        # Quiz Submission Section
                        if not file_state["quiz_submitted"]:
                            if st.button("✅ Submit Quiz", key=f"submit_quiz_{file_name}"):
                                file_state["quiz_submitted"] = True
                                correct_count = 0

                                for i, q in enumerate(file_state["quiz"]):
                                    user_answer = file_state["user_answers"].get(i, "")
                                    correct_answer_text = q["options"][ord(q["answer"]) - 65]  # Get full text of correct answer

                                    if user_answer == correct_answer_text:
                                        correct_count += 1

                                file_state["quiz_score"] = f"{correct_count} / {len(file_state['quiz'])}"
                                st.rerun()

                # ✅ Move the "See Correct Answers" Expander Outside
                if file_state["quiz_submitted"]:
                    st.success(f"🎯 Your Score: {file_state['quiz_score']}")

                    with st.expander("📖 See Correct Answers", expanded=False):
                        for i, q in enumerate(file_state["quiz"]):
                            st.write(f"**Q{i+1}: {q['question']}**")

                            # Display user's answer
                            user_answer = file_state["user_answers"].get(i, "Not answered")
                            correct_letter = q["answer"]
                            correct_text = q["options"][ord(correct_letter) - 65]
                            
                            # Extract just the text portion for comparison (without the letter)
                            correct_answer_text = correct_text.split(maxsplit=1)[-1] if " " in correct_text else correct_text
                            
                            # Check if user's answer matches the correct text (percentage value)
                            is_correct = str(user_answer).strip() == correct_answer_text.strip()
                            
                            # Display user's answer with color coding
                            if user_answer == "Not answered":
                                st.write(f"**Your Answer:** ❌ {user_answer}")
                            elif is_correct:
                                st.write(f"**Your Answer:** ✅ {user_answer} (Correct)")
                            else:
                                st.write(f"**Your Answer:** ❌ {user_answer} (Should be: {correct_answer_text})")
                            
                            # Display full correct answer (with letter)
                            st.write(f"**Correct Answer:** {correct_letter} {correct_text}")
                            
                            # Add some space between questions
                            st.write("---")

                    if file_state["quiz_score"] == f"{len(file_state['quiz'])} / {len(file_state['quiz'])}":
                        st.success("🏆 Perfect Score!")
