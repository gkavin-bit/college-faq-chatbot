import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="College FAQ Chatbot")

st.title("🎓 College FAQ Chatbot")
st.write("Ask your college-related questions below.")

questions = [
    "How can I apply for admission?",
    "What are the college fees?",
    "When is the semester exam?",
    "Where is the college library?",
    "Is hostel facility available?",
    "Tell me about placement training."
]

answers = [
    "Please contact the college office for admission details.",
    "Please contact the accounts department for fee details.",
    "Please check the official examination notice.",
    "Please check the college website for library details.",
    "Please contact the college office for hostel details.",
    "Please contact the placement department."
]

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(questions)

user_question = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_question.strip():
        user_vector = vectorizer.transform([user_question])
        scores = cosine_similarity(user_vector, vectors)[0]

        best_match = scores.argmax()

        if scores[best_match] >= 0.15:
            st.success(answers[best_match])
        else:
            st.warning(
                "Sorry, I could not understand. "
                "Please contact the college office."
            )
    else:
        st.warning("Please enter a question.")
