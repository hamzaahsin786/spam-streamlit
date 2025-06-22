import streamlit as st
import joblib

# Load model
model = joblib.load("spam_classifier.pkl")

# Page configuration
st.set_page_config(page_title="Spam Message Detector", page_icon="📧")

# Title and subtitle
st.title("📧 Spam Message Detector")
st.markdown("Check whether a message is **Spam** or **Not Spam** using a trained ML model.")

# Input form
with st.form(key="spam_form"):
    message = st.text_area("✍️ Enter your message below:", height=150)
    submit = st.form_submit_button("🚀 Detect Spam")

if submit:
    if message.strip():
        prediction = model.predict([message])
        result = "🚫 This message is likely **Spam**." if prediction[0] == 1 else "✅ This message is **Not Spam**."
        st.success(result)
    else:
        st.warning("Please enter a message to analyze.")

# Footer
st.markdown("---")
st.markdown("Made by **Muhammad Hamza** · Powered by `scikit-learn` · Hosted on Streamlit")