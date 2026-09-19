"""Legacy Streamlit entry point for the unified v1.1 product."""
import os
import streamlit as st

st.set_page_config(page_title="CustomerIQ", layout="centered")
st.title("CustomerIQ")
st.write("The prediction, batch, model transparency and methodology pages now share one FastAPI interface.")
st.info("Start the API with: python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000")
st.link_button("Open CustomerIQ", os.getenv("CUSTOMERIQ_WEB_URL", "http://127.0.0.1:8000"))
st.caption("This entry point does not load customer datasets or calculate scores.")
