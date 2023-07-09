import os
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain.tools import SceneXplainTool
from langchain.llms import OpenAI
from langchain.agents import initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents import load_tools
from htmlTemplates import css, bot_template, user_template
import requests
from io import BytesIO

def main():
    st.set_page_config(page_title="Image Captioning App")
    st.write(css, unsafe_allow_html=True)
    st.session_state.setdefault('image_url', '')
    st.session_state.setdefault('image', None)
    top_image = Image.open('trippyPattern.png')
    st.image(top_image)
    st.title("SceneXplain Image Captioning App")
    st.session_state.image_url = st.text_input(label="Image URL")
    if st.button("Submit URL") and st.session_state.image_url:
        response = requests.get(st.session_state.image_url)
        st.session_state.image = Image.open(BytesIO(response.content))
    if st.session_state.image_url:
        st.image( st.session_state.image)
        context = st.text_area(label="Additional Context (Optional)", height=100)
        if st.button("Generate Image Caption"):
            with st.spinner("Generating..."):
                tools = load_tools(["sceneXplain"])
                llm = OpenAI(temperature=0.5)
                memory = ConversationBufferMemory(memory_key="chat_history")
                agent = initialize_agent(tools, llm, memory=memory, agent="conversational-react-description", verbose=True)
                if context:
                    output = agent.run(
                        input=(f"Considering this context: {context}\nGenerate a caption for this image: {st.session_state.image_url}")
                    )
                else:
                    output = agent.run(
                        input=(f"Generate a caption for this image: {st.session_state.image_url}")
                    )
                st.markdown(bot_template.replace("{{MSG}}", output), unsafe_allow_html=True)


if __name__ == "__main__":
    load_dotenv()
    main()
