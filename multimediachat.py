import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
import fitz, time
from io import BytesIO

# Configure Google API key using Streamlit secrets
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY_NEW"]
genai.configure(api_key=GOOGLE_API_KEY)

def page_setup():
    st.header("Chat with different types of media/files!", anchor=False, divider="blue")

    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)


def get_typeofpdf():
    st.sidebar.header("Select type of Media", divider='orange')
    typepdf = st.sidebar.radio("Choose one:",
                               ("PDF files",
                                "Images",
                                "Video, mp4 file",
                                "Audio files"))
    return typepdf


def get_llminfo():
    st.sidebar.header("Options", divider='rainbow')
    tip1 = "Select a model you want to use."
    model = st.sidebar.radio("Choose LLM:",
                             ("gemini-1.5-flash",
                              "gemini-1.5-pro",
                             ), help=tip1)
    tip2 = "Lower temperatures are good for prompts that require a less open-ended or creative response, while higher temperatures can lead to more diverse or creative results. A temperature of 0 means that the highest probability tokens are always selected."
    temp = st.sidebar.slider("Temperature:", min_value=0.0,
                             max_value=2.0, value=1.0, step=0.25, help=tip2)
    tip3 = "Used for nucleus sampling. Specify a lower value for less random responses and a higher value for more random responses."
    topp = st.sidebar.slider("Top P:", min_value=0.0,
                             max_value=1.0, value=0.94, step=0.01, help=tip3)
    tip4 = "Number of response tokens, 8194 is limit."
    maxtokens = st.sidebar.slider("Maximum Tokens:", min_value=100,
                                  max_value=5000, value=2000, step=100, help=tip4)
    return model, temp, topp, maxtokens


def main():
    page_setup()
    typepdf = get_typeofpdf()
    model, temperature, top_p, max_tokens = get_llminfo()
    
    if typepdf == "PDF files":
        uploaded_files = st.file_uploader("Choose 1 or more PDF", type='pdf', accept_multiple_files=True)
           
        if uploaded_files:
            text = ""
            for pdf in uploaded_files:
                pdf_reader = PdfReader(BytesIO(pdf.read()))
                for page in pdf_reader.pages:
                    text += page.extract_text()

            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "max_output_tokens": max_tokens,
                "response_mime_type": "text/plain",
            }
            model = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config)
            st.write(model.count_tokens(text)) 
            question = st.text_input("Enter your question and hit return.")
            if question:
                response = model.generate_content([question, text])
                st.write(response.text)
                
    elif typepdf == "Images":
        image_file = st.file_uploader("Upload your image file.")
        if image_file:
            image_data = image_file.read()
            image_file = genai.upload_file(path=image_data)
            
            while image_file.state.name == "PROCESSING":
                time.sleep(10)
                image_file = genai.get_file(image_file.name)
            if image_file.state.name == "FAILED":
                raise ValueError(image_file.state.name)
            
            prompt2 = st.text_input("Enter your prompt.") 
            if prompt2:
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_output_tokens": max_tokens,
                }
                model = genai.GenerativeModel(model_name=model, generation_config=generation_config)
                response = model.generate_content([image_file, prompt2],
                                                  request_options={"timeout": 600})
                st.markdown(response.text)
                
                genai.delete_file(image_file.name)
                print(f'Deleted file {image_file.uri}')
           
    elif typepdf == "Video, mp4 file":
        video_file = st.file_uploader("Upload your video")
        if video_file:
            video_data = video_file.read()
            video_file = genai.upload_file(path=video_data)
            
            while video_file.state.name == "PROCESSING":
                time.sleep(10)
                video_file = genai.get_file(video_file.name)
            if video_file.state.name == "FAILED":
                raise ValueError(video_file.state.name)
            
            prompt3 = st.text_input("Enter your prompt.")
            if prompt3:
                model = genai.GenerativeModel(model_name=model)
                st.write("Making LLM inference request...")
                response = model.generate_content([video_file, prompt3],
                                                  request_options={"timeout": 600})
                st.markdown(response.text)
                
                genai.delete_file(video_file.name)
                print(f'Deleted file {video_file.uri}')
      
    elif typepdf == "Audio files":
        audio_file = st.file_uploader("Upload your audio")
        if audio_file:
            audio_data = audio_file.read()
            audio_file = genai.upload_file(path=audio_data)

            while audio_file.state.name == "PROCESSING":
                time.sleep(10)
                audio_file = genai.get_file(audio_file.name)
            if audio_file.state.name == "FAILED":
                raise ValueError(audio_file.state.name)

            prompt3 = st.text_input("Enter your prompt.")
            if prompt3:
                model = genai.GenerativeModel(model_name=model)
                response = model.generate_content([audio_file, prompt3],
                                                  request_options={"timeout": 600})
                st.markdown(response.text)
                
                genai.delete_file(audio_file.name)
                print(f'Deleted file {audio_file.uri}')


if __name__ == '__main__':
    main()
