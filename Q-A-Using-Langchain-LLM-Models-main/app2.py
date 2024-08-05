import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
from PyPDF2 import PdfReader
import spacy
from textblob import TextBlob
from googletrans import Translator
import subprocess
import base64
from gtts import gTTS
import uuid
import logging
from flask import send_from_directory

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Ensure spaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

genai.configure(api_key=google_api_key)

# Ensure the static directory exists
if not os.path.exists("static"):
    os.makedirs("static")

# Load models and embeddings once
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chat_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
prompt_template = """
Answer the question as detailed as possible from the provided context, make sure to provide all the details. If the answer is not in
provided context just say, "answer is not available in the context", don't provide the wrong answer.\n\n
Context:\n{context}\n
Question:\n{question}\n
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = load_qa_chain(chat_model, chain_type="stuff", prompt=prompt)

# Function to fetch text from a URL
def get_url_text(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = "\n".join([para.get_text() for para in paragraphs])
        return text
    except Exception as e:
        logging.error(f"Error fetching the URL content: {e}")
        return f"Error fetching the URL content: {e}"

# Function to fetch text from a PDF
def get_pdf_text(pdf_file):
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
        return text
    except Exception as e:
        logging.error(f"Error reading the PDF content: {e}")
        return f"Error reading the PDF content: {e}"

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create a vector store from text chunks
def get_vector_store(text_chunks):
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to handle user input
def user_input(user_question):
    try:
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        response = qa_chain({"input_documents": docs, "question": user_question})
        return docs, response["output_text"]
    except Exception as e:
        logging.error(f"Error handling user input: {e}")
        return [], f"Error: {e}"

# Function to perform named entity recognition (NER)
def perform_ner(text):
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except Exception as e:
        logging.error(f"Error performing NER: {e}")
        return []

# Function to perform sentiment analysis
def analyze_sentiment(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment
        return sentiment
    except Exception as e:
        logging.error(f"Error analyzing sentiment: {e}")
        return None

# Function to translate text to a specified language
def translate_text(text, target_lang='en'):
    try:
        if not text:
            return "No text to translate."
        translator = Translator()
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return "Error translating text."

# Function to summarize text
def summarize_text(text, num_sentences=3):
    try:
        doc = nlp(text)
        sentences = list(doc.sents)
        summary = " ".join([str(sent) for sent in sentences[:num_sentences]])
        return summary
    except Exception as e:
        logging.error(f"Error summarizing text: {e}")
        return "Error summarizing text."

# Function to convert text to speech
def text_to_speech(text):
    try:
        tts = gTTS(text)
        filename = f"static/{uuid.uuid4()}.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        logging.error(f"Error generating speech: {e}")
        return None

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"])
server = app.server

# Serve static files
@server.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

app.layout = html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("Langchain Q&A using Large Language Models 💁", className="text-center mt-4", style={'color': '#272343'}),
                dcc.Tabs([
                    dcc.Tab(label='Home', children=[
                        html.Div([
                            html.H3("Welcome to the Chat Article App!", className="text-center mt-4", style={'color': '#000080'}),
                            html.P("Use the navigation tabs to switch between different functionalities.", className="text-center mt-2", style={'color': '#4B0082'}),
                            html.Ul([
                                html.Li("Process URL: Analyze and ask questions about articles from a URL.", className="mt-2", style={'color': '#4B0082'}),
                                html.Li("Process PDF: Analyze and ask questions about articles from a PDF file.", className="mt-2", style={'color': '#4B0082'})
                            ])
                        ], className="p-4 rounded shadow-sm", style={'backgroundColor': '#e3f6f5'})
                    ]),
                    dcc.Tab(label='Process URL', children=[
                        html.Div([
                            dbc.Input(id="url-input", placeholder="Enter URL of the article", type="text", className="mt-3 mb-3"),
                            dbc.Button("Submit & Process URL", id="submit-url", color="primary", className="mb-3"),
                            dcc.Loading(id="url-loading", children=[
                                html.Div(id="url-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e9ecef'})
                            ]),
                            html.Div([
                                dbc.Input(id="url-question", placeholder="Ask a Question from the Article", type="text", className="mt-3 mb-3"),
                                dbc.Select(id="url-translate", options=[
                                    {'label': 'No Translation', 'value': ''},
                                    {'label': 'English', 'value': 'en'},
                                    {'label': 'Hindi', 'value': 'hi'},
                                    {'label': 'Bengali', 'value': 'bn'},
                                    {'label': 'Telugu', 'value': 'te'},
                                    {'label': 'Marathi', 'value': 'mr'},
                                    {'label': 'Tamil', 'value': 'ta'},
                                    {'label': 'Gujarati', 'value': 'gu'},
                                    {'label': 'Kannada', 'value': 'kn'},
                                    {'label': 'Malayalam', 'value': 'ml'},
                                    {'label': 'Odia', 'value': 'or'},
                                    {'label': 'Punjabi', 'value': 'pa'}
                                ], placeholder="Translate text to:", className="mt-3 mb-3"),
                                dbc.Button("Submit Question", id="submit-url-question", color="primary", className="mb-3"),
                                dcc.Loading(id="url-question-loading", children=[
                                    html.Div(id="url-question-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e9ecef'})
                                ]),
                                html.Audio(id="url-audio", controls=True, autoPlay=True, className="mt-3 mb-3")
                            ], className="p-4 rounded shadow-sm", style={'backgroundColor': '#e3f6f5'})
                        ])
                    ]),
                    dcc.Tab(label='Process PDF', children=[
                        html.Div([
                            dcc.Upload(
                                id='upload-pdf',
                                children=html.Div([
                                    'Drag and Drop or ',
                                    html.A('Select PDF File')
                                ]),
                                style={
                                    'width': '100%',
                                    'height': '60px',
                                    'lineHeight': '60px',
                                    'borderWidth': '1px',
                                    'borderStyle': 'dashed',
                                    'borderRadius': '5px',
                                    'textAlign': 'center',
                                    'margin': '10px',
                                    'backgroundColor': '#F8F8FF'
                                },
                                multiple=False
                            ),
                            dcc.Loading(id="pdf-loading", children=[
                                html.Div(id="pdf-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e3f6f5'})
                            ]),
                            html.Div([
                                dbc.Input(id="pdf-question", placeholder="Ask a Question from the Article", type="text", className="mt-3 mb-3"),
                                dbc.Select(id="pdf-translate", options=[
                                    {'label': 'No Translation', 'value': ''},
                                    {'label': 'English', 'value': 'en'},
                                    {'label': 'Hindi', 'value': 'hi'},
                                    {'label': 'Bengali', 'value': 'bn'},
                                    {'label': 'Telugu', 'value': 'te'},
                                    {'label': 'Marathi', 'value': 'mr'},
                                    {'label': 'Tamil', 'value': 'ta'},
                                    {'label': 'Gujarati', 'value': 'gu'},
                                    {'label': 'Kannada', 'value': 'kn'},
                                    {'label': 'Malayalam', 'value': 'ml'},
                                    {'label': 'Odia', 'value': 'or'},
                                    {'label': 'Punjabi', 'value': 'pa'}
                                ], placeholder="Translate text to:", className="mt-3 mb-3"),
                                dbc.Button("Submit Question", id="submit-pdf-question", color="primary", className="mb-3"),
                                dcc.Loading(id="pdf-question-loading", children=[
                                    html.Div(id="pdf-question-output", className="p-3 rounded shadow-sm", style={'backgroundColor': '#e3f6f5'})
                                ]),
                                html.Audio(id="pdf-audio", controls=True, autoPlay=True, className="mt-3 mb-3")
                            ], className="p-4 rounded shadow-sm", style={'backgroundColor': '#e3f6f5'})
                        ])
                    ])
                ], className="rounded shadow-sm", style={'backgroundColor': '#bae8e8', 'padding': '20px'})
            ])
        ])
    ])
])

@app.callback(
    Output("url-output", "children"),
    Input("submit-url", "n_clicks"),
    State("url-input", "value")
)
def process_url(n_clicks, url):
    if n_clicks and url:
        logging.debug(f"Processing URL: {url}")
        raw_text = get_url_text(url)
        if raw_text.startswith("Error"):
            return html.Div(raw_text, className="text-danger")
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return html.Div("Processing complete!", className="text-success")
    return ""

@app.callback(
    [Output("url-question-output", "children"), Output("url-audio", "src")],
    Input("submit-url-question", "n_clicks"),
    State("url-question", "value"),
    State("url-translate", "value")
)
def handle_url_question(n_clicks, user_question, target_lang):
    if n_clicks and user_question:
        logging.debug(f"User question: {user_question}")
        docs, response = user_input(user_question)
        if isinstance(response, str) and response.startswith("Error"):
            response_text = "Sorry, an error occurred while processing your question."
        elif not response:
            response_text = "Sorry, no answer could be found."
        else:
            response_text = response

        # Convert response to point-wise format
        points = response_text.split('. ')
        formatted_response = html.Ul([html.Li(point) for point in points if point])

        outputs = [html.Div(f"Reply:", className="mt-2"), formatted_response]

        if target_lang and target_lang != '':
            translation = translate_text(response_text, target_lang)
            if translation.startswith("Error") or translation == "No text to translate.":
                translation = "Translation not available."
            translated_points = translation.split('. ')
            formatted_translation = html.Ul([html.Li(point) for point in translated_points if point])
            outputs.append(html.Div(f"Translation:", className="mt-2"))
            outputs.append(formatted_translation)

        if "ner" in user_question.lower():
            entities = perform_ner(response_text)
            outputs.append(html.Div(f"Named Entities: {entities}", className="mt-2"))

        if "summarize" in user_question.lower():
            summary = summarize_text(response_text)
            outputs.append(html.Div(f"Summary: {summary}", className="mt-2"))

        if "sentiment" in user_question.lower():
            sentiment = analyze_sentiment(response_text)
            outputs.append(html.Div(f"Sentiment Analysis: {sentiment}", className="mt-2"))

        # Generate and return audio file for TTS
        audio_file = text_to_speech(response_text)
        audio_src = f"/static/{os.path.basename(audio_file)}" if audio_file else None
        return [outputs, audio_src]
    return [None, None]

@app.callback(
    Output("pdf-output", "children"),
    Input('upload-pdf', 'contents'),
    State('upload-pdf', 'filename')
)
def process_pdf(contents, filename):
    if contents:
        logging.debug(f"Processing PDF: {filename}")
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        # Use the current working directory or specify a path
        file_path = os.path.join(os.getcwd(), filename)
        with open(file_path, 'wb') as f:
            f.write(decoded)
        
        raw_text = get_pdf_text(file_path)
        if raw_text.startswith("Error"):
            return html.Div(raw_text, className="text-danger")
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        return html.Div("Processing complete!", className="text-success")
    return ""

@app.callback(
    [Output("pdf-question-output", "children"), Output("pdf-audio", "src")],
    Input("submit-pdf-question", "n_clicks"),
    State("pdf-question", "value"),
    State("pdf-translate", "value")
)
def handle_pdf_question(n_clicks, user_question, target_lang):
    if n_clicks and user_question:
        logging.debug(f"User question: {user_question}")
        docs, response = user_input(user_question)
        if isinstance(response, str) and response.startswith("Error"):
            response_text = "Sorry, an error occurred while processing your question."
        elif not response:
            response_text = "Sorry, no answer could be found."
        else:
            response_text = response

        # Convert response to point-wise format
        points = response_text.split('. ')
        formatted_response = html.Ul([html.Li(point) for point in points if point])

        outputs = [html.Div(f"Reply:", className="mt-2"), formatted_response]

        if target_lang and target_lang != '':
            translation = translate_text(response_text, target_lang)
            if translation.startswith("Error") or translation == "No text to translate.":
                translation = "Translation not available."
            translated_points = translation.split('. ')
            formatted_translation = html.Ul([html.Li(point) for point in translated_points if point])
            outputs.append(html.Div(f"Translation:", className="mt-2"))
            outputs.append(formatted_translation)

        if "ner" in user_question.lower():
            entities = perform_ner(response_text)
            outputs.append(html.Div(f"Named Entities: {entities}", className="mt-2"))

        if "summarize" in user_question.lower():
            summary = summarize_text(response_text)
            outputs.append(html.Div(f"Summary: {summary}", className="mt-2"))

        if "sentiment" in user_question.lower():
            sentiment = analyze_sentiment(response_text)
            outputs.append(html.Div(f"Sentiment Analysis: {sentiment}", className="mt-2"))

        # Generate and return audio file for TTS
        audio_file = text_to_speech(response_text)
        audio_src = f"/static/{os.path.basename(audio_file)}" if audio_file else None
        return [outputs, audio_src]
    return [None, None]

if __name__ == "__main__":
    app.run_server(debug=True)