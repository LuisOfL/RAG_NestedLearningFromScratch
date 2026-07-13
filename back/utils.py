import io
import os
from dotenv import load_dotenv
load_dotenv()
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from openai import AzureOpenAI

AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
DEPLOYMENT = "phi-4-mini-instruct"
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")


client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-08-01-preview"  
)


def obtener_texto_de_blob_pdf(BLOB_NAME, CONTAINER_NAME):
    try:

        if not all([AZURE_STORAGE_CONNECTION_STRING, DOCUMENT_INTELLIGENCE_ENDPOINT, DOCUMENT_INTELLIGENCE_KEY]):
            raise ValueError("Faltan configurar variables en tu archivo .env. Por favor verifícalo.")

        print(f"Conectando a Blob Storage para descargar: {BLOB_NAME}...")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)
        

        pdf_bytes = blob_client.download_blob().readall()
        
    
        pdf_stream = io.BytesIO(pdf_bytes)
        
        print("Enviando el archivo a Azure AI Document Intelligence...")
        client = DocumentIntelligenceClient(
            endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT, 
            credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
        )
        
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=pdf_stream,
            content_type="application/pdf",
            output_content_format="markdown"
        )
        
        resultado: AnalyzeResult = poller.result()
        texto_final_string = resultado.content
        return texto_final_string

    except Exception as e:
        print(f"Ocurrió un error en el proceso: {e}")
        return None






def preguntar_llm(prompt: str) -> str:
    try:
        respuesta = client.chat.completions.create(
            model=DEPLOYMENT, 
            messages=[
                {"role": "system", "content": "Eres un asistente de IA servicial y experto en datos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7 
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"Ocurrió un error al conectar con Azure OpenAI: {e}"