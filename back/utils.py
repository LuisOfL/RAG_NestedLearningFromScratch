import io
import os
from dotenv import load_dotenv
load_dotenv()
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import BlobServiceClient
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult


AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "pdf"
BLOB_NAME = "7245fcea1c7bf0a995952dabe3da456d.pdf" 

# 2. Credenciales de Azure AI Document Intelligence
DOCUMENT_INTELLIGENCE_ENDPOINT = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
DOCUMENT_INTELLIGENCE_KEY = os.getenv("DOCUMENT_INTELLIGENCE_KEY")

def obtener_texto_de_blob_pdf():
    try:
        # Validación preventiva de variables de entorno
        if not all([AZURE_STORAGE_CONNECTION_STRING, DOCUMENT_INTELLIGENCE_ENDPOINT, DOCUMENT_INTELLIGENCE_KEY]):
            raise ValueError("Faltan configurar variables en tu archivo .env. Por favor verifícalo.")

        # ---------------------------------------------------------
        # PASO 1: Descargar el PDF desde Blob Storage a la memoria
        # ---------------------------------------------------------
        print(f"Conectando a Blob Storage para descargar: {BLOB_NAME}...")
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=BLOB_NAME)
        
        # Descargamos los bytes completos del PDF de forma directa
        pdf_bytes = blob_client.download_blob().readall()
        
        # Convertimos esos bytes en un flujo de datos binarios en memoria RAM
        pdf_stream = io.BytesIO(pdf_bytes)
        
        # ---------------------------------------------------------
        # PASO 2: Procesar los bytes con Azure Document Intelligence
        # ---------------------------------------------------------
        print("Enviando el archivo a Azure AI Document Intelligence...")
        client = DocumentIntelligenceClient(
            endpoint=DOCUMENT_INTELLIGENCE_ENDPOINT, 
            credential=AzureKeyCredential(DOCUMENT_INTELLIGENCE_KEY)
        )
        
        # Invocación corregida con el parámetro 'body' requerido por el SDK moderno
        poller = client.begin_analyze_document(
            model_id="prebuilt-layout",
            body=pdf_stream,
            content_type="application/pdf",
            output_content_format="markdown"
        )
        
        resultado: AnalyzeResult = poller.result()
        
        # Extraemos el string completo estructurado
        texto_final_string = resultado.content
        return texto_final_string

    except Exception as e:
        print(f"Ocurrió un error en el proceso: {e}")
        return None


# ==========================================
# BLOQUE DE EJECUCIÓN PRINCIPAL
# ==========================================
if __name__ == "__main__":
    texto_extraido = obtener_texto_de_blob_pdf()
    
    if texto_extraido:
        print("\n¡Procesamiento exitoso! Contenido del string:\n")
        print(texto_extraido)
    else:
        print("\nNo se pudo extraer el texto debido a los errores reportados arriba.")