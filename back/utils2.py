import os
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2024-08-01-preview"  
)


DEPLOYMENT = "phi-4-mini-instruct"

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

if __name__ == "__main__":
    texto_usuario = "Explícame brevemente qué es una arquitectura RAG"
    
    print("Enviando consulta al modelo...")
    resultado = preguntar_llm(texto_usuario)
    
    print("\n--- Respuesta del LLM ---")
    print(resultado)