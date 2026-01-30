from pyngrok import ngrok

# Porta do seu app Flask/FastAPI/etc
PORT = 5000

# Abre o túnel ngrok
public_url = ngrok.connect(PORT)
print(f"URL pública: {public_url}")
input("Pressione Enter para encerrar o túnel...")
ngrok.disconnect(public_url)
