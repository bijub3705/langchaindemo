from flask import Flask, request, jsonify, render_template
from src.claims import Claim_RAG
from src.claims.document_loader import DocumentLoader

app = Flask(__name__)

@app.get("/")
def index():
    #member_id = request.args.get('member_id')
    #print(member_id)
    #document_loader.load()
    return render_template("index.html")

@app.post("/generate")
def generate():
    prompt = request.get_json().get("query")
    print(prompt)
    #response = generate_text(prompt)
    response = claim_rag.respond_to_query(prompt)
    return jsonify({"answer": response})

if __name__ == "__main__":
    document_loader = DocumentLoader()
    claim_rag = Claim_RAG(name="claims")
    app.run()