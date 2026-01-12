from openai import OpenAI

# ======================================================
# Ollama client (API locale)
# ======================================================
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # valeur factice requise par le client
)

# ======================================================
# Prompt HetGCoT DURCI (anti-hallucination)
# ======================================================
def build_hetgcot_prompt(drug: str, disease: str, evidence: list[str]) -> str:
    evidence_text = "\n".join([f"- {e}" for e in evidence])

    prompt = f"""
You are a biomedical research assistant.
You must strictly rely on the provided evidence paths.
Do NOT introduce general medical knowledge.
Do NOT add new biological mechanisms.
Only rephrase the evidence in a coherent explanation.

Drug: {drug}
Disease: {disease}

Evidence:
{evidence_text}

Task:
Explain the drug–disease relationship using only the evidence above.
Use a concise, mechanistic, scientific style.
"""
    return prompt.strip()

# ======================================================
# Generate explanation with Ollama
# ======================================================
def generate_explanation(drug: str, disease: str, evidence: list[str]) -> str:
    prompt = build_hetgcot_prompt(drug, disease, evidence)

    response = client.chat.completions.create(
        model="llama3",
        messages=[
            {"role": "system", "content": "Only use the provided evidence."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,  # faible = stable, pas d’hallucination
    )

    return response.choices[0].message.content

# ======================================================
# EXEMPLE COMPLET (TON CAS)
# ======================================================
if __name__ == "__main__":

    print("=== OLLAMA HetGCoT EXPLANATION ===")

    # Drug–Disease pair
    drug = "Glyburide"
    disease = "Idiopathic pulmonary fibrosis"

    # Evidence extraite du graphe (chemins multi-hop)
    evidence = [
        "Glyburide targets the gene TIMP2.",
        "TIMP2 is associated with idiopathic pulmonary fibrosis.",
        "TIMP2 participates in extracellular matrix organization.",
        "TIMP2 is involved in matrix metalloproteinase activation."
    ]

    explanation = generate_explanation(drug, disease, evidence)

    print("\n🧠 Explanation:\n")
    print(explanation)
