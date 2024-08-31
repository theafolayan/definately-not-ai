from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class RequestData(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_text(request_data: RequestData):
    inputs = tokenizer.encode(request_data.prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"response": text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
