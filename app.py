from flask import Flask, request, jsonify, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

#Configuration 
ADAPTER_MODEL_ID = "TabithaChebet/falcon_edu_proj" 
# The base model 
BASE_MODEL_ID = "tiiuae/Falcon3-1B-Instruct"
# Max length for generated response
MAX_NEW_TOKENS = 300

app = Flask(__name__)

# Global variables to hold the loaded model and tokenizer
model = None
tokenizer = None

def load_model():
    """Loads the base model, the adapter, and merges them for inference."""
    global model, tokenizer
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    # 2. Load the base model (Using bfloat16 for efficiency)
    print(f"Loading base model: {BASE_MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto" # Distributes across available devices (e.g., GPU)
    )
    
    # 3. Load the LoRA adapter weights
    print(f"Loading adapter from: {ADAPTER_MODEL_ID}...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_ID)

    # 4. Merge the adapter weights into the base model
    print("Merging adapter weights...")
    model = model.merge_and_unload()
    model.eval() # Set model to evaluation mode

    print("Model loading complete and merged. Ready for inference.")


@app.route("/")
def index():
    """Renders the simple HTML interface."""
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_text():
    """API endpoint for text generation."""
    if model is None or tokenizer is None:
        return jsonify({"error": "Model is not yet loaded. Please wait a moment."}), 503
    
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "Prompt is missing."}), 400
        
        print(f"Received prompt: {prompt[:50]}...")

        # Tokenize the prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate the response
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=MAX_NEW_TOKENS, 
                do_sample=True, 
                temperature=0.9,
                pad_token_id=tokenizer.eos_token_id 
            )

        # Decode the generated text and remove the prompt itself
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean the output to only return the model's continuation
        response_text = generated_text[len(prompt):].strip()

        print(f"Generated text: {response_text[:100]}...")

        return jsonify({"text": response_text})

    except Exception as e:
        print(f"Error during generation: {e}")
        return jsonify({"error": str(e)}), 500

# This function runs when the app starts
if __name__ == '__main__':
    # Load the model outside the request loop to do it only once
    load_model()
    # In a deployment environment like Render, gunicorn is used, 

    # On deployment, the Procfile entry below is used.
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
