import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel


# ------------------ 1. Define Model Architecture ------------------ #
class CodeReviewClassifier(nn.Module):
    def __init__(self, hidden_dim=768, num_labels=2):
        super(CodeReviewClassifier, self).__init__()
        self.codebert = RobertaModel.from_pretrained("microsoft/codebert-base")
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# ------------------ 2. Load Model and Tokenizer ------------------ #
# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

# Load model
model = CodeReviewClassifier()
model.load_state_dict(
    torch.load("codebert_review_model_balanced.pt", map_location=torch.device("cpu"))
)
model.eval()


# ------------------ 3. Define Prediction Function ------------------ #
def predict_issue(code_snippet, model, tokenizer, device="cpu"):
    model.eval()
    inputs = tokenizer(
        code_snippet, return_tensors="pt", padding=True, truncation=True, max_length=128
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    label = "🚫 Issue Detected" if pred == 1 else "✅ Code Looks Good"
    return label, confidence


# # ------------------ 4. Run Predictions on Test Code Snippets ------------------ #
# test_codes = [
#     "eval(input('Enter: '))",  # bad
#     "def add(x, y): return x + y",  # good
#     "import os; os.system('rm -rf /')",  # bad
#     "for i in range(5): print(i)",  # good
#     "f = open('file.txt'); content = f.read()",  # bad
#     "def divide(a, b): return a / b if b != 0 else None",  # good
# ]
# ------------------ 4. Run Predictions on User Input ------------------ #
while True:
    code = input("\nEnter a code snippet to analyze (or type 'exit' to quit):\n")
    if code.lower() == "exit":
        print("Exiting...")
        break
    result, conf = predict_issue(code, model, tokenizer)
    print(f"\n🔍 Code:\n{code}\n➡️ Result: {result} (Confidence: {conf:.2f})")

# # Predict and display results
# for code in test_codes:
#     result, conf = predict_issue(code, model, tokenizer)
#     print(f"\n🔍 Code:\n{code}\n➡️ Result: {result} (Confidence: {conf:.2f})")
