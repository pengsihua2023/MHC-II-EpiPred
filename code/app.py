from flask import Flask, request, jsonify, render_template
import torch
from transformers import EsmTokenizer  # 使用正确的tokenizer导入
from model.custom_model import CustomEsmForSequenceClassification

app = Flask(__name__)

# Load your fine-tuned model and tokenizer
model = CustomEsmForSequenceClassification.from_pretrained('./model')
tokenizer = EsmTokenizer.from_pretrained('./model')
model.eval()
model.to('cpu')  # Ensure the model is on CPU

# Label mapping
id2label = {0: 'Non-Epitope', 1: 'Epitope'}

@app.route('/')
def analysis():  # 根路由现在映射到此函数
    return render_template('Analysis.html')

@app.route('/Instructions.html')
def instructions():
    return render_template('Instructions.html')

@app.route('/Data.html')
def data():
    return render_template('Data.html')

@app.route('/Abstract.html')
def abstract():
    return render_template('Abstract.html')

@app.route('/Source_code.html')
def source_code():
    return render_template('Source_code.html')

@app.route('/Versions.html')
def versions():
    return render_template('Versions.html')

@app.route('/Downloads.html')
def downloads():
    return render_template('Downloads.html')

def sliding_window(sequence, window_size):
    return [(sequence[i:i + window_size], i, i + window_size - 1) for i in range(len(sequence) - window_size + 1)]

def predict_sequence(sequence, window_size):
    subsequences = sliding_window(sequence, window_size)
    results = []
    for subseq, start, end in subsequences:
        inputs = tokenizer(subseq, return_tensors="pt", padding=True, truncation=True, max_length=45)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs['logits']  # 使用字典键访问logits
            probabilities = torch.softmax(logits, dim=-1)  # 使用softmax计算概率值
            prediction_id = torch.argmax(logits, dim=-1).item()
            prediction_label = id2label[prediction_id]
            prediction_prob = probabilities[0][prediction_id].item()  # 获取预测类别的概率值

            # 将概率值加入结果
            results.append({
                'sequence': subseq,
                'start': start + 1,  # Convert to 1-based index
                'end': end + 1,
                'windowSize': window_size,
                'prediction': prediction_label,
                'probability': prediction_prob  # 返回预测的概率值
            })
    return results

@app.route('/predict-epitope-II', methods=['POST'])
def predict():
    data = request.get_json()
    sequence = data.get('sequence')
    window_size = int(data.get('windowSize'))

    if not sequence or window_size < 8 or window_size > 48:
        return jsonify([]), 400

    results = predict_sequence(sequence, window_size)
    return jsonify(results)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)
