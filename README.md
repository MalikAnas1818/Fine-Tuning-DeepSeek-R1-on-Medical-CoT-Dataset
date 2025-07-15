# 🧠 Fine-Tuning DeepSeek-R1 on Medical CoT Dataset

This project demonstrates how to fine-tune the `DeepSeek-R1-Distill-Llama-8B` model using the [FreedomIntelligence/medical-o1-reasoning-SFT](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT) dataset. The model is fine-tuned using Unsloth’s fast `QLoRA` implementation, optimized for Colab (T4 GPU compatible).
---
## 🚀 Features

- ✅ Fine-tunes LLaMA-based DeepSeek-R1 model on medical reasoning data
- ✅ Uses Chain-of-Thought (CoT) prompting format
- ✅ Powered by Unsloth + LoRA + Hugging Face ecosystem
- ✅ End-to-end code tested on Google Colab
- ✅ WANDB integration for training logs
---
## 🧾 Dataset

- **Name:** FreedomIntelligence/medical-o1-reasoning-SFT
- **Size Used:** First 500 training samples
- **Fields:** `Question`, `Complex_CoT`, `Response`
---
## 🛠️ Libraries Used

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install triton==2.0.0 --force-reinstall
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install peft==0.10.0 trl==0.7.9 xformers==0.0.28.post3 accelerate bitsandbytes
```
---
## 🧪 Inference Example

```python
question = """A 59-year-old man presents with a fever..."""
FastLanguageModel.for_inference(model)
inputs = tokenizer([prompt_style.format(question, "")], return_tensors="pt").to("cuda")
outputs = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=1200)
print(tokenizer.decode(outputs[0]))
```

---
## 🧠 Fine-Tuning Flow

1. **Model:** `dee/DeepSeek-R1-Distill-Llama-8B` loaded with 4-bit QLoRA
2. **Prompt:** Custom CoT-based prompt format with `<think>` tags
3. **Trainer:** `SFTTrainer` from TRL used for supervised fine-tuning
4. **LoRA:** Injected on key attention/feed-forward layers
5. **WANDB:** Tracks loss, learning rate, etc.

---
## 🔐 Setup

- Add your Hugging Face token via:
```python
from google.colab import userdata
hf_token = userdata.get("HF_TOKEN")
login(hf_token)
```
- Add your WandB token:
```python
wandb.login(key=userdata.get("WANDB_API_TOKEN"))
```
---
## 📊 Training Configuration

- **Batch Size:** 2 (with gradient accumulation)
- **Epochs:** 1
- **Max Steps:** 60
- **Optimizer:** `adamw_8bit`
- **Scheduler:** `linear`
- **Output Dir:** `outputs`
---
## ✅ Results

The fine-tuned model demonstrates improved medical reasoning and diagnostic accuracy on complex questions using Chain-of-Thought prompting.

---
## 📎 Credits

- [Unsloth](https://github.com/unslothai/unsloth)
- [DeepSeek](https://huggingface.co/dee/DeepSeek-R1-Distill-Llama-8B)
- [FreedomIntelligence Dataset](https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT)

---
## 🤝 Contribution

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---
## 🧠 Maintainer
Made with ❤️ by **Muhammad Anis Faseel**  
Connect on [LinkedIn]([https://www.linkedin.com](https://www.linkedin.com/in/muhammad-anis-09619a353/))
