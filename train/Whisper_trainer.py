import whisper
import torch
import torchaudio


device = torch.device('cuda:0')
model = whisper.load_model('/data1/yumingdong/pretrain_model/openai-whisper-large-v3/model.pt').to(device)
inputs = torch.randn(1, 128, 3000).to(device)
text = torch.full((1, 80), 7).to(device)
output = model(inputs, text)
print(output.shape)
predicted_id = torch.argmax(output, dim=-1)
print(predicted_id)
print(model.decode)
