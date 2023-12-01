import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from dataloader import get_dataloader
from model import *
import tqdm


def train(model, batch_size, device, half):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CTCLoss()
    dataloader = get_dataloader('/data1/yumingdong/test/data_ls/train-clean-100', batch_size, shuffle=False)

    model.to(device)
    model.train()
    epoch = 1000
    if half:
        model.half()
    count_model(model)
    for i in range(epoch):
        for batch in tqdm(dataloader):
            wav, text, text_length = batch
            text_length = text_length.to(device)
            wav = wav.to(device)
            text = text.to(device)
            optimizer.zero_grad()

            if isinstance(model, Whisper):
                wav = wav.cpu().numpy()
                wav = model.feature_extractor(wav, sampling_rate=16000).input_features
                wav = np.array(wav)
                wav = torch.as_tensor(wav).to(device)
                if half:
                    wav = wav.half()
                logits = model(wav, text)
            else:
                if half:
                    wav = wav.half()
                logits = model(wav)          # B, T, C

            logits = logits.transpose(0, 1)  # T, B, C
            length = torch.full((logits.shape[1], ), logits.shape[0]).to(device)

            if half:
                logits = logits.float()
            loss = criterion(logits, text, length, text_length)
            loss.backward()
            optimizer.step()


def train_with_amp(model, batch_size, device, half):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CTCLoss()
    dataloader = get_dataloader('/data1/yumingdong/test/data_ls/train-clean-100', batch_size, shuffle=True)

    model.to(device)
    model.train()
    epoch = 1000
    count_model(model)
    scaler = GradScaler()
    for i in range(epoch):
        for batch in tqdm(dataloader):
            wav, text, text_length = batch
            text_length = text_length.to(device)
            wav = wav.to(device)
            text = text.to(device)
            optimizer.zero_grad()

            with autocast():
                if isinstance(model, Whisper):
                    wav = wav.cpu().numpy()
                    wav = model.feature_extractor(wav, sampling_rate=16000).input_features
                    wav = np.array(wav)
                    wav = torch.as_tensor(wav).to(device)
                    logits = model(wav, text)
                else:
                    if half:
                        wav = wav.half()
                    logits = model(wav)              # B, T, C

                logits = logits.transpose(0, 1)  # T, B, C
                length = torch.full((logits.shape[1], ), logits.shape[0]).to(device)

                logits = logits.float()
                loss = criterion(logits, text, length, text_length)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()