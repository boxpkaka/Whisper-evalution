import torch
from transformers import MCTCTForCTC, MCTCTProcessor
from utils.count_model import count_model
from dataloader import get_dataloader
from eval.eval_whisper import save_eval


def eval_mms(model_path: str, dataset_dir: str, export_dir: str, device: torch.device):
    model = MCTCTForCTC.from_pretrained(model_path)
    processor = MCTCTProcessor.from_pretrained(model_path)

    model.config.forced_decoder_ids = None
    print('param:    ', count_model(model))
    dataloader = get_dataloader(dataset_dir, None, 1, False, type='whisper_huggingface')

    count = 1
    refs = []
    trans = []
    model.to(device)
    model.eval()
    for batch in dataloader:
        wav, sr, wav_length, ref, idx = batch
        wav = wav[0][:wav_length[0]]
        idx = idx[0]

        input_features = processor(wav, sampling_rate=16000, return_tensors='pt').input_features.to(device)
        logits = model(input_features).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]

        refs.append(f'{ref[0]} ({idx})')
        trans.append(f'{transcription} ({idx})')
        print(str(count) + '/' + str(len(dataloader)))
        print('reference: ', ref[0])
        print('mms:       ', transcription)
        count += 1

    save_eval(export_dir, refs, trans)
