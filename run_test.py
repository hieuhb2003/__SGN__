import argparse
import torch
import utils
from loader.data_loader import Corpus
from utils import build_loaders, build_model
from config import Config as C, MSVDLoaderConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--ckpt_fpath", type=str)
    return parser.parse_args()
def get_caption_for_video(video_id, model, data_loader, vocab):
    # Lặp qua dữ liệu và tìm caption cho video_id
    for batch in data_loader:
        for pos, neg in batch:
            pos_vid, _, pos_captions = pos
            if video_id in pos_vid:
                # Lấy chỉ mục của video trong batch
                idx = pos_vid.index(video_id)
                # Lấy caption tương ứng
                caption_idx = torch.argmax(pos_captions[:, idx, :], dim=0)
                caption_words = [vocab.idx2word[i.item()] for i in caption_idx]
                caption = ' '.join(caption_words)
                return caption
    return None
def run(inputVideoID, ckpt_fpath):
    C.loader = MSVDLoaderConfig
    train_iter, val_iter, test_iter, vocab = build_loaders(C)
    model = build_model(C, vocab)
    model.load_state_dict(torch.load(ckpt_fpath))
    model.cuda()
    print(get_caption_for_video(inputVideoID, model, test_iter, vocab))
    print(utils.generate_caption(model,test_iter,vocab, inputVideoID))

# def getCapTionInTest(inputVideoID, ckpt_fpath):
#     C = MSVDLoaderConfig()  # Tạo một đối tượng cấu hình
#     corpus = Corpus(C)  # Tạo đối tượng Corpus
#     # Tải model và các DataLoader từ đối tượng Corpus
#     train_loader = corpus.train_data_loader
#     val_loader = corpus.val_data_loader
#     test_loader = corpus.test_data_loader
#     vocab = corpus.vocab
#
#     train_iter, val_iter, test_iter, vocab2 = build_loaders(C)
#     if(vocab == vocab2): print(1)
#     # Xây dựng model
#     model = build_model(C, vocab)
#     # Load trạng thái của model từ file checkpoint
#     model.load_state_dict(torch.load(ckpt_fpath))
#     model.cuda()  # Chuyển model sang GPU nếu có
#     # Tìm caption cho video đầu vào
#     caption = get_caption_for_video(inputVideoID, model, test_loader, vocab)
#     print("Caption for video {}: {}".format(inputVideoID, caption))
#     print(test_loader)
#     print("________________")
#     print(test_iter)


if __name__ == '__main__':
    args = parse_args()
    run(args.input, args.ckpt_fpath)