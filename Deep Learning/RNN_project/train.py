import argparse
import torch
import numpy as np
from tqdm import tqdm
from model import BaseModel
from dataset import TextDataset, make_data_loader

def acc(pred,label):
    pred = pred.argmax(dim=-1)
    return torch.sum(pred == label).item()

def train(args, train_loader, valid_loader, model):

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    min_loss = np.Inf
    
    for epoch in range(args.num_epochs):

        print(f"[Epoch {epoch+1} / {args.num_epochs}]")
        
        train_losses = []
        train_acc = 0.0
        train_total = 0

        valid_losses = []
        valid_acc = 0.0
        valid_total = 0

        model.train()
        for idx, (text, label) in enumerate(tqdm(train_loader)):

            # input_lengths = torch.LongTensor([torch.max(input_seq2idx[i, :].data.nonzero())+1 for i in range(input_seq2idx.size(0))])
            # input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
            # text = text[sorted_idx]
            # label = label[sorted_idx]

            text = text.to(args.device)
            label = label.to(args.device)            
            optimizer.zero_grad()

            output, _ = model(text)
            
            label = label.squeeze()
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_total += label.size(0)
            train_acc += acc(output, label)

        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = train_acc/train_total

        model.eval()
        with torch.no_grad():
            for idx, (text, label) in enumerate(tqdm(valid_loader)):
                # input_lengths = torch.LongTensor([torch.max(input_seq2idx[i, :].data.nonzero())+1 for i in range(input_seq2idx.size(0))])
                # input_lengths, sorted_idx = input_lengths.sort(0, descending=True)
                # print(sorted_idx)
                # text = text[sorted_idx]
                # label = label[sorted_idx]

                text = text.to(args.device)
                label = label.to(args.device)            

                output, _ = model(text)
                
                label = label.squeeze()
                loss = criterion(output, label)

                valid_losses.append(loss.item())
                valid_total += label.size(0)
                valid_acc += acc(output, label)

            epoch_valid_loss = np.mean(valid_losses)
            epoch_valid_acc = valid_acc/valid_total

        print(f'Epoch {epoch+1}') 
        print(f'train_loss : {epoch_train_loss}')
        print('train_accuracy : {:.3f}'.format(epoch_train_acc*100))
        print(f'valid_loss : {epoch_valid_loss}')
        print('valid_accuracy : {:.3f}'.format(epoch_valid_acc*100))

        # Save Model (train -> valid로 변경)
        if epoch_valid_loss < min_loss:
            torch.save(model.state_dict(), 'model.pt')
            print('Valid loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(min_loss, epoch_valid_loss))
            min_loss = epoch_valid_loss
        print()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='2022 DL Term Project #2')
    parser.add_argument('--data_dir', type=str, default='./Data')
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training (default: 64)")
    parser.add_argument('--vocab_size', type=int, default=30000, help="maximum vocab size")
    parser.add_argument('--batch_first', action='store_true', help="If true, then the model returns the batch first")
    parser.add_argument('--learning_rate', type=float, default=0.005, help="Learning rate (default: 0.001)")
    parser.add_argument('--num_epochs', type=int, default=8, help="Number of epochs to train for (default: 5)")
    
    args = parser.parse_args()

    # Model hyperparameters
    input_size = args.vocab_size
    output_size = 4     # num of classes
    embedding_dim = 300 # embedding dimension
    hidden_dim = 128  # hidden size of LSTM
    num_layers = 3

    # Make Train Loader
    train_dataset = TextDataset(args.data_dir, 'train', args.vocab_size)
    args.pad_idx = train_dataset.sentences_vocab.wtoi['<PAD>']
    train_loader = make_data_loader(train_dataset, args.batch_size, args.batch_first, shuffle=True)

    # Make Valid Loader
    valid_dataset = TextDataset(args.data_dir, 'valid', args.vocab_size)
    args.pad_idx = valid_dataset.sentences_vocab.wtoi['<PAD>']
    valid_loader = make_data_loader(valid_dataset, args.batch_size, args.batch_first)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    print("device : ", device)

    # instantiate model
    model = BaseModel(input_size, output_size, embedding_dim, hidden_dim, num_layers, batch_first=args.batch_first)
    model = model.to(device)

    # Training The Model
    train(args, train_loader, valid_loader, model)