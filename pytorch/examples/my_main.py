import torch
import torch.nn as nn
from torch.autograd import Variable
import data
import model
import math

# ============ Initialize variable ==========================
MODEL_DIR = 'best.model'
DATA_DIR = './data/penn/'
BATCH_SIZE = 4
# sequence length
BPTT = 35
criterion = nn.CrossEntropyLoss()
# learning rate
lr = 20
best_val_loss = None
iteration = 10

# ============ Prepare data ================================
corpus = data.Corpus(DATA_DIR)
# A manual data initializer for language model
# corpus.data = tokenize('./data/penn/train.txt')
# corpus.data = tokenize('./data/penn/valid.txt')
# corpus.data = tokenize('./data/penn/test.txt')
# Tokenize: replace each word by its unique id
# eval_batch_size  = 10
val_data = batchify(corpus,val, 10)
test_data = batchify(corpus.test, 10)
ntokens = len(corpus.dictionary)

# ============ Initialize CPU and GPU ======================
torch.manual_seed(args.seed)
# Skip cuda
# torch.cuda.manual_seed(args.seed)
# model.cuda()

# ============ Initialize model ============================
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)
# model.RNNModel(model, ntokens, emsize, nhid, nlayers, dropout, tied)
# model   : type of recurrent net, ex. "LSTM", "RNN_TANH"
# emsize  : size of word embeddings, ex. 212, 432
# nhid    : number of hidden units per layer, ex. 200
# nlayers : number of layers, ex. 2
# dropout : dropout applied to layers, ex. 0, 1
# tied    : tie the owrd embedding and softmax weight, ex. 1111

# ============ Methods in Epoch ============================
def get_batch(source, i, evaluation=False):
  seq_len = min(args.bptt, len(source) - 1 - i)
  data = Variable(source[i:i+seq_len], volatile=evaluation)
  target = Variable(source[i+1:i+1+seq_len].view(-1))
  # 23 32 43 54 65 76 87 9
  # <---  data  --->
  #       <--- target --->
  return data, target

def repackage_hidden(h):
  # Wraps hidden states in new Variables, to detach them from their history.
  # [[1,3,4],[5,[32,42]]] -> [1,3,4,5,32,42]
  if type(h) == Variable:
    return Variable(h.data)
  else:
    return tuple(repackage_hidden(v) for v in h)

# ============ Epoch =======================================
def train():
  model.train()
  total_loss = 0
  start_time = time.time()
  ntokens = len(corpus.dictionary)
  hidden = model.init_hidden(BATCH_SIZE)
  for batch, i in enumerate(range(0, train_data.size(0) - 1, BPTT)):
    data, targets = get_batch(train_data, i)
    hidden = repackage_hidden(hidden)
    model.zero_grad()
    output, hidden = model(data, hidden)
    loss = criterion(output.view(-1, ntokens), targets)
    # Back propagation?
    loss.backward()
    torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
    for p in model.parameters():
      p.data.add_(-lr, p.grad.data)
    total_loss += loss.data
    if batch % args.log_interval == 0 and batch > 0:
      cur_loss = total_loss[0] / args.log_interval
      elapsed = time.time() - start_time
      print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | loss {:5.2f} | ppl {:8.2f}'.format(epoch, batch, len(train_data) // args.bptt, lr, elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()

# ============ Learn =============
try:
  for epoch in iteration:
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)))
    print('-' * 89)
    # save model if best
    if not best_val_loss or val_loss < best_val_loss:
      with open(MODEL_DIR, 'wb') as f:
        torch.save(model, f)
      best_val_loss = val_loss
    else:
    # update learning rate
      lr /= 4.0
except KeyboardInterrupt:
  print('-' * 89)
  print('Exiting from training early')

# ============ Test ===============
# Load the best saved model.
with open(MODEL_DIR, 'rb') as f:
  model = torch.load(f)
# Run on test data.
test_loss = evaluate(test_data)
print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)

# Ideal three functions
# train()
# loss()
# predict(data)
