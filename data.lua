--------------------------------------------------------------------------------
-- RNN code for Camfind data captioning of images
-- E. Culurciello, April 2015
-- code inspiration from: https://github.com/wojzaremba/lstm
--------------------------------------------------------------------------------

local datain = {}

local stringx = require('pl.stringx')
 
datain.vocab = {} -- vocabulary object
datain.vocab.idx = 0 -- size of vocabulary
datain.vocab.map = {} -- word embeddings
datain.vocab.words = {} -- words

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size [x_inp:size(1)/batch_size] x [batch_size]
local function replicate(x_inp, batch_size)
   local s = x_inp:size(1)
   local x = torch.zeros(torch.floor(s / batch_size), batch_size)
   for i = 1, batch_size do
     local start = torch.round((i - 1) * s / batch_size) + 1
     local finish = start + x:size(1) - 1
     x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
   end
   return x
end


-- turns the data words into numbers in the same dictionary, see example:
-- data: duralast car battery <eos> silver macbook keyboard <eos> 4 earth sugar snap peas <eos> 4 earth sugar snap peas <eos>  
-- #s:   1        2   3       4     5      6       7        4     8 9     10    11   12   4     8 9     10    11   12   4
local function make_dataset(data, fname)
  local vocab = datain.vocab
  data = stringx.split(data)
  print(string.format("Loading %s, size of data = %d", fname, #data))
  local x = torch.zeros(#data)
  for i = 1, #data do
    if vocab.map[data[i]] == nil then
      vocab.idx = vocab.idx + 1
      vocab.map[data[i]] = vocab.idx
      vocab.words[vocab.idx] = data[i] 
    end
    x[i] = vocab.map[data[i]]
  end
  return x
end


local function load_data(fname, trainsize, testsize, valsize)
  -- file of caption data:
  local ff = assert(io.open(fname, "r"))
  
  local traindata = ''
  local testdata = ''
  local valdata = ''

  ff:read() -- skip 1st line of junk
  if opt.verbose then print('Captions:') end
  for i = 1, trainsize+testsize+valsize do
    local line = ff:read()
    a, caption  = stringx.splitv(line, '\t')
    caption = caption .. ' <eos> ' -- notice spaces on both side, so <eos> is like a word!!!!
    if opt.verbose then print(i, caption) end

    if i <= trainsize then 
      traindata = traindata .. caption
    elseif i > trainsize and i <= trainsize+testsize then 
      testdata = testdata .. caption
    else
      valdata = valdata .. caption
    end
  end

  ff:close()

  return make_dataset(traindata,'train'), make_dataset(testdata,'test'), make_dataset(valdata,'validation')
end


function datain.get_dataset(fname, batch_size, trainsize, testsize, valsize)
  local traindata, testdata, validdata
  traindata, testdata, validdata = load_data(fname, trainsize, testsize, valsize) -- train,test,validate sizes
  
  local xtrain = replicate(traindata, batch_size)
  if opt.verbose then
    print('Dictionary:', datain.vocab.words)
    print('training set:') print(xtrain)
  end
  local xval = replicate(validdata, batch_size)
  -- Intentionally we repeat dimensions without offseting.
  -- Pass over this batch corresponds to the fully sequential processing.
  local xtest = testdata:resize(testdata:size(1), 1):expand(testdata:size(1), batch_size)

  return xtrain,xtest,xval
end

return datain
