--------------------------------------------------------------------------------
-- Demo for generic teradeep detector
-- E. Culurciello, October 2014
-- with help from A. Canziani, JH Jin, A. Dundar, B. Martini
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'pl'
local ok,cunn = pcall(require, 'fbcunn')
if not ok then
    ok,cunn = pcall(require,'cunn')
    if ok then
        print("warning: fbcunn not found. Falling back to cunn") 
        LookupTable = nn.LookupTable
    else
        print("Could not find cunn or fbcunn. Either is required")
        os.exit()
    end
else
    deviceParams = cutorch.getDeviceProperties(1)
    cudaComputeCapability = deviceopt.major + deviceopt.minor/10
    LookupTable = nn.LookupTable
end
require('nngraph')
require('base')
local ptb = require('data')

-- Local definitions -----------------------------------------------------------
local pf = function(...) print(string.format(...)) end
local Cr = sys.COLORS.red
local Cb = sys.COLORS.blue
local Cg = sys.COLORS.green
local Cn = sys.COLORS.none
local THIS = sys.COLORS.blue .. 'THIS' .. Cn

-- Title definition -----------------------------------------------------------
title = [[RNN test for CamFind data]]


-- Options ---------------------------------------------------------------------
opt = lapp(title .. [[
--nt            (default 8)     Number of threads for multiprocessing
--batch_size    (default 20)    processing batch size
--seq_length    (default 5)     max words in sequence for caption
--layers        (default 2)     number of layers of RNN
--decay         (default 1.15)  decay parameter for LSTM
--rnn_size      (default 200)   RNN number of neurons / LSTM cells
--dropout       (default 0)     Dropout parameter
--init_weight   (default 0.04)  initial weights RNN
--lr            (default 1)     Learnin rate
--vocab_size    (default 10000) Vocabolary size for words dictionary
--max_epoch     (default 14)    lower learning rate every max_epoch epochs
--max_max_epoch (default 50)    Max epochs in training loop
--max_grad_norm (default 10)    Max gradient normalization
--trainsize     (default 100)   train set size
--testsize      (default 100)   test set size
--valsize       (default 100)   validation set size
]])

pf(Cb..title..Cn)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.nt)
print('Number of threads used:', torch.getnumthreads())


-- functions:

local function transfer_data(x)
  return x:cuda()
end

local state_train, state_valid, state_test
local model = {}
local paramx, paramdx

local function lstm(i, prev_c, prev_h)
  local function new_input_sum()
    local i2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
    local h2h            = nn.Linear(opt.rnn_size, opt.rnn_size)
    return nn.CAddTable()({i2h(i), h2h(prev_h)})
  end
  local in_gate          = nn.Sigmoid()(new_input_sum())
  local forget_gate      = nn.Sigmoid()(new_input_sum())
  local in_gate2         = nn.Tanh()(new_input_sum())
  local next_c           = nn.CAddTable()({
    nn.CMulTable()({forget_gate, prev_c}),
    nn.CMulTable()({in_gate,     in_gate2})
  })
  local out_gate         = nn.Sigmoid()(new_input_sum())
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
  return next_c, next_h
end


local function create_network()
  local x                = nn.Identity()()
  local y                = nn.Identity()()
  local prev_s           = nn.Identity()()
  local i                = {[0] = LookupTable(opt.vocab_size,
                                                    opt.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2 * opt.layers)}
  for layer_idx = 1, opt.layers do
    local prev_c         = split[2 * layer_idx - 1]
    local prev_h         = split[2 * layer_idx]
    local dropped        = nn.Dropout(opt.dropout)(i[layer_idx - 1])
    local next_c, next_h = lstm(dropped, prev_c, prev_h)
    table.insert(next_s, next_c)
    table.insert(next_s, next_h)
    i[layer_idx] = next_h
  end
  local h2y              = nn.Linear(opt.rnn_size, opt.vocab_size)
  local dropped          = nn.Dropout(opt.dropout)(i[opt.layers])
  local pred             = nn.LogSoftMax()(h2y(dropped))
  local err              = nn.ClassNLLCriterion()({pred, y})
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  module:getParameters():uniform(-opt.init_weight, opt.init_weight)
  return transfer_data(module)
end


local function setup()
  print("Creating a RNN LSTM network.")
  local core_network = create_network()
  paramx, paramdx = core_network:getParameters()
  model.s = {}
  model.ds = {}
  model.start_s = {}
  for j = 0, opt.seq_length do
    model.s[j] = {}
    for d = 1, 2 * opt.layers do
      model.s[j][d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    end
  end
  for d = 1, 2 * opt.layers do
    model.start_s[d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
    model.ds[d] = transfer_data(torch.zeros(opt.batch_size, opt.rnn_size))
  end
  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, opt.seq_length)
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(opt.seq_length))
end


local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 * opt.layers do
      model.start_s[d]:zero()
    end
  end
end


local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end


local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + opt.seq_length > state.data:size(1) then
    reset_state(state)
  end
  for i = 1, opt.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end
  g_replace_table(model.start_s, model.s[opt.seq_length])
  return model.err:mean()
end


local function bp(state)
  paramdx:zero()
  reset_ds()
  for i = opt.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end
  state.pos = state.pos + opt.seq_length
  model.norm_dw = paramdx:norm()
  if model.norm_dw > opt.max_grad_norm then
    local shrink_factor = opt.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end
  paramx:add(paramdx:mul(-opt.lr))
end


local function run_valid()
  reset_state(state_valid)
  g_disable_dropout(model.rnns)
  local len = (state_valid.data:size(1) - 1) / (opt.seq_length)
  local perp = 0
  for i = 1, len do
    perp = perp + fp(state_valid)
  end
  print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
  g_enable_dropout(model.rnns)
end


local function run_test()
  reset_state(state_test)
  g_disable_dropout(model.rnns)
  local perp = 0
  local len = state_test.data:size(1)
  g_replace_table(model.s[0], model.start_s)
  for i = 1, (len - 1) do
    local x = state_test.data[i]
    local y = state_test.data[i + 1]
    perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
    perp = perp + perp_tmp[1]
    g_replace_table(model.s[0], model.s[1])
  end
  print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
  g_enable_dropout(model.rnns)
end


local function main()
  g_init_gpu(arg)

  -- load dataset:
  local dstrain, dstest,dsval = ptb.get_dataset(opt.batch_size, opt.trainsize, opt.testsize, opt.valsize)
  state_train = {data=transfer_data(dstrain)}
  state_valid = {data=transfer_data(dsval)}
  state_test =  {data=transfer_data(dstest)}
  
  print("Network parameters:")
  print(params)
  local states = {state_train, state_valid, state_test}
  for _, state in pairs(states) do
    reset_state(state)
  end
  
  setup()
  
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  
  print("Starting training.")
  
  local words_per_step = opt.seq_length * opt.batch_size
  local epoch_size = torch.floor(state_train.data:size(1) / opt.seq_length)
  local perps
  
  while epoch < opt.max_max_epoch do
    local perp = fp(state_train)
    
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_train)
    total_cases = total_cases + opt.seq_length * opt.batch_size
    epoch = step / epoch_size
    
    if step % torch.round(epoch_size / 10) == 10 then
      local wps = torch.floor(total_cases / torch.toc(start_time))
      local since_beginning = g_d(torch.toc(beginning_time) / 60)
      print('epoch = ' .. g_f3(epoch) ..
            ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
            ', wps = ' .. wps ..
            ', dw:norm() = ' .. g_f3(model.norm_dw) ..
            ', lr = ' ..  g_f3(opt.lr) ..
            ', since beginning = ' .. since_beginning .. ' mins.')
    end

    if step % epoch_size == 0 then
      run_valid()
      if epoch > opt.max_epoch then
          opt.lr = opt.lr / opt.decay
      end
    end
    
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end

  end
  run_test()
  print("Training is over.")
end

main()
