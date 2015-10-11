# This file is a python adaptation for use with lunatic-python

# This file trains a character-level multi-layer RNN on text data

# Code is based on implementation in
# https://github.com/oxford-cs-ml-2015/practical6
# but modified to have multi-layer support, GPU support, as well as
# many other common model/optimization bells and whistles.
# The practical6 code is in turn based on
# https://github.com/wojciechz/learning_to_execute
# which is turn based on other stuff in Torch, etc... (long lineage)

# Lunatic-python internals
# Will be removable in future versions
import ctypes
luajitlib = ctypes.CDLL("libluajit.so", mode=ctypes.RTLD_GLOBAL)
follylib = ctypes.CDLL("libfolly.so", mode=ctypes.RTLD_GLOBAL)
THlib = ctypes.CDLL("libTH.so", mode=ctypes.RTLD_GLOBAL)
luaTlib = ctypes.CDLL("libluaT.so", mode=ctypes.RTLD_GLOBAL)
# end of Lunatic-python internals

import numpy as np
import lua
import gc

lua.require('torch')
lua.execute('torch.setdefaulttensortype("torch.FloatTensor")')
lua.require('nn')
lua.require('nngraph')
lua.require('optim')
lua.require('lfs')
lua.require('pl')

lua.require('util.OneHot')
lua.require('util.misc')
CharSplitLMMinibatchLoader = lua.require('util.CharSplitLMMinibatchLoader')
lua.execute('model_utils = require "util.model_utils"')
LSTM = lua.require('model.LSTM')
GRU = lua.require('model.GRU')
RNN = lua.require('model.RNN')

# Make some lua globals available for python
lua_globals = lua.globals()
torch = lua_globals.torch
path = lua_globals.path
nn = lua_globals.nn
optim = lua_globals.optim
math = lua_globals.math
table = lua_globals.table
string = lua_globals.string
model_utils = lua_globals.model_utils
collectgarbage = lua_globals.collectgarbage
def new_table():
    return lua.eval('{}')

# Parse python args
from sys import argv
arg = {}
for i in range(0,len(argv)):
    arg[i] = argv[i]

cmd = lua.eval("torch.CmdLine()")
cmd.text(cmd)
cmd.text(cmd,'Train a character-level language model')
cmd.text(cmd)
cmd.text(cmd,'Options')
# data
cmd.option(cmd,'-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
# model params
cmd.option(cmd,'-rnn_size', 128, 'size of LSTM internal state')
cmd.option(cmd,'-num_layers', 2, 'number of layers in the LSTM')
cmd.option(cmd,'-model', 'lstm', 'lstm,gru or rnn')
# optimization
cmd.option(cmd,'-learning_rate',2e-3,'learning rate')
cmd.option(cmd,'-learning_rate_decay',0.97,'learning rate decay')
cmd.option(cmd,'-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd.option(cmd,'-decay_rate',0.95,'decay rate for rmsprop')
cmd.option(cmd,'-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd.option(cmd,'-seq_length',50,'number of timesteps to unroll for')
cmd.option(cmd,'-batch_size',50,'number of sequences to train on in parallel')
cmd.option(cmd,'-max_epochs',50,'number of full passes through the training data')
cmd.option(cmd,'-grad_clip',5,'clip gradients at this value')
cmd.option(cmd,'-train_frac',0.95,'fraction of data that goes into train set')
cmd.option(cmd,'-val_frac',0.05,'fraction of data that goes into validation set')
            # test_frac will be computed as (1 - train_frac - val_frac)
cmd.option(cmd,'-init_from', '', 'initialize network parameters from checkpoint at this path')
# bookkeeping
cmd.option(cmd,'-seed',123,'torch manual random number generator seed')
cmd.option(cmd,'-print_every',1,'how many steps/minibatches between printing out the loss')
cmd.option(cmd,'-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd.option(cmd,'-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd.option(cmd,'-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
# GPU/CPU
cmd.option(cmd,'-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd.option(cmd,'-opencl',0,'use OpenCL (instead of CUDA)')
cmd.text(cmd)

# parse input params
opt = cmd.parse(cmd, lua.toTable(arg))
lua_globals.opt = opt # Make command line options available in lua side
torch.manualSeed(opt.seed)
# train / val / test split for data, in fractions
test_frac = max(0, 1 - (opt.train_frac + opt.val_frac))
split_sizes = new_table() # Can change when proper list <-> table is in
split_sizes[1] = opt.train_frac
split_sizes[2] = opt.val_frac
split_sizes[3] = test_frac

# initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0:
    try:
        lua.require('cunn')
        lua.require('cutorch')
        print('using CUDA on GPU ' + opt.gpuid + '...')
        cutorch = lua_globals.cutorch
        cutorch.setDevice(opt.gpuid + 1) # note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    except:
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 # overwrite user setting


# initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1:
    try:
        lua.require('clnn')
        lua.require('cltorch')
        print('using OpenCL on GPU ' + opt.gpuid + '...')
        cltorch = lua_globals.cltorch
        cltorch.setDevice(opt.gpuid + 1) # note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    except:
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 # overwrite user setting

# create the data loader class
loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
lua_globals.loader = loader # add the loader to the lua global table
vocab_size = loader.vocab_size  # the number of distinct characters
vocab = loader.vocab_mapping
print('vocab size: ' + str(vocab_size))
# make sure output directory exists
if not path.exists(opt.checkpoint_dir):
    lfs.mkdir(opt.checkpoint_dir)

# define the model: prototypes for one timestep, then clone them in time
do_random_init = True
if len(opt.init_from) > 0:
    print('loading an LSTM from checkpoint ' + opt.init_from)
    checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    # overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting rnn_size=' + str(checkpoint.opt.rnn_size) + ', num_layers=' + str(checkpoint.opt.num_layers) + ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false
else:
    print('creating an ' + opt.model + ' with ' + str(opt.num_layers) + ' layers')
    protos = new_table()
    if opt.model == 'lstm':
        protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'gru':
        protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    elif opt.model == 'rnn':
        protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    protos.criterion = nn.ClassNLLCriterion()


# the initial state of the cell/hidden states
init_state = new_table()
lua_globals.init_state = init_state
lua.execute("""
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end
""")

# ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0:
    lua.eval("for k,v in pairs(protos) do v:cuda() end")
if opt.gpuid >= 0 and opt.opencl == 1:
    lua.eval("for k,v in pairs(protos) do v:cl() end")

# put the above things into one flattened parameters tensor
lua_globals.rnn = protos.rnn
lua.execute("params, grad_params = model_utils.combine_all_parameters(rnn)")

# initialization
if do_random_init:
    lua.execute("params:uniform(-0.08, 0.08)") # small numbers uniform

print('number of parameters in the model: ' + str(lua.eval("params:nElement()")))
# make a bunch of clones after flattening, as that reallocates memory
clones = new_table()
protos = lua.toDict(protos)
for name,proto in protos.iteritems():
    print('cloning ' + name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
lua_globals.clones = clones

# evaluate the loss over an entire split
# can be accessed with lg.eval_split
eval_split = lua.eval("""function(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x = x:cl()
            y = y:cl()
        end
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst] 
            loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n
    return loss
end""")

# do fwd/bwd and return loss, grad_params
init_state_global = lua_globals.clone_list(init_state)
lua_globals.init_state_global = init_state_global

lua.execute("""feval = function(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
    end
    ------------------- forward pass -------------------
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        local p, gp = clones.rnn[t]:parameters()
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end""")

# start optimization here
train_losses = new_table()
val_losses = new_table()
lua.execute("optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}")
iterations = opt.max_epochs * loader.ntrain
iterations_per_epoch = loader.ntrain
loss0 = None
print("Starting training loop")
for i in range(1, iterations):
    epoch = i / float(loader.ntrain)

    timer = torch.Timer()
    lua.execute("_, loss = optim.rmsprop(feval, params, optim_state)")
    loss = lua.eval("loss[1]")
    time = timer.time(timer).real

    train_loss = loss # the loss is inside a list, pop it
    train_losses[i] = train_loss

    # exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1:
        if epoch >= opt.learning_rate_decay_after:
            decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor # decay it
            print('decayed learning rate by a factor ' + decay_factor + ' to ' + optim_state.learningRate)

    # every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations:
        # evaluate loss on validation data
        val_loss = eval_split(2) # 2 = validation
        val_losses[i] = val_loss

        savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' + savefile)
        checkpoint = new_table()
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)

    if i % opt.print_every == 0:
        gp_norm = lua.eval("grad_params:norm() / params:norm()")
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, gp_norm, time))
   
    if i % 10 == 0:
        collectgarbage()

    # handle early stopping if things are going really bad
    if loss != loss:
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break # halt
    if not loss0:
        loss0 = loss
    if loss > loss0 * 3:
        break # halt

