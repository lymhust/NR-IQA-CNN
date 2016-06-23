require 'xlua'
require 'optim'
require 'pl'
require 'cunn'
local c = require 'trepl.colorize'


opt = lapp[[
   -s,--save                  (default "logs")                    subdirectory to save logs
   -b,--batchSize             (default 1)                         batch size
   -r,--learningRate          (default 1e-4)                      learning rate
   --learningRateDecay        (default 1e-7)                      learning rate decay
   --weightDecay              (default 5e-4)                      weightDecay
   -m,--momentum              (default 0.9)                       momentum
   --epoch_step               (default 50)                        epoch step
   --model                    (default nin_imagenet_pretrain)     model name
   --max_epoch                (default 400)                       maximum number of iterations
   --type                     (default cuda)                      cuda or double
]]

print(opt)

print(c.blue '==>' ..' configuring model')
dofile('data_preprocess.lua')
model = dofile('models/'..opt.model..'.lua'):cuda()

print(model)

print('Will save at '..opt.save)
paths.mkdir(opt.save)
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))
testLogger:setNames{'% mean class accuracy (train set)', '% mean class accuracy (test set)'}
testLogger.showPlot = false

parameters,gradParameters = model:getParameters()

print(c.blue'==>' ..' setting criterion')
criterion = nn.MSECriterion():cuda()

print(c.blue'==>' ..' configuring optimizer')
optimState = {
  learningRate = opt.learningRate,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.learningRateDecay,
}


function train()

  local cost = {}
  model:training()
  epoch = epoch or 1
  -- drop learning rate every "epoch_step" epochs
  --if epoch % opt.epoch_step == 0 then optimState.learningRate = optimState.learningRate/2 end
  -- shuffle at each epoch
  shuffle = torch.randperm(#trainNameList)

  for t = 1, #trainNameList, opt.batchSize do
    -- disp progress
    xlua.progress(t, #trainNameList)

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1,#trainNameList) do
      -- load new sample
      local tpname = trainNameList[shuffle[i]]
      tpname = './datas/LIVE_imagepatches/'..tpname
      local input = preprocess(image.load(tpname), img_mean)
      local target = torch.Tensor({trainLabels[shuffle[i]]})

      if opt.type == 'double' then input = input:double()
      elseif opt.type == 'cuda' then input = input:cuda() target = target:cuda() end
      table.insert(inputs, input)
      table.insert(targets, target)
    end

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- f is the average of all criterions
      local f = 0

      -- evaluate function for complete mini batch
      for i = 1,#inputs do
        -- estimate f
        --print(inputs[i])
        local output = model:forward(inputs[i])
        -- make sigmoid output binary
        --        local output = torch.Tensor(#out):zero()
        --        output[out:ge(0.5)] = 1
        --        output[out:lt(0.5)] = 0
        local err = criterion:forward(output, targets[i])

        f = f + err
        table.insert(cost, f)

        -- estimate df/dW
        local df_do = criterion:backward(output, targets[i])
        model:backward(inputs[i], df_do)

      end

      -- normalize gradients and f(X)
      gradParameters:div(#inputs)
      f = f/#inputs

      -- return f and df/dX
      return f, gradParameters
    end

    optim.sgd(feval, parameters, optimState)

    gnuplot.plot(torch.Tensor(cost))

  end
end


function test()

  model:evaluate()

  -- Test on train, test and validate images
  local function testfunc(NameList, Labels)
    local outputs = torch.Tensor(#NameList)
    local targets = torch.Tensor(#Labels)
    for t = 1, #NameList, opt.batchSize do
      xlua.progress(t, #NameList)
      local tpname = NameList[t]
      tpname = './datas/LIVE_imagepatches/'..tpname
      local input = preprocess(image.load(tpname), img_mean)
      targets[t] = Labels[t]
      outputs[t] = model:forward(input:cuda()):double()
    end
    local lcc = lcc(targets, outputs)
    local srocc = srocc(targets, outputs)
    return lcc, srocc
  end
  
  local trainlcc, trainsrocc = testfunc(trainNameList, trainLabels)
  local testlcc, testsrocc = testfunc(testNameList, testLabels)
  local validatelcc, validatesrocc = testfunc(validateNameList, validateLabels)
  
  print('LCC train: '..trainlcc..' SROCC train: '..trainsrocc)
  print('LCC test: '..testlcc..' SROCC test: '..testsrocc)
  print('LCC validate: '..testlcc..' SROCC validate: '..testsrocc)

  return trainlcc, trainsrocc, testlcc, testsrocc, validatelcc, validatesrocc

end

local LCC_train = {}
local SROCC_train = {}
local LCC_test = {}
local SROCC_test = {}
local LCC_validate = {}
local SROCC_validate = {}
for i = 1, opt.max_epoch do
  train()
  print('Epoch '..i)
  trainlcc, trainsrocc, testlcc, testsrocc = test()
  table.insert(LCC_train, trainlcc)
  table.insert(SROCC_train, trainsrocc)
  table.insert(LCC_test, testlcc)
  table.insert(SROCC_test, testsrocc)
  table.insert(LCC_validate, validatelcc)
  table.insert(SROCC_validate, validatesrocc)
end


