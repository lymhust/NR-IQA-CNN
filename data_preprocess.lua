require 'loadcaffe'
require 'image'
require 'cutorch'
require 'gnuplot'
require 'paths'
gnuplot.setterm('x11')

-- Helper functions
-- Load the image names
function load_imgnames()
  local namelist = {'bikes', 'buildings', 'buildings2', 'caps', 'carnivaldolls', 'cemetry', 'churchandcapitol', 'coinsinfountain', 'dancers',
    'flowersonih35', 'house', 'lighthouse', 'lighthouse2', 'manfishing', 'monarch', 'ocean', 'paintedhouse', 'parrots', 'plane',
    'rapids', 'sail1', 'sail2', 'sail3', 'sail4', 'statue', 'stream', 'studentsculpture', 'woman', 'womanhat'}
  local trainNum = 17    -- 60%
  local testNum = 6      -- 20%
  local validateNum = 6  -- 20%
  local indx = torch.randperm(#namelist)
  local trainNameList = {}
  local testNameList = {}
  local validateNameList = {}

  for i = 1, trainNum do
    local tmptable = paths.dir('./datas/LIVE_imagepatches/'..namelist[indx[i]])
    for j = 3, #tmptable do
      table.insert(trainNameList, namelist[indx[i]]..'/'..tmptable[j])
    end
  end

  for i = trainNum+1, trainNum+testNum do
    local tmptable = paths.dir('./datas/LIVE_imagepatches/'..namelist[indx[i]])
    for j = 3, #tmptable do
      table.insert(testNameList, namelist[indx[i]]..'/'..tmptable[j])
    end
  end

  for i = trainNum+testNum+1, #namelist do
    local tmptable = paths.dir('./datas/LIVE_imagepatches/'..namelist[indx[i]])
    for j = 3, #tmptable do
      table.insert(validateNameList, namelist[indx[i]]..'/'..tmptable[j])
    end
  end

  -- Create image labels
  local function createLabels(imglist)
    local imglabels = {}
    for i = 1, #imglist do
      local tmpname = imglist[i]
      local pos = 0
      local id = {}
      while (pos ~= nil) do
        pos = string.find(tmpname, '-', pos + 1)
        table.insert(id, pos)
      end
      -- Load image labels
      if id[1] ~= nil then
        table.insert(imglabels, tonumber(string.sub(tmpname, id[1]+1, id[2]-1)))
      else
        table.insert(imglabels, 85.0)
      end
    end
    return imglabels
  end

  local trainLabels = createLabels(trainNameList)
  local testLabels = createLabels(testNameList)
  local validateLabels = createLabels(validateNameList)

  return trainNameList, testNameList, validateNameList, trainLabels, testLabels, validateLabels

end

-- Convert an image from RGB to BGR format and subtracts mean
function preprocess(im, img_mean)
  -- RGB2BGR
  im = im * 255
  local imtmp = im:clone()
  imtmp[{1,{},{}}] = im[{3,{},{}}]
  imtmp[{3,{},{}}] = im[{1,{},{}}]
  -- subtract imagenet mean
  return imtmp - img_mean
end

-- Define lcc and srocc
function lcc(X, Y)
  local N = (#X)[1]
  local sig_XY = X:clone()
  sig_XY = torch.Tensor.cmul(sig_XY, X, Y):sum()
  local sig_X = X:sum()
  local sig_X_2 = X:clone()
  sig_X_2 = torch.Tensor.pow(sig_X_2, X, 2):sum()
  local sig_Y = Y:sum()
  local sig_Y_2 = Y:clone()
  sig_Y_2 = torch.Tensor.pow(sig_Y_2, Y, 2):sum()
  local tm1 = sig_XY - (sig_X * sig_Y) / N
  local tm2 = sig_X_2 - (sig_X * sig_X) / N
  local tm3 = sig_Y_2 - (sig_Y * sig_Y) / N
  local r = tm1 / math.sqrt(tm2 * tm3)
  return r
end

function srocc(X, Y)
  _, X = X:sort(X)
  _, Y = Y:sort(Y)
  _, X = X:sort(X)
  _, Y = Y:sort(Y)
  local r = lcc(X:double(), Y:double())
  return r
end

-----------------------------------------------------------------------------
img_mean = torch.load('./models/ilsvrc_2012_mean.t7').img_mean:transpose(3,1)
img_mean = image.scale(img_mean, 224, 224, 'bilinear')
trainNameList, testNameList, validateNameList, trainLabels, testLabels, validateLabels = load_imgnames()

-- Normalize labels
trainLabels = torch.Tensor(trainLabels)
testLabels = torch.Tensor(testLabels)
validateLabels = torch.Tensor(validateLabels)

local tmplabel = torch.cat(trainLabels, testLabels)
tmplabel = tmplabel:cat(validateLabels)
trainLabels = trainLabels - tmplabel:min()
testLabels = testLabels - tmplabel:min()
validateLabels = validateLabels - tmplabel:min()
tmplabel = torch.cat(trainLabels, testLabels)
tmplabel = tmplabel:cat(validateLabels)
trainLabels = trainLabels:div(tmplabel:max())
testLabels = testLabels:div(tmplabel:max())
validateLabels = validateLabels:div(tmplabel:max())


print('Train Images: '..#trainNameList..', Train Labels: '..(#trainLabels)[1])
print('Test Images: '..#testNameList..', Test Labels: '..(#testLabels)[1])
print('Validate Images: '..#validateNameList..', Validate Labels: '..(#validateLabels)[1])
  
-- Test
print(trainNameList[120])
print(trainLabels[120])




