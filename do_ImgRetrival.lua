require 'loadcaffe'
require 'image'
require 'cutorch'
require 'hdf5'
require 'gnuplot'
gnuplot.setterm('x11')

function load_synset()
  local file = io.open './models/synset_words.txt'
  local list = {}
  while true do
    local line = file:read()
    if not line then break end
    table.insert(list, string.sub(line,11))
  end
  return list
end

-- Converts an image from RGB to BGR format and subtracts mean
function preprocess(im, img_mean)
  -- rescale the image
  local im3 = image.scale(im,224,224,'bilinear')
  -- RGB2BGR
  local im4 = im3:clone()
  im4[{1,{},{}}] = im3[{3,{},{}}]
  im4[{3,{},{}}] = im3[{1,{},{}}]

  -- subtract imagenet mean
  return im4 - image.scale(img_mean, 224, 224, 'bilinear')
end


subpath = './models/'
-- Setting up networks and downloading stuff if needed
proto_name = subpath .. 'nin_imagenet_deploy.prototxt'
model_name = subpath .. 'nin_imagenet.caffemodel'
img_mean_name = subpath .. 'ilsvrc_2012_mean.t7'
image_name = subpath .. 'Goldfish3.jpg'

print '==> Loading network'
-- Using network in network http://openreview.net/document/9b05a3bb-3a5e-49cb-91f7-0f482af65aea
net = loadcaffe.load(proto_name, model_name, 'cudnn')
for i = 1, 1 do
  net.modules[#net.modules] = nil -- remove the top softmax
end
print(net)

-- as we want to classify, let's disable dropouts by enabling evaluation mode
net:evaluate()

print '==> Loading synsets'
synset_words = load_synset()

print '==> Loading imagenet mean'
img_mean = torch.load(img_mean_name).img_mean:transpose(3,1)

print '==> Loading test images'
local myFile = hdf5.open('/media/lymo/软件/image_net.h5', 'r')
local testImages = myFile:read('/DS1'):all()
myFile:close()
testImages = testImages:transpose(3, 4)

local FEA = torch.Tensor(testImages:size(1), 1000)

--local kernel = image.gaussian(35)

for t = 1, testImages:size(1) do
  local im = testImages[{ t,{},{},{} }]
  local I = preprocess(im, img_mean)
-- local im_y = image.rgb2y(I)
--  I[{1,{},{}}] = im_y
--  I[{2,{},{}}] = im_y
--  I[{3,{},{}}] = im_y

--  I = image.convolve(I, kernel)
--  I = image.scale(I, 16, 16, 'bilinear')
--  I = image.scale(I, 224, 224, 'bilinear')
--  image.display(I)

  prob = net:forward(I:cuda()):view(-1):float()
  print(t)
  local fea = torch.Tensor(1000)
  FEA[{ t,{} }] = prob

    _, cls = prob:sort(true)
    for i = 1, 2 do
      print('predicted class '..tostring(i)..': ', synset_words[cls[i] ])
    end

end

-- save features
myFile = hdf5.open('./data/features_net.h5', 'w')
myFile:write('/DS1', FEA)
myFile:close()
print('Finish')



















  -- show multiConvNet weights
  --  local d1 = image.toDisplayTensor{input=model:get(1):parameters()[1]:clone(),
  --    padding=2,
  --    nrow=math.floor(math.sqrt(64)),
  --    symmetric=true,
  --  }
  --  print '==> visualizing MultiConvNet filters'
  --  print('Layer 1 filters:')
  --  image.display{image=d1, legend='Layer 1 filters'}
  --  local myFile = hdf5.open(folderadd..'/multiConvNet_w1.h5', 'w')
  --  myFile:write('/DS1', d1)
  --  myFile:close()


    -- display and save CNN output
--    local imgnum = (#pred)[1]
--    local displayedImg = pred[{ {1,imgnum},{},{} }]
--    displayedImg = image.toDisplayTensor{input=displayedImg,
--      padding=2,
--      nrow=math.floor(math.sqrt(imgnum)),
--      symmetric=true,
--    }
    --print(#displayedImg)
--    d = image.display{image=displayedImg, legend='Convolution Output', win = d, zoom = 2}
--    CNNOutputAll[{ t,{},{} }] = displayedImg
    --    myFile = hdf5.open(folderadd..'/CNNOutput_ImgFP.h5', 'w')
    --    myFile:write('/DS1', tmpimg)
    --    myFile:close()

