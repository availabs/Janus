require 'torch'
require 'nn'
require 'image'
require 'cv.highgui'
require 'cv.videoio'


require 'qt'
require 'qttorch'
require 'qtwidget'

require 'fast_neural_style.ShaveImage'
require 'fast_neural_style.TotalVariation'
require 'fast_neural_style.InstanceNormalization'
local utils = require 'fast_neural_style.utils'
local preprocess = require 'fast_neural_style.preprocess'
local cv = require 'cv'

local function main()
      local dtype, use_cudnn = utils.setup_gpu(-1,'cuda', 1==1)
      local models = {}

      local preprocess_method = nil
      checkpoint_path = 'models/instance_norm/candy.t7'

      print('loading model from ', checkpoint_path)
      local checkpoint = torch.load(checkpoint_path)
      local model = checkpoint.model
      model:evaluate()
      model:type(dtype)
      if use_cudnn then
      	 cudnn.convert(model,cudnn)
      end
      table.insert(models,model)
      local this_preprocess_method = checkpoint.opt.preprocessing or 'vgg'
      if not preprocess_method then
      	 print('got here')
	 preprocess_method = this_preprocess_method
	 print(preprocess_method)
      else
	if this_preprocess_method ~= preprocess_method then
	   error('All models must use the same preprocessing')
	end
       end

       local preprocess = preprocess[preprocess_method]
       local capture = cv.VideoCapture{device=0}
       if not capture:isOpened() then
	  print("Failed to get webcam")
	  os.exit(-1)
       end
       cv.namedWindow{winname="myvid",flags=cv.WINDOW_AUTOSIZE}
       
       while capture:grab() do
	  _,frame = capture:retrieve()
	  local w = frame:size(2)
	  local h = frame:size(1)

	  local img_pre = preprocess.preprocess(frame):type(dtype)

	  local imgs_out = {}
	  --for i, model in ipairs(models) do
	  local img_out_pre = model:forward(img_pre)
	     
	  local img_out = preprocess.deprocess(img_out_pre)[1]:float()
	  table.insert(imgs_out,img_out)
	  --end

	  cv.imshow('myvid',img_out)
	  if cv.waitKey{30} >= 0 then break end   
       	     
	 
