# Restricted Boltzmann machines with 0..1 input. 
# Created by Marek Lipert, 2014

#include("headers.jl")
#include("png.jl")

function argmax(y)
   max = -Inf32
   aMax = -1
   for i=1:size(y)[1]
     if y[i] > max
      aMax = i
      max = y[i]
     end
   end
   return aMax
end

function trace(a)
  @assert length(size(a)) == 2 "Only square matrices can have trace"
  @assert size(a,1) ==  size(a,2) "You can compute trace for square matrices only"
  t = 0.0
  for i = 1:size(a,1)
    t = t + a[i,i]
  end
  t
end

probabilities(layer,x) = softmax(layer.W * x .+ layer.hiddenBias)

classify(layer,x) = argmax(probabilities(layer,x)) - 1 

loss(layer,x,y) = trace((log(probabilities(layer,x))' * y)) / size(x,2)

dW(layer,x,y) = ((y - probabilities(layer,x)) * x') / size(x,2)   

dH(layer,x,y) = mean(y - probabilities(layer,x),2)

function step(r,x,y,lr = 0.001)
 dw = dW(r,x,y)
 dh = dH(r,x,y)

 r.W = r.W + dw*lr
 r.hiddenBias = r.hiddenBias + dh * lr
end

function getCostUpdatesSoftmax(layer,lr,persistentH,steps,epoch)
  dW(layer,layer.input,layer.inputTeacher)*lr , 0,  dH(layer,layer.input,layer.inputTeacher)*lr, 0, loss(layer,layer.input,layer.inputTeacher)
end

function validateLayer(layerNumber)
  r = readBrain(layerNumber)
  
  file = open("learn-"*string(layerNumber)*".dat","r")
  numbers = split(readline(file)," ")
  
  allSamples = int(numbers[1])
  visibleSize = int(numbers[2])
  classes = int(numbers[3])
  
  @printf("Validating %d samples\n",allSamples)
  successes = 0
  batchSize = 1
  batches = integer(allSamples/batchSize)
  
  input = Array(Float32,visibleSize,batchSize)
  inputTeacher = Array(Float32,classes,batchSize)
  
   seekstart(file)
   readline(file)
   tic()
   totalCost = 0.0
   for i=1:batches
    j=1
    while j <= batchSize
     floats, teacher = readSample(file,visibleSize,classes)
     digit = classify(r,floats)
     if teacher[digit+1] > 0.5
       successes = successes + 1
     #else 
     #  println("Error: "*string((i-1)*batchSize+j)*" predicted: "*string(digit)*" actually is: "*string(argmax(teacher)-1))
     #  for i=1:10
#         println("--> "*string(i-1)*": "string(probabilities(r,floats)[i]))
#       end
     end  
     j = j + 1
    end
   end
  @printf("Error %.2f percent. \n",100*float(allSamples-successes)/allSamples)
end


function trainSoftmaxLayer(layerNumber,numberOfEpochs = 15,learningRate = 0.1, learningModulator = 0.97)
  file = open("learn-"*string(layerNumber)*".dat","r")
  numbers = split(readline(file)," ")
  batchSize = 600
  
  allSamples = int(numbers[1])
  visibleSize = int(numbers[2])
  classes = int(numbers[3])
  
  batches = integer(allSamples/batchSize)
  
  input = Array(Float32,visibleSize,batchSize)
  inputTeacher = Array(Float32,classes,batchSize)
  
  println("Starting epoch 1. Processing "*string(batches)*" batches of "*string(visibleSize)*"*"*string(batchSize)*" size.")
  layer = NeuralLayer(input,inputTeacher,visibleSize,classes) 
  fill!(layer.W, 0)
  persistentH = nothing
  
  modulatedLearningRate = learningRate
  
  for epoch = 1:numberOfEpochs
   seekstart(file)
   readline(file)
   tic()
   totalCost = 0.0
   for i=1:batches
    j=1
    while j <= batchSize
     floats, teacher = readSample(file,visibleSize,classes)
    
     for k=1:visibleSize
       input[k,j] = floats[k]
     end
     for k=1:classes
       inputTeacher[k,j] = teacher[k]
     end
     j = j + 1
    end
    
    Wu, Vu, Hu, ph, cost = getCostUpdatesSoftmax(layer,modulatedLearningRate,persistentH,15,epoch)
    persistentH = layer.persistentStart
    layer.W = layer.W + Wu
    layer.visibleBias = layer.visibleBias + Vu
    layer.hiddenBias  = layer.hiddenBias  + Hu
    totalCost = totalCost + cost/batchSize
    if i % 1 == 0
      println(string(i)*"/"*string(batches))
    end

  end  
   
  toc()
  tileRasterImages("filters"*string(layerNumber)*"-"*string(epoch)*".png",layer.W,computeRectangleFromNumber(visibleSize),computeRectangleFromNumber(classes),(1,1))
  println("Completed epoch "*string(epoch)*" with cost: "*string(totalCost/batches))
  modulatedLearningRate = modulatedLearningRate * learningModulator
 end
 close(file)
 file = open("brain-"*string(layerNumber)*".net","w")
 serialize(file,layer)
 close(file)
  
end


#trainSoftmaxLayer(5,80)
#validateLayer(5)