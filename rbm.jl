# Restricted Boltzmann machines with 0..1 input. 
# Created by Marek Lipert, 2014

include("png.jl")
include("headers.jl")

###############################################################
#
# Computes estimate of negative entropy. The higher the number, 
# the better a layer is at discerning input.
#
###############################################################

function getPseudoLikehoodCost(rbm)
  bitNo = rbm.flipBit
  
  xi = similar(rbm.input)
  for j=1:rbm.batchSize
   for i=1:rbm.visibleNo
    xi[i,j] = rbm.input[i,j] > 0.5 ? 1.0 : 0.0
   end
  end

  fe_xi = Array(Float32,rbm.batchSize)
  for i =1:rbm.batchSize
    fe_xi[i] = freeEnergy(rbm,xi[:,i])
  end
  
  xi_flip = xi
  for j=1:rbm.batchSize
   for i=1:rbm.visibleNo
    if i==bitNo
      xi_flip[i,j] = 1 - xi_flip[i,j] 
    end
   end
  end

  fe_xi_flip = Array(Float32,rbm.batchSize)
  for i =1:rbm.batchSize
    fe_xi_flip[i] = freeEnergy(rbm,xi_flip[:,i])
  end
  rbm.flipBit = (rbm.flipBit + 1) % rbm.visibleNo
  mean(rbm.visibleNo * log(sigmoid(fe_xi_flip-fe_xi))) 
end

#############################################################
# 
# Methods of Gibbs sampling
#
# We assume visible[nVisible,nExamples]
#
#############################################################

# Box-Muller (just in case we needed it for some future gaussian changes)

function generateGaussianSample(x0 = 0.0, sigma = 1.0)
   const two_pi = 6.2831853071795864769252866
   u = rand()
   v = rand()
   u = u < 1e-100 ? 1e-100 : u
   x0 + sqrt(- 2 * sigma * log(u))*cos(v*two_pi)
end



function propUp(rbm,visible)
   @assert rbm.visibleNo == size(visible,1)
   preSigmoid = Array(Float32,rbm.hiddenNo,size(visible,2))
   for i in 1:size(visible,2)
      preSigmoid[:,i] = rbm.W * visible[:,i] + rbm.hiddenBias
   end
   sigmoid(preSigmoid) 
end

function sampleHGivenV(rbm,visible)
  probability = propUp(rbm,visible)
  Float32[ rand() < probability[i,j] ? 1.0 : 0.0 for i=1:size(probability,1),j=1:size(probability,2)]
end

function propDown(rbm,hidden)
   @assert rbm.hiddenNo == size(hidden,1)
   preSigmoid = Array(Float32,rbm.visibleNo,size(hidden,2))
   for i in 1:size(hidden,2)
      preSigmoid[:,i] = rbm.W' * hidden[:,i] + rbm.visibleBias
   end
   rbm.isGaussian == 1 ? preSigmoid : sigmoid(preSigmoid) 
end


function sampleVGivenH(rbm,hidden)
  probability = propDown(rbm,hidden)
  # We could sample from a gaussian with x0 = probability and sigma = 1, but we just ignore the noise and take the mean as a reconstruction
  rbm.isGaussian ? probability : Float32[ rand() < probability[i,j] ? 1.0 : 0.0 for i=1:size(probability,1),j=1:size(probability,2)]
end

function gibbs_HVH(rbm,hiddenSample)
   sam1 = sampleVGivenH(rbm,hiddenSample)
   sam2 = sampleHGivenV(rbm,sam1)
   sam1,sam2
end

function gibbs_VHV(rbm,visibleSample)
   sam1 = sampleHGivenV(rbm,visibleSample)
   sam2 = sampleVGivenH(rbm,sam1)
   sam1,sam2
end

#############################################################
#
# This is a function we try to minimize with gradient descent
#
#############################################################

function freeEnergy(rbm,visibleSample)
  @assert size(size(visibleSample),1) == 1  "VisibleSample has to be one-dimensional for this method"
  
  exponent = rbm.W * visibleSample + rbm.hiddenBias
  vBiasTerm = (visibleSample' * rbm.visibleBias)[1]
  hidden = log1p(exp(exponent))
  for i = 1:size(exponent,1) 
    if exponent[i] > 7
       hidden[i] = exponent[i] + log1p(exp(-exponent[i]))
    end
  end
  hiddenTerm = sum(hidden)
  -hiddenTerm - vBiasTerm
end
  
#############################################################
#
# The actual Learning function
#
#############################################################

function getCostUpdates(rbm, learningRate = 0.1, persistent = nothing, k = 1,idx = 1)
  visibleStart = rbm.input
  hiddenStart = nothing

  if typeof(persistent) == Void 
    hiddenStart = sampleHGivenV(rbm,visibleStart)
  else
    hiddenStart = persistent
  end
  Wupdate = similar(rbm.W)
  Hupdate = similar(rbm.hiddenBias)
  Vupdate = similar(rbm.visibleBias)

  hiddenStop = hiddenStart
  visibleStop = visibleStart

  for i = 1:k
   visibleStop, hiddenStop = gibbs_HVH(rbm,hiddenStop)
  end
  
  stopSigmoid = propUp(rbm,visibleStop)
  startSigmoid = propUp(rbm,visibleStart)
     
  Wupdate = (-learningRate/size(stopSigmoid,2)) .* (stopSigmoid * visibleStop' - startSigmoid * visibleStart')  
  Vupdate = -learningRate .* mean(visibleStop-visibleStart,2)
  Hupdate = -learningRate .* mean(stopSigmoid-startSigmoid,2)  
   
  cost = getPseudoLikehoodCost(rbm)
  
  Wupdate,Vupdate[:,1],Hupdate[:,1],hiddenStop,cost
end
  

#############################################################
#
# This prepares data for training of next RBM layer and 
# saves it in "learn-%d.dat" file.
#
#############################################################

  

function prepareNextLayer(previousNumber)
    rbm = readBrain(previousNumber)
    
    file = open("learn-"*string(previousNumber)*".dat","r")
    outFile = open("learn-"*string(previousNumber+1)*".dat","w")
    numbers = split(readline(file)," ")
    batchSize = 20
  
    allSamples = int(numbers[1])
    visibleSize = int(numbers[2])
    classes = int(numbers[3])
    
    @printf(outFile,"%d %d %d\n\n",allSamples,rbm.hiddenNo,classes)
    
    batches = integer(allSamples/batchSize)
  
    input = Array(Float32,visibleSize,batchSize)
    teacherInput = Array(Float32,classes,batchSize)
    
    readline(file)
    for i=1:batches
     #@printf("Preparing batch %d/%d\n",i,batches)
     j=1
     while j <= batchSize
      floats, teacher = readSample(file,visibleSize,classes)
    
      for k=1:visibleSize
        input[k,j] = floats[k]
      end
      for k=1:classes
        teacherInput[k,j] = teacher[k]
      end
      j = j + 1
     end
     
     sigmoid= propUp(rbm,input)
     
     for row in 1:size(sigmoid,2)
      for column in 1:size(sigmoid,1)
        if column!=size(sigmoid,1)
          @printf(outFile,"%.8f ",sigmoid[column,row])
        else
          @printf(outFile,"%.8f",sigmoid[column,row])
        end
      end
        @printf(outFile,"\n")
        for column in 1:size(teacherInput,1)
        if column!=size(teacherInput,1)
          @printf(outFile,"%d ",int(teacherInput[column,row]))
        else
          @printf(outFile,"%d",int(teacherInput[column,row]))
        end
      end
        @printf(outFile,"\n\n")
     end
    end
    close(outFile)
    close(file)
end



#############################################################
#
# The actual training
#
#############################################################

function trainRbmLayer(layerNumber,isGaussian = 0, outNeurons = 500,numberOfEpochs = 15,batchSize = 20,learningRate = 0.1,learningModulator = 0.95)
  ###################################
  file = open("learn-"*string(layerNumber)*".dat","r")
  numbers = split(readline(file)," ")  
  allSamples = parse(Int, numbers[1])
  visibleSize = parse(Int, numbers[2])
  classes = parse(Int, numbers[3])
  
  batches = convert(Int, allSamples/batchSize)
  
  input = Array(Float32,visibleSize,batchSize)
  inputTeacher = Array(Float32,classes,batchSize)
  
  println("Starting epoch 1. Processing "*string(batches)*" batches of "*string(visibleSize)*"*"*string(batchSize)*" size.")
  
########################

  rbm = NeuralLayer(isGaussian, input,inputTeacher,visibleSize,outNeurons) 
  
  tileRasterImages("filters"*string(layerNumber)*"-0.png",rbm.W,computeRectangleFromNumber(visibleSize),computeRectangleFromNumber(outNeurons),(1,1))
  persistentH = nothing
  modulatedLearningRate = learningRate
  
  #read in all the data 
  @printf("Reading data\n")  
  floats = Array(Float32,visibleSize,allSamples)
  teacher = Array(Float32,classes,allSamples)
   seekstart(file)
   readline(file)

  for i=1:allSamples
    iFloats, iTeacher = readSample(file,visibleSize,classes)
    for j=1:visibleSize
      floats[j,i] = iFloats[j]
    end
    for j=1:classes
      teacher[j,i] = iTeacher[j]
    end
  end

  
  ############################################
  for epoch = 1:numberOfEpochs
   @printf("Starting epoch %d\n", epoch)
   tic()
   totalCost = 0.0
   absoluteSample = 1
   
   for i=1:batches
    @printf("\r%d/%d", i, batches)
    j=1
    while j <= batchSize
     
     for k=1:visibleSize
       input[k,j] = floats[k,absoluteSample]
     end
     for k=1:classes
       inputTeacher[k,j] = teacher[k,absoluteSample]
     end
     j = j + 1
     absoluteSample = absoluteSample + 1
    end
    
    Wu, Vu, Hu, ph,cost = getCostUpdates(rbm,modulatedLearningRate,persistentH,15,epoch)
    rbm.persistentStart = copy(ph)
    persistentH = rbm.persistentStart
    rbm.W = rbm.W + Wu
    rbm.visibleBias = rbm.visibleBias + Vu
    rbm.hiddenBias  = rbm.hiddenBias  + Hu
    totalCost = totalCost + cost
  end  
  modulatedLearningRate = modulatedLearningRate * learningModulator
  toc()
  tileRasterImages("filters"*string(layerNumber)*"-"*string(epoch)*".png",rbm.W,computeRectangleFromNumber(visibleSize),computeRectangleFromNumber(outNeurons),(1,1))
  println("Completed epoch "*string(epoch)*" with cost: "*string(totalCost/batches))
 end
 close(file)
  
 file = open("brain-"*string(layerNumber)*".net","w")
 serialize(file,rbm)
 close(file)

  
end

