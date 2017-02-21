# Restricted Boltzmann machines with 0..1 input. 
# Created by Marek Lipert, 2014


####################################################################

sigmoid(x) = 1.0./(1.0+exp(-x))

function softmax(vector) 
 if length(size(vector)) == 1
   return exp(vector)/sum(exp(vector))
 end
 retVal = exp(vector)
 for i = 1:size(vector,2)
   retVal[:,i] = retVal[:,i] / sum(exp(vector[:,i]))
 end
  retVal
end


####################################################################
 
type NeuralLayer
   W::Array{Float32}
   hiddenBias::Array{Float32}
   visibleBias::Array{Float32}
   input::Array{Float32}
   inputTeacher::Array{Float32}
   persistentStart::Array{Float32}
      
   isGaussian::Bool
   hiddenNo::Integer
   visibleNo::Integer
   batchSize::Integer
   flipBit::Integer
end

function NeuralLayer(isGaussian = 0, input = nothing, inputTeacher = nothing, nVisible = 786,  nHidden = 500, W = nothing, hBias = nothing,  vBias = nothing)
      internalW = nothing
      internalHBias = nothing
      internalVBias = nothing
      internalInput = nothing
      internalBatchSize = nothing
      
      
      srand(2)
      
      if typeof(input) != Void
        internalInput = input
        @assert size(internalInput,1) == nVisible "Wrong dimension of input vector"
        internalBatchSize = size(internalInput,2)
      else
        internalInput = Array(Float32,nVisible,20)
        internalBatchSize = 20

      end
      
      
      if typeof(inputTeacher) != Void
        internalInputTeacher = inputTeacher
        @assert internalBatchSize == size(internalInputTeacher,2) "Sizes of input and output do not match"
      else
        internalInputTeacher = Array(Float32,nHidden,internalBatchSize)
      end
      
      if typeof(W) != Void
        internalW = W
      else
        r = 8.0*sqrt(6.0/(nHidden+nVisible))
        internalW = Float32[(rand()-0.5)*r for i=1:nHidden,j=1:nVisible]      
      end
      
      if typeof(hBias) != Void
         internalHBias = hBias
      else 
         internalHBias = Array(Float32,nHidden)
         fill!(internalHBias,0.0)
      end

      if typeof(vBias) != Void
         internalVBias = vBias
      else 
         internalVBias = Array(Float32,nVisible)
         fill!(internalVBias,0.0)
      end
      
  
      NeuralLayer(internalW,internalHBias,internalVBias,internalInput,internalInputTeacher,Array(Float32,nHidden,internalBatchSize),isGaussian,nHidden,nVisible,internalBatchSize,1) 
end

#############################################################
#
# Reads brain data from file
#
#############################################################

function readBrain(layerNumber)
  fileName = "brain-"*string(layerNumber)*".net"
  if !isfile(fileName)
    return nothing
  end
  file = open(fileName,"r")
  deserialize(file)
end



function readSplittedLineWithoutSpaces(stream,rsize)
    found = false
    line = ""
    while true
      line = readline(stream)
      for char in line
        if char in ['0','1','2','3','4','5','6','7','8','9']
          found = true
          break
        end
      end
      if found
        break
      end
    end
    @assert found "No line with numbers found inside file"
    splitedRaw = split(line," ")
    floats = nothing
    splited = Array(String,size(splitedRaw,1))
     k = 1
     m = 1
     for k = 1:size(splitedRaw,1)
        if length(splitedRaw[k]) != 0 && splitedRaw[k]!="\n"
          splited[m] = splitedRaw[k]
          m = m + 1
        end
     end
     m = m - 1
     if m != rsize 
        println("ERROR: Line with improper number of arguments: "*string(size(splited)[1])*" should be: "*string(rsize))
        @assert false "Look up"
     end
     map(x -> parse(Float32, x), splited)
end



function readSample(stream,visibleSize,classNo)
     sample = readSplittedLineWithoutSpaces(stream,visibleSize)
     teacher = readSplittedLineWithoutSpaces(stream,classNo)
     return sample,teacher
end
