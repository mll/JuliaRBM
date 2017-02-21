# Restricted Boltzmann machines with 0..1 input. 
# Created by Marek Lipert, 2014


include("rbm.jl")
include("png.jl")


function sameVectors(v1,v2)
  @assert size(v1,1) == size(v2,1)
  sum = 0.0
  for i=1:size(v1,1)
    sum = sum + abs(v1[i]-v2[i])
  end
  if sum >= 5.0e-6
   @printf("--> %.8f",sum)
  end
  sum < 5.0e-6 
end

function extractSample(sampleNo, numberOfLayers)
  nextFloats = nothing

  for layerNo = 1:numberOfLayers
    file = open("learn-"*string(layerNo)*".dat","r")
    numbers = split(readline(file)," ")
    
    allSamples = int(numbers[1])
    visibleSize = int(numbers[2])
    classes = int(numbers[3])
    
    floats = nothing
    teacher = nothing
    
    @assert sampleNo <= allSamples
    for i = 1:sampleNo
      floats, teacher = readSample(file,visibleSize,classes)
    end
    
    if !isa(nextFloats,Nothing)    
      @assert sameVectors(nextFloats,floats)
    end
    
    brain = readBrain(layerNo)
    
    @assert brain.visibleNo == visibleSize
    
    nextFloats,pre = propUp(brain,map(float32,floats))

    seq = Array(Float32,1,visibleSize)
    seq[1,:] = floats
    

    
    tileRasterImages("sample-"*string(sampleNo)*"-"*string(layerNo)*".png",seq,computeRectangleFromNumber(visibleSize),(1,1),(1,1))
    close(file)
  end
end


extractSample(8,5)