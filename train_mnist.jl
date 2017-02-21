# Restricted Boltzmann machines with 0..1 input. 
# Created by Marek Lipert, 2014

include("rbm.jl")
include("softmax.jl")

function trainDBN(layers,epochs)
  for i = 1:length(layers)
      if typeof(readBrain(i)) == Nothing
         trainRbmLayer(0,i,layers[i],epochs,20,0.1)
      end
      println("Preparing data for layer "*string(i+1))
      prepareNextLayer(i)
  end
  trainSoftmaxLayer(length(layers)+1,5000,0.01,0.999)
end

@printf("Layer 1:\n\n")
trainRbmLayer(1)
@printf("Full training")
#trainDBN((500,500,2000,30),50)
