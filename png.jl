# Restricted Boltzmann machines with 0..1 input. 
# Created by Marek Lipert, 2014

#############################################################
#
# Drawing methods
#
#############################################################

function encodeGrayscalePng(filename::String, data::Vector{UInt8},width::Integer,height::Integer)
  local retVal::Int32 
  @assert width*height == size(data)[1] "Incorrect data size"
  retVal = ccall( (:lodepng_encode_file,"./lodepng"), Int32, (Ptr{UInt8},Ptr{UInt8},UInt32,UInt32,UInt32,UInt32), string(filename), data,convert(UInt32,width),convert(UInt32,height),0,8) 
  @assert retVal == 0 "Error from lodepng: "*string(retVal)
end
  
function scaleToByteInterval(ndar,eps = 1e-8)
  nd2 = ndar - minimum(ndar)
  nd2 = nd2 * 1.0/(maximum(nd2)+eps)
  nd2 = nd2 * 255.0
end

function tileRasterImages(name,array,imageShape,tileShape,tileSpacing)
   @assert isa(imageShape,Tuple{Integer,Integer})
   @assert isa(tileShape,Tuple{Integer,Integer})
   @assert isa(tileSpacing,Tuple{Integer,Integer})
    
   ySize = imageShape[2]*tileShape[2] + tileSpacing[2]*(tileShape[2]-1)   
   xSize = imageShape[1]*tileShape[1] + tileSpacing[1]*(tileShape[1]-1)   

   finalArray = Array(UInt8,xSize*ySize)

   fill!(finalArray,0)
   for x = 1 : tileShape[1]
        for y = 1 : tileShape[2]
          pictureNo = x + tileShape[1]*(y-1)
          if pictureNo > size(array,1)
            continue
          end
          scaled = scaleToByteInterval(array[pictureNo,:])
          for xI = 1 : imageShape[1]
           for yI = 1 : imageShape[2]
             finalArray[(y-1)*(imageShape[2]+tileSpacing[2])*xSize + (x-1)*(imageShape[1]+tileSpacing[1]) + xI + (yI-1)*xSize] = round(Int, (scaled[xI + (yI-1)*imageShape[1]]))
           end
          end
        end 
   end
   encodeGrayscalePng(name, finalArray,xSize,ySize)      
end


#############################################################
#
# Computes dimensions for rectangle of tiles for filter drawing
#
#############################################################


function computeRectangleFromNumber(number)
  square::Integer = round(Int, sqrt(number))
  i = square
  while i>=1 
    for j = 1:number
      if i*j == number 
        return (i,j)
      end
    end
    i = i - 1
  end
  @assert false "Should never arrive here"
end
