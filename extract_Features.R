# cvCells
# Using shape and texture features to classify microscopy images

############################
# UPDATED: 11/08/2018 
# Error in .C("cfft_r2c_2d", as.integer(nC), as.integer(nR), as.double(data),  : 
# "cfft_r2c_2d" not available for .C() for package "fftwtools"
# computeFeatures default call generates this error!
# changed compute function to 4x individual calls, feature count drops from 125 to 45
############################

# Part I - Data Management ----

# load image processing lib
library(EBImage)

# species names here for individual dBase
sp.sec<-"all"

# basic book-keeping -- list all files
refs.all <-list.files(pattern = paste0("*-g-",sp.sec))
segs.all <-list.files(pattern = paste0("*-g-",sp.sec))

# for all samples in library
refs.all <-list.files(pattern = "*-g-")
segs.all <-list.files(pattern = "*-s-")

beg = 1
end = as.numeric(length(refs.all))

# short labels for meta-data
ref <-as.character(strsplit(refs.all,".tif"))
seg <-as.character(strsplit(segs.all,".tif"))



# main loop
for (i in beg:end){
  
  # read and label segmented images
  refs.array <-readImage(refs.all[i])
  seg.array <-readImage(segs.all[i])
  seg.labelled <-bwlabel(seg.array)
  
  # construct holders for feature-results
  writer.1 <-paste0("shapes",i)
  writer.2 <-paste0("textures",i)
  writer.3 <-paste0("moments",i)
  writer.4 <-paste0("basics",i)
  
  # compute features
  fts.shp <-computeFeatures.shape(seg.labelled,refs.array)
  fts.tex <-computeFeatures.haralick(seg.labelled,refs.array)
  
  # extra feats
  fts.moment <-computeFeatures.moment(seg.labelled,refs.array)
  fts.basic <-computeFeatures.basic(seg.labelled,refs.array)
  
  # assigning source file ids to rownames for book-keeping
  rownames(fts.shp) <-rep(ref[i],dim(fts.shp)[1])
  
  # use assign for each feature set
  assign(writer.1,fts.shp)
  assign(writer.2,fts.tex)
  
  # extra feats
  assign(writer.3,fts.moment)
  assign(writer.4,fts.basic)
  

}

# concenate pieces into one list
rm(list=ls(pattern = "^fts"))
pieces.shp <-Filter(function(x) is(x, "matrix"),mget(ls(pattern = "^shapes")))
pieces.tex <-Filter(function(x) is(x, "matrix"), mget(ls(pattern= "^textures")))
pieces.mom <-Filter(function(x) is(x, "matrix"), mget(ls(pattern= "^moments")))
pieces.bas <-Filter(function(x) is(x, "matrix"), mget(ls(pattern= "^basics")))

# construct matrices 
data.shapes <-do.call(rbind,pieces.shp)
data.textures <-do.call(rbind,pieces.tex)
data.moments <-do.call(rbind,pieces.mom)
data.basics <-do.call(rbind,pieces.bas)

# trim data
crt <-which(data.shapes[,1]>=7500) # size criteria here
data.shapes.trim <-data.shapes[crt,] # apply to shape data
data.textures.trim <-data.textures[crt,] # apply for textures
data.moments.trim <-data.moments[crt,] # apply for moments
data.basics.trim <-data.basics[crt,] # apply for basics

# bind rows for full array of features
array.images <-cbind(data.shapes.trim,data.textures.trim,
                     data.moments.trim,data.basics.trim)


# Part II - Data Export ----

# libs needed
library(stringr)
library(magrittr)

rm(list = ls(pattern = "^shapes"))
rm(list = ls(pattern = "^textures"))
rm(list = ls(pattern = "^moments"))
rm(list = ls(pattern = "^basics"))


# to DF - change name here!!
array.dfs <-as.data.frame(array.images)

# create a column of image-names 
rNames <-rownames(array.dfs)
rNames.tag <-str_sub(rNames,-4)

array.dfs %<>% cbind(rNames.tag,.) # bind to array
save(array.dfs,file=paste0("array-updated","-",sp.sec,".Rdata")) # export as .Rdata file
write.csv(array.dfs,file = paste0("array-updated","-",sp.sec,".csv"),row.names = FALSE) # write out feature matrix to .csv


