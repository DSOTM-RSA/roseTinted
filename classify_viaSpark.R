# preProcess - Multiclass: preProcessing in caret

##################################
# UPDATED 11/08/2018
# additional lines added for new updated incoming feature data-set

setwd("~/Documents/GitArchive/cvCells/macro-tests/in-grey-seg")

library(caret)
load("array-all.Rdata")
load("array-updated-all.Rdata")


# identifying correlated parameters (i.e not for PLS)
# remove x and y pos and factor
feats <- array.dfs[c(-1,-2,-3)] 
feats <- array.dfs[c(-1,-34,-35)] 

descrCor <- cor(feats)
highlyCorDescr <- findCorrelation(descrCor, cutoff = .95)
filteredDescr <- feats[,-highlyCorDescr]

# trim data back
array.all.trim<-cbind(array.dfs[1],filteredDescr)
array.updated.all.trim<-cbind(array.dfs[1],filteredDescr)

#  write out feature matrix to .csv 
write.csv(array.updated.all.trim,file = "array.updated.all.trim.csv",row.names = FALSE) 

# clean up
rm(array.dfs,feats,descrCor,highlyCorDescr,filteredDescr,array.all.trim) 

##############################
# Analysis One: Using Spark ML ----

# order for single use sparklyr
library(sparklyr)

# connect to spark instance
sc <- spark_connect(master = "local",version="1.6.2") 

# load dplyr for easy data management verbs
library(dplyr) 

# load feature data-set (trimmed un-correlated feats)
where <- getwd()
features_tbl <- spark_read_csv(sc, name = 'featLib', 
                               path = paste0(where,"/array.all.trim.csv"))

features_tbl <- spark_read_csv(sc, name = 'featLib', 
                               path = paste0(where,"/array.updated.all.trim.csv"))


# building a full random forest
rf_full_model <- features_tbl %>% 
  ml_random_forest(rNames_tag ~., type = "classification")

# make predictions
rf_full_predict <- sdf_predict(rf_full_model, features_tbl) %>% 
  ft_string_indexer("rNames_tag","rNames_idx") %>% collect

rf_full_predict <- sdf_predict(features_tbl, rf_full_model) %>% 
  ft_string_indexer("rNames_tag","rNames_idx") %>% collect

# print the classification results
table(rf_full_predict$rNames_idx,rf_full_predict$prediction) 

# mapping of labels to indicies
ft_string2idx <- features_tbl %>% 
  ft_string_indexer("rNames_tag", "rNames_idx") %>%
  ft_index_to_string("rNames_idx", "rNames_remap") %>%
  collect

# show mapping
table(ft_string2idx$rNames_idx,ft_string2idx$rNames_remap) 


