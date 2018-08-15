############################
# Analysis Two: Using H2o ML ----

# order for single use H20
library(rsparkling)
library(sparklyr)
library(dplyr)
library(h2o)

h2o.init()

# connect to spark instance
sc <- spark_connect(master = "local",version="1.6.2") 

# load feature data-set (trimmed un-correlate feats)
setwd("~/Documents/GitArchive/roseTinted/in-grey-seg")
where <- getwd()
#features_tbl <- spark_read_csv(sc, name = 'featLib', path = paste0(where,"/array.all.trim.csv"))
features_tbl <- spark_read_csv(sc, name = 'featLib', path = "array.all.trim.csv")

partitions <- features_tbl %>% 
  sdf_partition(training = 0.75, test = 0.5, seed = 1099) # partioning into train and test using Spark data-frame framework

training <- as_h2o_frame(sc, partitions$training,strict_version_check = FALSE) # make H2O dataframes
test <- as_h2o_frame(sc, partitions$test,strict_version_check = FALSE) # make H20 dataframes

# kmeans test
kmeans_model <- h2o.kmeans(training_frame = training, 
                           x = 2:31,
                           k = 3,
                           seed = 1)

h2o.centers(kmeans_model)
h2o.centroid_stats(kmeans_model)

# pca test
pca_model <- h2o.prcomp(training_frame = training,
                        x = 2:31,
                        k = 4,
                        seed = 1)
print(pca_model)


# RF test
dat.H2o.train <- as_h2o_frame(sc, features_tbl,strict_version_check = FALSE)
y <- "rNames_tag"
x <- setdiff(names(dat.H2o.train), y)
dat.H2o.train[,y] <- as.factor(dat.H2o.train[,y])


# split into training and testing in H20 frameowrk
splits <- h2o.splitFrame(dat.H2o.train, seed = 1)


rf_model <- h2o.randomForest(x = x, 
                             y = y,
                             training_frame = splits[[1]],
                             validation_frame = splits[[2]],
                             nbins = 32,
                             max_depth = 5,
                             ntrees = 20,
                             seed = 1)

h2o.confusionMatrix(rf_model) # metrics on full data-set : potenital peak accuracy
h2o.confusionMatrix(rf_model, valid = TRUE) # metrics on test set : production setting

# get variable importance
h2o.varimp_plot(rf_model,num_of_features = 10)

# GBM test
gbm_model <- h2o.gbm(x = x, 
                     y = y,
                     training_frame = splits[[1]],
                     validation_frame = splits[[2]],                     
                     ntrees = 20,
                     max_depth = 3,
                     learn_rate = 0.01,
                     col_sample_rate = 0.7,
                     seed = 1)

h2o.confusionMatrix(gbm_model) # metrics on full data-set : potenital peak accuracy
h2o.confusionMatrix(gbm_model, valid = TRUE) # metrics on test set : production setting

# get variable importance
h2o.varimp_plot(gbm_model,num_of_features = 10)



# deep-learning test
deep_model <- h2o.deeplearning(x = x, y = y, training_frame = splits[[1]],
                               validation_frame = splits[[2]],
                               epochs=60, 
                               variable_importances = TRUE,
                               max_runtime_secs=30)

deep_predictions <- h2o.predict(deep_model, splits[[1]])


deeper_model <- h2o.deeplearning(x = x, y = y, training_frame = splits[[1]],
                               validation_frame = splits[[2]],
                               epochs=60, hidden = c(64,64),
                               variable_importances = TRUE,
                               max_runtime_secs=30)

deeper_predictions <- h2o.predict(deeper_model, splits[[1]])

# get variable importance
h2o.varimp_plot(gbm_model,num_of_features = 10)

h2o.confusionMatrix(deep_model) # metrics on full data-set : potenital peak accuracy
h2o.confusionMatrix(deep_model, valid = TRUE) # metrics on test set : production setting

h2o.confusionMatrix(deeper_model) # metrics on full data-set : potenital peak accuracy
h2o.confusionMatrix(deeper_model, valid = TRUE) # metrics on test set : production setting

