# Create data directory
if (!dir.exists("data")) dir.create("data")

# Download data
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url  <- "https://d396qaws40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(train_url, "data/pml-training.csv")
download.file(test_url,  "data/pml-testing.csv")

# Read and check dimensions
train_raw <- read.csv("data/pml-training.csv", na.strings = c("NA", "", "#DIV/0!"))
test_raw  <- read.csv("data/pml-testing.csv",  na.strings = c("NA", "", "#DIV/0!"))

dim(train_raw)  # Should be 19622 rows, 160 columns
dim(test_raw)   # Should be 20 rows, 160 columns



library(caret)

# ---- 1. Remove columns with excessive missing values ----
keep_cols <- colSums(is.na(train_raw)) / nrow(train_raw) < 0.9
train_clean <- train_raw[, keep_cols]
test_clean  <- test_raw[,  keep_cols]

# ---- 2. Remove identifier / timestamp columns BY NAME (not position) ----
drop_names <- c("X", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2",
                "cvtd_timestamp", "new_window", "num_window")
train_clean <- train_clean[, !names(train_clean) %in% drop_names]
test_clean  <- test_clean[,  !names(test_clean)  %in% drop_names]

# ---- 3. Ensure classe is a factor (training only) ----
train_clean$classe <- as.factor(train_clean$classe)

dim(train_clean)
dim(test_clean)

# ---- 4. Train/validation split and random forest baseline ----
set.seed(123)
inTrain   <- createDataPartition(train_clean$classe, p = 0.7, list = FALSE)
train_set <- train_clean[inTrain, ]
valid_set <- train_clean[-inTrain, ]

dim(train_set)
dim(valid_set)

library(randomForest)
rf_model <- randomForest(classe ~ ., data = train_set, ntree = 100)

# Validation set accuracy
pred_valid <- predict(rf_model, valid_set)
cm <- confusionMatrix(pred_valid, valid_set$classe)
cm$overall["Accuracy"]

# ---- 5. Align test set to the model's predictor names, then predict ----
model_vars <- names(train_set)[names(train_set) != "classe"]
test_clean <- test_raw[, model_vars]          # drops problem_id, enforces order

# Verify alignment BEFORE predicting
identical(names(test_clean), model_vars)      # must be TRUE
dim(test_clean)                               # 20 x 52

# Predict the 20 test cases
pred_test <- predict(rf_model, test_clean)
pred_test

write.csv(data.frame(problem_id = test_raw$problem_id, prediction = pred_test),
          "predictions.csv", row.names = FALSE)