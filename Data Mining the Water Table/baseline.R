library(dplyr)

data <- read.csv("./data/train.csv")
labels <- read.csv("./data/labels.csv")
submit <- read.csv("./data/submit.csv")


df <- data %>% inner_join(labels, by="id")


glm(status_group ~ ., data=df, family = binomial(link="logit"))


# Some values are categorical with only one level... remove
which(sapply(df, function(x) (is.character(x) | is.factor(x)) & length(unique(x))<2))

# Too many levels
which(sapply(df, function(x) (is.character(x) | is.factor(x)) & length(unique(x))>50))

preprocess <- function(df){
  df$recorded_by <- NULL
  df$date_recorded <- NULL
  df$funder <- NULL
  df$installer <- NULL
  df$wpt_name <- NULL
  df$subvillage <- NULL
  df$lga <- NULL
  df$ward <- NULL
  df$scheme_name <- NULL
  df$scheme_management <- NULL
  return(df)
}

new_df <- preprocess(df)

# Split into train and test
idxs <- sample.int(n=nrow(new_df), size=floor(0.75*nrow(new_df)), replace=F)
train <-new_df[idxs,]
test <- new_df[-idxs,]


# Logistic regression is problematic: coefficients are bad
#model <- glm(status_group ~ ., data=train, family=binomial(link="logit"))


library(randomForest)
model <- randomForest(status_group ~ ., data=train)

y_preds <- predict(model, test)


cat("Classification rate: ",sum(test$status_group==y_preds)/nrow(test))

new_submit <- preprocess(submit)


# Hack to equalize the levels in the training and validation set
new_submit <- rbind(train[1,1:length(names(train))-1], new_submit)
new_submit <- new_submit[-1,]


out <- as.data.frame(new_submit$id)
names(out) <- c("id")

y_sub <- predict(model, new_submit)
out$status_group <- y_sub

names(out)

write.csv(out2,file="./out/demo.csv", row.names = F)
