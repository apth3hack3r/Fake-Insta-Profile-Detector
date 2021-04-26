train = read.csv("train.csv")  # loading training data
attach(train)
names(train)
dim(train)

test = read.csv("test.csv")    # loading test data

#Few plots which are also shown in ppt.

library(ggplot2)
ggplot(data=train,aes(x=fake,fill=as.factor(private))) +geom_bar(position="fill")
ggplot(data=train,aes(x=fake,fill=as.factor(external.URL))) +geom_bar(position="fill")
ggplot(data=train,aes(x=fake,fill=as.factor(profile.pic))) +geom_bar(position="fill")

# Now we start building differnt models

#### Logistic Regression ####

#model=glm(fake~profile.pic+nums.length.fullname+nums.length.username+fullname.words+name..username+description.length+X.followers+X.follows+X.posts+external.URL+private,data=train, family = binomial)
#summary(model)

# the model above was built using all predictor variables

#model using statistically important variables
model_lr=glm(fake~profile.pic+nums.length.username+X.followers+X.follows+X.posts,data=train, family = binomial)
summary(model_lr)

probs=predict(model_lr,test,type="response")

lr_pred=rep("0",120)
lr_pred[probs>0.5]="1"

table(lr_pred,test$fake)    #confusion matrix
mean(lr_pred==test$fake)    #mean of true positive and true negative (accuracy)
mean(lr_pred!=test$fake)    #mean of false positive and false negative (error) 

#### Classification Tree ####

library(tree)                             # library required to call tree function
tree_mod=tree(as.factor(fake)~.,train)    # building tree model
summary(tree_mod)
plot(tree_mod)
text(tree_mod, pretty=0)                  #plotting tree with predictor variables named on it

tree_pred=predict(tree_mod,test,type="class") # making prediction on test set
table(tree_pred, test$fake)                   #confusion matrix
mean(tree_pred==test$fake)                    #mean of true positive and true negative (accuracy)
mean(tree_pred!=test$fake)                    #mean of false positive and false negative (error)


#### Ridge Regression ####


library(glmnet)
set.seed(123)
x= model.matrix(train$fake~.,train)[,-1]
y=train$fake
cv.ridge = cv.glmnet(x, y, alpha = 0, family = "binomial")
l1.model = glmnet(x, y, alpha = 0, family = "binomial",lambda = cv.ridge$lambda.min)
x.test = model.matrix(test$fake ~.,data=test)[,-1]
probabilities = predict(l1.model, newx = x.test)

predicted.classes = ifelse(probabilities > 0.5, "1", "0")

observed.classes = test$fake
table(predicted.classes,observed.classes)
mean(predicted.classes == observed.classes)


