#######################################################################################################################

#########################################PACKAGES###########################################################################

library(ada)
library(corrplot)
library(kableExtra)
library(DT)
library(data.table)
library(tidyr)
library(tidyverse)
library(ggpubr)
library(ggcorrplot)
library(dplyr)
library(forcats)
library(scales)
library(rsample)
library(MASS)
library(class)
library(tidymodels)
library(MASS)
library(discrim)
library(purrr)
library(ROCR)
library(kknn)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caret)
library(doParallel)
library(ranger)
library(dials)
library(themis)
library(xgboost)
library(baguette)

#######################################################################################################################

#########################################DONNEES###########################################################################
data <- read.csv("online_shoppers_intention.csv", header = T, sep = ",")

data <- data%>%
  rename(Nb_page_administrative = Administrative)%>%
  rename(Duree_administrative = Administrative_Duration)%>%
  rename(Nb_page_information = Informational)%>%
  rename(Duree_information = Informational_Duration)%>%
  rename(Nb_page_produit = ProductRelated)%>%
  rename(Duree_produit = ProductRelated_Duration)%>%
  rename(Pourcentage_pub_quit = BounceRates)%>%
  rename(Taux_sortie= ExitRates)%>%
  rename(Page_av_achat = PageValues)%>%
  rename(Jour_special = SpecialDay)%>%
  rename(Mois = Month)%>%
  rename(Systeme_exploitation = OperatingSystems)%>%
  rename(Navigateur = Browser)%>%
  rename(Pays = Region)%>%
  rename(Type_trafic = TrafficType)%>%
  rename(Type_visiteur = VisitorType)%>%
  rename(Achat = Revenue)


data$Type_visiteur<-factor(data$Type_visiteur)
levels(data$Type_visiteur) <- c("Nouveaux", "Autres", "Anciens")

data$Mois<-as.factor(data$Mois)


data$Weekend<-as.factor(data$Weekend)
levels(data$Weekend) <- c("Non","Oui")

data$Achat <- as.factor(data$Achat)
levels(data$Achat) <- c("Non","Oui")

data$Nb_page_total = data$Nb_page_administrative+data$Nb_page_information+data$Nb_page_produit
data$Duree_total = data$Duree_administrative+data$Duree_information+data$Duree_produit

#######################################################################################################################

#########################################SPLIT###########################################################################

set.seed(78)

data_split <- initial_split(data, strata = Achat, prop = 2/3)
data_train <- training(data_split)
data_test <- testing(data_split)

#######################################################################################################################

#########################################KNN###########################################################################


knn_rec <- recipe(Achat~., data = data_train)%>%
  step_dummy(all_nominal_predictors())%>%
  step_zv(all_numeric_predictors())%>%
  step_normalize(all_numeric_predictors())%>%
  prep(training = data_train)

#Pas de NA pas besoin de les imputer

knn_model <- nearest_neighbor(neighbors = tune())%>%
  set_mode("classification")%>%
  set_engine("kknn")


knn_wf <- workflow()%>%
  add_recipe(knn_rec)%>%
  add_model(knn_model)

knn_cv <- vfold_cv(data_train, v=5, repeats = 5)

knn_grid <- tibble(neighbors=1:30)

doParallel::registerDoParallel(cores = 10)

knn_tune <- tune_grid(
  object = knn_wf,
  resamples = knn_cv,
  grid = knn_grid,
  metrics = metric_set(accuracy)
)

stopImplicitCluster()

best_k <- select_best(knn_tune)

knn_final_wf <- knn_wf%>%
  finalize_workflow(best_k)

save(knn_tune, file = "knn_tune.Rdata")

save(knn_final_wf, file = "knn_final_wf.Rdata")

#######################################################################################################################

#########################################LDA###########################################################################

lda_rec <- recipe(Achat~., data = data_train)%>%
  step_dummy(all_nominal_predictors())%>%
  step_zv(all_numeric_predictors())%>%
  step_corr(all_numeric_predictors())%>%
  prep(training = data_train)

lda_model <- discrim_linear() %>% 
  set_mode("classification") %>%
  set_engine("MASS")

lda_wf <- workflow()%>%
  add_model(lda_model)%>%
  add_recipe(lda_rec)

save(lda_wf, file = "lda_wf.Rdata")

#######################################################################################################################

#########################################QDA###########################################################################

qda_rec <- recipe(Achat~., data = data_train)%>%
  step_dummy(all_nominal_predictors())%>%
  step_corr(all_numeric_predictors())%>%
  prep(training = data_train)

qda_mod <- discrim_quad() %>% 
  set_mode("classification") %>%
  set_engine("MASS")

qda_wf <- workflow()%>%
  add_model(qda_mod)%>%
  add_recipe(qda_rec)

save(qda_wf, file = "qda_wf.Rdata")

#######################################################################################################################

#########################################ARBRES###########################################################################

tree_rec <- recipe(Achat~., data = data_train)

tree_mod <- decision_tree()%>%
  set_mode("classification")%>%
  set_engine("rpart")%>%
  set_args(cost_complexity = tune())

tree_wf <- workflow()%>%
  add_model(tree_mod)%>%
  add_recipe(tree_rec)

tree_cv <- vfold_cv(data_train, v=5, repeats = 5)

tree_grid <- grid_regular(cost_complexity(range = c(-5,-0.1)),
                          levels = 20)

registerDoParallel(cores = 8)

tree_tune <- tune_grid(
  tree_wf,
  resamples = tree_cv,
  grid = tree_grid,
  metrics = metric_set(accuracy)
)

stopImplicitCluster()

tree_final_wf <- tree_wf%>% 
  finalize_workflow(select_best(tree_tune))

save(tree_final_wf, file = "tree_final_wf.Rdata")
save(tree_tune, file = "tree_tune.Rdata")

#######################################################################################################################

#########################################BAGGING###########################################################################

bag_rec <- recipe(Achat~., data = data_train)%>%
  step_corr(all_numeric_predictors())%>%
  prep(training = data_train)

bag_mod <- bag_tree(cost_complexity = tune(),
                    tree_depth = tune())%>%
  set_engine("rpart")%>%
  set_mode("classification")

bag_wf <- workflow()%>%
  add_model(bag_mod)%>%
  add_recipe(bag_rec)

bag_cv <- vfold_cv(data_train, v=5, repeats = 2)

bag_grid <- grid_regular(cost_complexity(range = c(-5,-0.1)),
                         tree_depth(range = c(2,100)),
                         levels = 10)

registerDoParallel(cores = 8)

system.time(
  bag_tune <- tune_grid(
    bag_wf, 
    resamples = bag_cv, 
    grid = bag_grid, 
    metrics = metric_set(accuracy)
  )
)

stopImplicitCluster()


bag_final_wf <- bag_wf%>% 
  finalize_workflow(select_best(bag_tune))

save(bag_final_wf, file = "bag_final_wf.Rdata")
save(bag_tune, file = "bag_tune.Rdata")

#######################################################################################################################

#########################################RANDOM FOREST###########################################################################


rf_rec <- recipe(Achat~., data = data_train)%>%
  step_zv(all_numeric_predictors())%>%
  step_corr(all_numeric_predictors())%>%
  prep(training = data_train)

rf_mod <- rand_forest(mtry = tune(),
                       min_n = tune(),
                       trees = tune())%>%
  set_engine("randomForest", importance = T)%>%
  set_mode("classification")

rf_wf <- workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(rf_rec)

rf_cv <- vfold_cv(data_train, v=4, repeats = 2)

rf_grid <- crossing(mtry = 1:19,
                    trees = c(50, 100, 200, 500),
                    min_n = c(2, 10, 50, 100))


registerDoParallel(cores = 8)

system.time(
  rf_tune <- tune_grid(
    rf_wf, 
    resamples = rf_cv, 
    grid = rf_grid, 
    metrics = metric_set(accuracy)
  )
)

stopImplicitCluster()

rf_final_wf <- rf_wf%>% 
  finalize_workflow(select_best(rf_tune))

save(rf_final_wf, file = "rf_final_wf.Rdata")
save(rf_tune, file = "rf_tune.Rdata")
  
#######################################################################################################################

#########################################BOOSTING###########################################################################


boost_rec <- recipe(Achat~., data = data_train)%>%
  step_zv(all_numeric_predictors())%>%
  step_corr(all_numeric_predictors())%>%
  step_dummy(all_nominal_predictors())%>%
  prep(training = data_train)

boost_mod <- boost_tree(mtry = tune(),
                      min_n = tune(),
                      trees = tune(),
                      learn_rate = tune())%>%
  set_engine("xgboost", importance = T)%>%
  set_mode("classification")

boost_wf <- workflow() %>% 
  add_model(boost_mod) %>% 
  add_recipe(boost_rec)

boost_cv <- vfold_cv(data_train, v=5, repeats = 2)

boost_grid <- crossing(mtry = 1:17,
                    trees = c(50, 100, 200, 500),
                    min_n = c(2, 10, 20, 50, 100),
                    learn_rate = c(0.001,0.01,0.1,1))


registerDoParallel(cores = 8)

system.time(
  boost_tune <- tune_grid(
    boost_wf, 
    resamples = boost_cv, 
    grid = boost_grid, 
    metrics = metric_set(accuracy)
  )
)

stopImplicitCluster()

boost_final_wf <- boost_wf%>% 
  finalize_workflow(select_best(boost_tune))

save(boost_final_wf, file = "boost_final_wf.Rdata")
save(boost_tune, file = "boost_tune.Rdata")




