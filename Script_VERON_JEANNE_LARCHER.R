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

data$Achat<-as.factor(data$Achat)
levels(data$Achat) <- c("Non","Oui")

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

knn_cv <- vfold_cv(data, v=5, repeats = 5)

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
