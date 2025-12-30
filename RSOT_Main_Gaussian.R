
library(here)
# Dynamically source the utility file from your project root
source(here("RSOTR_Main_Gaussian_Utils.R"))
RunSet = 1
# Define your output directory relative to the project root
output_dir <- here("Gaussian", paste0("Set", RunSet))
# Create the folder if it doesn't exist
if(!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
# Instead of setwd(), just use output_dir when saving files:
# fwrite(df, file.path(output_dir, "results.csv"))

####################----- PARAMETERS-----####################
number_of_runs=30
for(runId in 1:number_of_runs){
  print(paste0("Run ",runId," starts:"))
  version=copy(runId)
  dimensions=c(10,10,10)
  number_of_datasets=1
  number_of_modes=length(dimensions)
  
  #------------------------> Train Test Splitting------------------------
  set.seed(1)
  N_global=450
  train_test_ratio=0.7
  N=ceiling(N_global*train_test_ratio)
  N_test=N_global-N
  
  #------------------------> Alpha Selection------------------------
  num_of_folds=5
  
  #------------------------> U_dr ------------------------
  #--generate
  sotrl_u_mean=0
  sotrl_u_sigma=0.5
  sotrl_ground_truth_R=2#4#5
  #--initialize 
  sotrl_init_u_mean=0
  sotrl_init_u_sigma=0.5
  
  #------------------------> X_it------------------------
  tensorX_iid_mean=0.1
  tensorX_is_noniid=F
  tensorX_rho=0.001
  tensorX_beta=0.01
  tensorX_mean_vector_sigma=0.1
  
  #------------------------> Y_t------------------------
  sotrl_y_sigma=0.5
  sotrl_y_mu=0
  distribution="gaussian" 
  is_scaled=F
  
  theta_vec=copy(gaussian_theta_vec)
  b_theta=copy(gaussian_b_mean)
  b_deriv=copy(gaussian_b_deriv_mean)
  b_deriv2=copy(gaussian_b_deriv2_mean)
  h_y=copy(gaussian_h_mean)
  eta_sigma=copy(gaussian_eta_sigma)
  eta_deriv_sigma=copy(gaussian_eta_deriv_sigma)
  eta_deriv2_sigma=copy(gaussian_eta_deriv2_sigma)
  b_sigma=copy(gaussian_b_sigma)
  b_deriv_sigma=copy(gaussian_b_deriv_sigma)
  b_deriv2_sigma=copy(gaussian_b_deriv2_sigma)
  
  #------------------------> Outliers------------------------
  outlier_q=0.08
  outlier_mean=2
  outlier_sd=1
  add_outlier=T
  #------------------------> Seeds------------------------
  #runId 1 and 2
  sotrl_generate_udtr_seed=copy(runId)*RunSet
  sotrl_generate_yt_seed=copy(runId)*RunSet
  sotrl_generate_xit_seed=copy(runId)*RunSet
  sotrl_initialize_udtr_seed=runId*RunSet+10
  sotrl_initialize_hosvd_seed=runId*RunSet+7
  sotrl_outlier_seed=15
  
  #------------------------> Grid Search------------------------
  maxval=4
  dividend=2
  alpha_list=c(maxval/dividend,maxval/dividend^(3/2),maxval/dividend^(4/2),maxval/dividend^(5/2),maxval/dividend^(6/2),maxval/dividend^(7/2))#c(0.5,0.75,1,1.25,1.5,2,2.5,3)# alpha_list=c(2,5,8,10,15)#c(seq(0.01,0.1,by=0.02),0.1)
  rank_list=c(2,3,4)
  #------------------------> Convergence------------------------
  local_iter_max=10
  local_iter_max_alpha_selection=15
  local_iter_max_main=15
  stopping_cond_L_init=0.0001
  stopping_cond_L_alpha_selection=0.0001
  stopping_cond_L_local_main=0.0001
  ####################----- DATA GENERATION-----####################
  ####################-----> X_it-----####################
  TensorX_ti_list=generate_X_it(tensorX_beta=tensorX_beta,rho=tensorX_rho,N_global
                                ,number_of_datasets,dimensions
                                ,seed=sotrl_generate_xit_seed,tensorX_mean_vector_sigma
                                ,is_nonIID=tensorX_is_noniid,IIDmean=tensorX_iid_mean)
  
  ####################----->> Train Test Splitting-----####################
  train_samples=vector("list",c(number_of_datasets))
  test_samples=vector("list",c(number_of_datasets))
  TensorX_ti_train_list=vector("list",c(number_of_datasets))
  TensorX_ti_test_list=vector("list",c(number_of_datasets))
  TensorX_train_global=vector("list",1)
  TensorX_test_global=vector("list",1)
  set.seed(1)
  for(t in 1:number_of_datasets){
    train_size=N[t]
    test_size=N_global[t]-N[t]
    train_indices=sample(seq_len(N_global[t]),size=train_size)
    train_samples[[t]]=train_indices
    test_samples[[t]]=setdiff(seq_len(N_global[t]),train_indices)
    
    TensorX_ti_train_list[[t]]=TensorX_ti_list[[t]][train_samples[[t]]]
    TensorX_ti_test_list[[t]]=TensorX_ti_list[[t]][test_samples[[t]]]
    
    TensorX_train_global[[1]]=c(TensorX_train_global[[1]],TensorX_ti_train_list[[t]])
    TensorX_test_global[[1]]=c(TensorX_test_global[[1]],TensorX_ti_test_list[[t]])
  }
  rm(train_size);rm(test_size);rm(train_indices)
  
  #TensorX_ti_list order adjustment
  TensorX_ti_list=vector("list",c(number_of_datasets))
  for(t in 1:number_of_datasets){
    TensorX_ti_list[[t]]=c(TensorX_ti_train_list[[t]],TensorX_ti_test_list[[t]])
  }
  
  ####################-----> Y_t-----####################
  U_dtr_groundtruth=sotr_generate_U_dtr(u_t_mean=sotrl_u_mean,u_t_sigma=sotrl_u_sigma
                                        ,dimensions,R=sotrl_ground_truth_R
                                        ,number_of_modes,number_of_datasets
                                        ,seed=sotrl_generate_udtr_seed)
  
  Y_t_list=r_gsotr_generate_Y_t(sigma=sotrl_y_sigma,mu=sotrl_y_mu,number_of_datasets
                                ,number_of_modes,U_dtr_groundtruth,TensorX_ti_train_list
                                ,N,seed=sotrl_generate_yt_seed,distribution=distribution
                                ,outlier_seed = sotrl_outlier_seed,outlier_q
                                ,outlier_mean,outlier_sd,add_outlier=T
                                ,bernoulli_multiplier,poisson_multiplier)
  
  Y_t=Y_t_list[[1]]
  prob_list=Y_t_list[[2]]
  epsilon_list=Y_t_list[[3]]
  Y_t_list_test=r_gsotr_generate_Y_t(sigma=sotrl_y_sigma,mu=sotrl_y_mu,number_of_datasets
                                     ,number_of_modes,U_dtr_groundtruth,TensorX_ti_test_list
                                     ,N_test,seed=(sotrl_generate_yt_seed+12),distribution=distribution
                                     ,outlier_seed = sotrl_outlier_seed,outlier_q
                                     ,outlier_mean,outlier_sd,add_outlier=F
                                     ,bernoulli_multiplier,poisson_multiplier)
  
  Y_t_test=Y_t_list_test[[1]]
  prob_list_test=Y_t_list_test[[2]]
  epsilon_list_test=Y_t_list_test[[3]]
  
  if(max(Y_t[[1]])>20 | min(Y_t[[1]])<(-20)){
    print("y out of range")
  }
  stdev_est=1

  ####################----- ALPHA SELECTION -----####################
  print(paste0("Run ",runId," cross validation:"))
  tic()
  set.seed(num_of_folds+RunSet*runId)
  predictions_test_cv_list=data.table()#rep(list(vector("list",c(num_of_folds))),length(alpha_list))
  summary_table_L_R_GSOTR_alpha_cv=data.table()
  for(t in 1:number_of_datasets){
    cv_indices=createFolds(Y_t[[t]],k=num_of_folds,list=TRUE,returnTrain = FALSE)
    cv_train_TensorX_ti_list=lapply(TensorX_ti_train_list,copy)
    cv_test_TensorX_ti_list=lapply(TensorX_ti_train_list,copy)
    cv_train_Y_t=lapply(Y_t,copy)
    cv_test_Y_t=lapply(Y_t,copy)
    cv_train_epsilon=lapply(epsilon_list,copy)
    cv_test_epsilon=lapply(epsilon_list,copy)
    cv_train_prob=lapply(prob_list,copy)
    cv_test_prob=lapply(prob_list,copy)
    cv_train_N=copy(N)
    cv_test_N=copy(N)
    for(fold_iter in 1:num_of_folds){
      print(paste0("************************************** Fold:",fold_iter," **************************************"))
      cv_train_TensorX_ti_list[[t]]=TensorX_ti_train_list[[t]][setdiff(1:N[t],cv_indices[[fold_iter]])]
      cv_test_TensorX_ti_list[[t]]=TensorX_ti_train_list[[t]][cv_indices[[fold_iter]]]
      cv_train_Y_t[[t]]=Y_t[[t]][setdiff(1:N[t],cv_indices[[fold_iter]])]
      cv_test_Y_t[[t]]=Y_t[[t]][cv_indices[[fold_iter]]]
      cv_train_N[t]=N[t]-length(cv_indices[[fold_iter]])
      cv_test_N[t]=length(cv_indices[[fold_iter]])
      cv_train_epsilon[[t]]=epsilon_list[[t]][setdiff(1:N[t],cv_indices[[fold_iter]])]
      cv_test_epsilon[[t]]=epsilon_list[[t]][cv_indices[[fold_iter]]]
      cv_train_prob[[t]]=prob_list[[t]][setdiff(1:N[t],cv_indices[[fold_iter]])]
      cv_test_prob[[t]]=prob_list[[t]][cv_indices[[fold_iter]]]
      #initialization with GSOTR (rank from previous stage)
      for(rank_iter in rank_list){
        summary_table_L_R_GSOTR_alpha_iter=data.table()
        Mu_t_cv_init=rep(0,number_of_datasets)
        U_dtr_list_cv_init=initialize_u_l2(u_t_mean=sotrl_init_u_mean,u_t_sigma=sotrl_init_u_sigma
                                           ,dimensions,init_tensor_R=rank_iter,rank_iter,number_of_modes,number_of_datasets
                                           ,sotrl_initialize_udtr_seed,sotrl_initialize_hosvd_seed
                                           ,type="decomposition",cv_train_TensorX_ti_list)
        Mu_t_cv_init=gsotr_update_mu(b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,Mu_t_cv_init,number_of_datasets
                                     ,number_of_modes,U_dtr_list_cv_init,cv_train_TensorX_ti_list,cv_train_N
                                     ,cv_train_Y_t,t,stopping_cond=0.000001,calculate_gsotr_loss)
        summary_table_L_LOCALMODEL=data.table()
        total_iter_number=1
        full_iter=gsotr_local_iteration(b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,eta_deriv_sigma,eta_deriv2_sigma
                                        ,b_deriv_sigma,b_deriv2_sigma,U_dtr_list_cv_init
                                        ,cv_train_TensorX_ti_list,cv_test_TensorX_ti_list
                                        ,t,cv_train_Y_t,cv_test_Y_t,Mu_t_cv_init,cv_train_N
                                        ,cv_test_N,rank_iter,dimensions,stopping_cond_L=stopping_cond_L_init
                                        ,summary_table_L_LOCALMODEL,total_iter=total_iter_number
                                        ,local_iter_max=local_iter_max,main_iter=1,distribution
                                        ,is_scaled,number_of_modes,number_of_datasets,cv_train_epsilon
                                        ,cv_test_epsilon,cv_train_prob,cv_test_prob,scale=1,calculate_gsotr_loss)
        U_dtr_list_cv_init=full_iter[[4]]
        Mu_t_cv_init=full_iter[[5]]
        alpha_iter_id=1
        for(alpha_iter in alpha_list){
          print(paste0("Alpha: ",alpha_iter))
          total_iter_number=1
          Mu_t=copy(Mu_t_cv_init)
          U_dtr_list=copy(U_dtr_list_cv_init)
          full_iter=r_gsotr_local_iteration(h_y,b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,b_sigma
                                            ,b_deriv_sigma,b_deriv2_sigma,eta_sigma,eta_deriv_sigma
                                            ,eta_deriv2_sigma,U_dtr_list ,cv_train_TensorX_ti_list
                                            ,cv_test_TensorX_ti_list,t,cv_train_Y_t,cv_test_Y_t,Mu_t
                                            ,cv_train_N,cv_test_N,rank_iter,dimensions
                                            ,stopping_cond_L=stopping_cond_L_alpha_selection
                                            ,summary_table_L_R_GSOTR_alpha_iter,total_iter=total_iter_number
                                            ,local_iter_max=local_iter_max_alpha_selection,main_iter=1
                                            ,alpha_iter,distribution,is_scaled,number_of_modes,number_of_datasets
                                            ,cv_train_prob,cv_test_prob,cv_train_epsilon,cv_test_epsilon
                                            ,scale=1,calculate_r_gsotr_loss)
          summary_table_L_R_GSOTR_alpha_iter=full_iter[[1]]
          summary_table_L_R_GSOTR_alpha_iter[,fold:=fold_iter]
          summary_table_L_R_GSOTR_alpha_iter[,R:=rank_iter]
          summary_table_L_R_GSOTR_alpha_cv=rbind(summary_table_L_R_GSOTR_alpha_cv,summary_table_L_R_GSOTR_alpha_iter)
          total_iter_number=full_iter[[2]]
          U_dtr_list=full_iter[[4]]
          Mu_t=full_iter[[5]]
          predictions_test_list=gsotr_get_predictions(U_dtr_list,Mu_t,cv_test_TensorX_ti_list
                                                      ,number_of_datasets,number_of_modes,cv_test_N
                                                      ,distribution=distribution,t)
          predictions_test=predictions_test_list[[1]]
          predictions_test_cv_list_iter=data.table(Predicted=as.numeric(predictions_test),Actual=as.numeric(cv_test_Y_t[[t]]))
          predictions_test_cv_list_iter[,alpha:=alpha_iter]
          predictions_test_cv_list_iter[,fold:=fold_iter]
          predictions_test_cv_list_iter[,R:=rank_iter]
          predictions_test_cv_list=rbind(predictions_test_cv_list,predictions_test_cv_list_iter)
          alpha_iter_id=alpha_iter_id+1
        }
      }
    }
      
  }
  predictions_test_cv_list[,dev:=Actual-Predicted]
  median_residual=predictions_test_cv_list[,.(median_dev=median(dev)),.(alpha,R)]
  predictions_test_cv_list=merge(predictions_test_cv_list,median_residual,by=c("alpha","R"))
  predictions_test_cv_list[,median_median_dev:=abs(dev-median_dev)]
  median_median_residual=predictions_test_cv_list[,.(median_median_dev=median(median_median_dev)),.(alpha,R)]
  
  truncated_data=predictions_test_cv_list[,.(alpha,R,Actual,Predicted,dev)]
  for(alpha_iter in unique(truncated_data$alpha)){
    for(rank_iter in unique(truncated_data$R)){
      truncated_data_alpha=truncated_data[alpha==alpha_iter]
      truncated_data_alpha=truncated_data_alpha[R==rank_iter]
      truncated_data_alpha[,devsq:=dev^2]
      truncated_data_alpha[,weighted_devsq:=devsq/Predicted]
      truncated_subset=truncated_data_alpha$devsq[order(truncated_data_alpha$devsq)][1:(N-round(N*0))]
      truncated_subset2=truncated_data_alpha$devsq[order(truncated_data_alpha$devsq)][1:(N-round(N*0.05))]
      truncated_subset3=truncated_data_alpha$devsq[order(truncated_data_alpha$devsq)][1:(N-round(N*0.1))]

      median_median_residual[alpha==alpha_iter & R==rank_iter,rmse_trunc:=sqrt(sum(truncated_subset)/length(truncated_subset))]
      median_median_residual[alpha==alpha_iter & R==rank_iter,rmse_trunc2:=sqrt(sum(truncated_subset2)/length(truncated_subset2))]
      median_median_residual[alpha==alpha_iter & R==rank_iter,rmse_trunc3:=sqrt(sum(truncated_subset3)/length(truncated_subset3))]
      
    }
  }
  fwrite(median_median_residual,paste0("median_median_residual",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(summary_table_L_R_GSOTR_alpha_cv,paste0("R_GSOTR_SIMRUNS_alphaCV_summary_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(predictions_test_cv_list,paste0("R_GSOTR_SIMRUNS_alphaCV_predictions_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  toc()
  alphaR_selectedall=median_median_residual[which.min(median_median_dev)]
  alphaR_main2=median_median_residual[which.min(rmse_trunc2)]
  alphaR_main3=median_median_residual[which.min(rmse_trunc3)]
  rm(predictions_test_cv_list)
  rm(summary_table_L_R_GSOTR_alpha_cv)
  
  ####################----- R_GSOTR INITIALIZATION -----####################
  print(paste0("Run ",runId," initialization:"))
  tic()
  summary_table_L_LOCALMODEL=data.table()
  AIC_summary=data.table()
  total_iter_number=1
  performance_table=data.table()
  predictions_tab_train=data.table()
  predictions_tab_test=data.table()
  rank_ind=1
  U_dtr_list_all=rep(list(vector("list",c(number_of_datasets))),length(rank_list))
  Mu_t_all=rep(list(rep(0,number_of_datasets)),length(rank_list))

  for(rank_iter in unique(alphaR_main3$R)){
    Mu_t=rep(0,number_of_datasets)
    U_dtr_list=initialize_u_l2(u_t_mean=sotrl_init_u_mean,u_t_sigma=sotrl_init_u_sigma
                               ,dimensions,init_tensor_R=rank_iter,rank_iter,number_of_modes,number_of_datasets
                               ,sotrl_initialize_udtr_seed,sotrl_initialize_hosvd_seed
                               ,type="decomposition",TensorX_ti_train_list)
    for(t in 1:number_of_datasets){
      Mu_t=gsotr_update_mu(b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,Mu_t,number_of_datasets
                           ,number_of_modes,U_dtr_list,TensorX_ti_train_list,N,Y_t,t
                           ,stopping_cond=0.000001,calculate_gsotr_loss)
      full_iter=gsotr_local_iteration(b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,eta_deriv_sigma
                                      ,eta_deriv2_sigma,b_deriv_sigma,b_deriv2_sigma,U_dtr_list
                                      ,TensorX_ti_train_list,TensorX_ti_test_list,t,Y_t,Y_t_test,Mu_t
                                      ,N,N_test,rank_iter,dimensions,stopping_cond_L=stopping_cond_L_init
                                      ,summary_table_L_LOCALMODEL,total_iter=total_iter_number
                                      ,local_iter_max=local_iter_max,main_iter=1,distribution
                                      ,is_scaled,number_of_modes,number_of_datasets,epsilon_list
                                      ,epsilon_list_test,prob_list,prob_list_test,scale=1,calculate_gsotr_loss)
      summary_table_L_LOCALMODEL=full_iter[[1]]
      total_iter_number=full_iter[[2]]
      U_dtr_list=full_iter[[4]]
      Mu_t=full_iter[[5]]
      Sigma_t=full_iter[[6]]
      AIC_summary=rbind(AIC_summary,data.table(Site=t,R=rank_iter,AIC=full_iter[[7]]
                                               ,AIC_error=full_iter[[8]]
                                               ,AIC_param=full_iter[[9]]
                                               ,BIC=full_iter[[10]]
                                               ,BIC_error=full_iter[[11]]
                                               ,BIC_param=full_iter[[12]]
                                               ,L=full_iter[[3]]))
      
      predictions_train_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_train_list
                                                   ,number_of_datasets,number_of_modes,N
                                                   ,distribution=distribution,t)
      predictions_test_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_test_list
                                                  ,number_of_datasets,number_of_modes,N_test
                                                  ,distribution=distribution,t)
      
      predictions_train=predictions_train_list[[1]]
      predictions_test=predictions_test_list[[1]]
      class_train=c()
      class_test=c()
      
      predictions_tab_train=rbind(predictions_tab_train,data.table(R=rank_iter
                                                                   ,Predicted_train=as.numeric(predictions_train)
                                                                   ,Actual_train=as.numeric(Y_t[[1]])
      ))
      predictions_tab_test=rbind(predictions_tab_test,data.table(R=rank_iter
                                                                 ,Predicted_test=as.numeric(predictions_test)
                                                                 ,Actual_test=as.numeric(Y_t_test[[1]])
      ))
      
      summary_table_L_LOCALMODEL_lastiters=merge(summary_table_L_LOCALMODEL[R==rank_iter & Site==t]
                                                 ,summary_table_L_LOCALMODEL[R==rank_iter & Site==t,.(LocalIter=max(LocalIter)),.(Site)]
                                                 ,by=c("Site","LocalIter"))
      
      performance_table=r_gsotr_add_iter_to_performance_table(performance_table,model_type="R_GSOTR_INIT",Y_t[[t]]
                                                              ,predictions_train,Y_t_test[[t]],predictions_test
                                                              ,summary_table_L_lastiters=summary_table_L_LOCALMODEL_lastiters
                                                              ,round=1,N,N_test,number_of_datasets,distribution = distribution
                                                              ,class_train,class_test,rank_iter,alpha="Initial"
                                                              ,t,prob_list[[t]],prob_list_test[[t]],epsilon_list[[t]]
                                                              ,epsilon_list_test[[t]],scale=1)
    }
    U_dtr_list_all[[rank_ind]]=U_dtr_list
    Mu_t_all[[rank_ind]]=Mu_t
    rank_ind=rank_ind+1
  }
  fwrite(AIC_summary,paste0("R_GSOTR_GAUSSIAN_init_AIC_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(performance_table,paste0("R_GSOTR_GAUSSIAN_init_perfomance_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(summary_table_L_LOCALMODEL,paste0("R_GSOTR_GAUSSIAN_init_summary_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(predictions_tab_train,paste0("R_GSOTR_GAUSSIAN_init_predictions_train_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(predictions_tab_test,paste0("R_GSOTR_GAUSSIAN_init_predictions_test_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  toc()
  rm(AIC_summary)
  rm(summary_table_L_LOCALMODEL)

  ####################----- R_GSOTR MAIN -----####################
  print(paste0("Run ",runId," main:"))
  tic()
  summary_table_L_R_GSOTR=data.table()
  total_iter_number=1
  U_dtr_list_init=copy(U_dtr_list)
  Mu_t_init=copy(Mu_t)
  predictions_tab_main_train=data.table()
  predictions_tab_main_test=data.table()
  for(alpha_selected in unique(alphaR_main3$alpha)){ 
    Mu_t=copy(Mu_t_init)
    U_dtr_list=copy(U_dtr_list_init)
    for(t in 1:number_of_datasets){
      full_iter=r_gsotr_local_iteration(h_y,b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,b_sigma,b_deriv_sigma
                                        ,b_deriv2_sigma,eta_sigma,eta_deriv_sigma,eta_deriv2_sigma,U_dtr_list
                                        ,TensorX_ti_train_list,TensorX_ti_test_list,t,Y_t,Y_t_test,Mu_t,N,N_test,min(alphaR_main3$R)
                                        ,dimensions,stopping_cond_L=stopping_cond_L_local_main,summary_table_L_R_GSOTR
                                        ,total_iter=total_iter_number,local_iter_max=local_iter_max_main,main_iter=1
                                        ,alpha_selected,distribution,is_scaled,number_of_modes,number_of_datasets
                                        ,prob_list,prob_list_test,epsilon_list,epsilon_list_test,scale=1
                                        ,calculate_r_gsotr_loss)
      summary_table_L_R_GSOTR=full_iter[[1]]
      total_iter_number=full_iter[[2]]
      U_dtr_list=full_iter[[4]]
      Mu_t=full_iter[[5]]
      Sigma_t=full_iter[[6]]
      
      predictions_train_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_train_list
                                                   ,number_of_datasets,number_of_modes,N
                                                   ,distribution=distribution,t)
      predictions_test_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_test_list
                                                  ,number_of_datasets,number_of_modes,N_test
                                                  ,distribution=distribution,t)
      
      predictions_train=predictions_train_list[[1]]
      predictions_test=predictions_test_list[[1]]
      class_train=c()
      class_test=c()
      predictions_tab_main_train=rbind(predictions_tab_main_train
                                 ,data.table(alpha=alpha_selected
                                             ,Predicted_train=as.numeric(predictions_train)
                                             ,Actual_train=as.numeric(Y_t[[1]])
                                             ))
      predictions_tab_main_test=rbind(predictions_tab_main_test
                                       ,data.table(alpha=alpha_selected
                                                   ,Predicted_test=as.numeric(predictions_test)
                                                   ,Actual_test=as.numeric(Y_t_test[[1]])
                                       ))
      summary_table_L_R_GSOTR_lastiters=merge(summary_table_L_R_GSOTR[alpha==alpha_selected & Site==t]
                                              ,summary_table_L_R_GSOTR[alpha==alpha_selected & Site==t,.(LocalIter=max(LocalIter)),.(Site)]
                                              ,by=c("Site","LocalIter"))
      
      performance_table=r_gsotr_add_iter_to_performance_table(performance_table,model_type="R_GSOTR" 
                                                              ,Y_t[[t]],predictions_train
                                                              ,Y_t_test[[t]],predictions_test
                                                              ,summary_table_L_R_GSOTR_lastiters
                                                              ,round=1,N,N_test,number_of_datasets
                                                              ,distribution = distribution
                                                              ,class_train,class_test,min(alphaR_main3$R),alpha_selected
                                                              ,t,prob_list[[t]],prob_list_test[[t]],epsilon_list[[t]]
                                                              ,epsilon_list_test[[t]],scale=1)

    }
  }
  
  fwrite(performance_table,paste0("R_GSOTR_SIMRUNS_main_perfomance_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(summary_table_L_R_GSOTR,paste0("R_GSOTR_SIMRUNS_main_summary_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(predictions_tab_main_train,paste0("R_GSOTR_SIMRUNS_main_predictions_train",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(predictions_tab_main_test,paste0("R_GSOTR_SIMRUNS_main_predictions_test",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  rm(performance_table)
  rm(summary_table_L_R_GSOTR)
  toc()
  
  ######################################## BENCHMARK SOTR #############################################
  local_iter_max=15
  stopping_cond_L_init=0.0001
  print(paste0("Run ",runId," initialization:"))
  tic()
  summary_table_L_LOCALMODEL=data.table()
  AIC_summary=data.table()
  total_iter_number=1
  performance_table=data.table()
  predictions_tab_train=data.table()
  predictions_tab_test=data.table()
  rank_ind=1
  U_dtr_list_all=rep(list(vector("list",c(number_of_datasets))),length(rank_list))
  Mu_t_all=rep(list(rep(0,number_of_datasets)),length(rank_list))
  for(rank_iter in rank_list){
    Mu_t=rep(0,number_of_datasets)
    U_dtr_list=initialize_u_l2(u_t_mean=sotrl_init_u_mean,u_t_sigma=sotrl_init_u_sigma
                               ,dimensions,init_tensor_R=rank_iter,rank_iter,number_of_modes,number_of_datasets
                               ,sotrl_initialize_udtr_seed,sotrl_initialize_hosvd_seed
                               ,type="decomposition",TensorX_ti_train_list)
    for(t in 1:number_of_datasets){
      Mu_t=gsotr_update_mu(b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,Mu_t,number_of_datasets
                           ,number_of_modes,U_dtr_list,TensorX_ti_train_list,N,Y_t,t
                           ,stopping_cond=0.000001,calculate_gsotr_loss)
      full_iter=gsotr_local_iteration(b_theta,b_deriv,b_deriv2,theta_vec,stdev_est,eta_deriv_sigma
                                      ,eta_deriv2_sigma,b_deriv_sigma,b_deriv2_sigma,U_dtr_list
                                      ,TensorX_ti_train_list,TensorX_ti_test_list,t,Y_t,Y_t_test,Mu_t
                                      ,N,N_test,rank_iter,dimensions,stopping_cond_L=stopping_cond_L_init
                                      ,summary_table_L_LOCALMODEL,total_iter=total_iter_number
                                      ,local_iter_max=local_iter_max,main_iter=1,distribution
                                      ,is_scaled,number_of_modes,number_of_datasets,epsilon_list
                                      ,epsilon_list_test,prob_list,prob_list_test,scale=1,calculate_gsotr_loss)
      summary_table_L_LOCALMODEL=full_iter[[1]]
      total_iter_number=full_iter[[2]]
      U_dtr_list=full_iter[[4]]
      Mu_t=full_iter[[5]]
      Sigma_t=full_iter[[6]]
      AIC_summary=rbind(AIC_summary,data.table(Site=t,R=rank_iter,AIC=full_iter[[7]]
                                               ,AIC_error=full_iter[[8]]
                                               ,AIC_param=full_iter[[9]]
                                               ,BIC=full_iter[[10]]
                                               ,BIC_error=full_iter[[11]]
                                               ,BIC_param=full_iter[[12]]
                                               ,L=full_iter[[3]]))
      
      predictions_train_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_train_list
                                                   ,number_of_datasets,number_of_modes,N
                                                   ,distribution=distribution,t)
      predictions_test_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_test_list
                                                  ,number_of_datasets,number_of_modes,N_test
                                                  ,distribution=distribution,t)
      
      predictions_train=predictions_train_list[[1]]
      predictions_test=predictions_test_list[[1]]
      class_train=c()
      class_test=c()
      
      predictions_tab_train=rbind(predictions_tab_train,data.table(R=rank_iter
                                                                   ,Predicted_train=as.numeric(predictions_train)
                                                                   ,Actual_train=as.numeric(Y_t[[1]])
      ))
      predictions_tab_test=rbind(predictions_tab_test,data.table(R=rank_iter
                                                                 ,Predicted_test=as.numeric(predictions_test)
                                                                 ,Actual_test=as.numeric(Y_t_test[[1]])
      ))
      
      summary_table_L_LOCALMODEL_lastiters=merge(summary_table_L_LOCALMODEL[R==rank_iter & Site==t]
                                                 ,summary_table_L_LOCALMODEL[R==rank_iter & Site==t,.(LocalIter=max(LocalIter)),.(Site)]
                                                 ,by=c("Site","LocalIter"))
      
      performance_table=r_gsotr_add_iter_to_performance_table(performance_table,model_type="R_GSOTR_INIT",Y_t[[t]]
                                                              ,predictions_train,Y_t_test[[t]],predictions_test
                                                              ,summary_table_L_lastiters=summary_table_L_LOCALMODEL_lastiters
                                                              ,round=1,N,N_test,number_of_datasets,distribution = distribution
                                                              ,class_train,class_test,rank_iter,alpha="Initial"
                                                              ,t,prob_list[[t]],prob_list_test[[t]],epsilon_list[[t]]
                                                              ,epsilon_list_test[[t]],scale=1)
    }
    U_dtr_list_all[[rank_ind]]=U_dtr_list
    Mu_t_all[[rank_ind]]=Mu_t
    rank_ind=rank_ind+1
    
  }
  fwrite(AIC_summary,paste0("BENCHMARK_GSOTR_GAUSSIAN_AIC_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(performance_table,paste0("BENCHMARK_GSOTR_GAUSSIAN_perfomance_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(summary_table_L_LOCALMODEL,paste0("BENCHMARK_GSOTR_GAUSSIAN_summary_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(predictions_tab_train,paste0("BENCHMARK_GSOTR_GAUSSIAN_predictions_train_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  fwrite(predictions_tab_test,paste0("BENCHMARK_GSOTR_GAUSSIAN_predictions_test_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  toc()
  rm(AIC_summary)
  rm(summary_table_L_LOCALMODEL)

  
  ####################----- BENCHMARK RLR -----####################
  #-----vectorizing data:
  X_linear_train=data.table()
  t=1
  for(i in 1:N[t]){
    X_linear_train=rbind(X_linear_train,t(vec(k_unfold(TensorX_ti_train_list[[t]][[i]],1))))
  }
  Y_linear_train=Y_t[[t]]
  
  X_linear_test=data.table()
  t=1
  for(i in 1:N_test[t]){
    X_linear_test=rbind(X_linear_test,t(vec(k_unfold(TensorX_ti_test_list[[t]][[i]],1))))
  }
  Y_linear_test=Y_t_test[[t]]
  
  X_linear=rbind(X_linear_train,X_linear_test)
  
  pca_train=prcomp(X_linear,retx = T)
  pca_var=pca_train$sdev^2
  number_of_prcomp=which(cumsum(pca_var)/sum(pca_var) >= 0.90)[1]
  X_pca=as.data.table(pca_train$x[,1:number_of_prcomp])
  X_pca_train=X_pca[1:N[t]]
  X_pca_test=X_pca[(N[t]+1):nrow(X_pca)]
  
  train=cbind(target=as.numeric(Y_linear_train),X_pca_train)
  
  model_fit=rlm(target~.,data=train)
  Y_predicted=predict(model_fit,newdata=X_pca_train)
  Y_predicted_test=predict(model_fit, newdata=X_pca_test)
  scale=1
  rmse_train_scaled=calculate_RMSE(Y_linear_train,Y_predicted)
  rmse_test_scaled=calculate_RMSE(Y_linear_test,Y_predicted_test)
  rmse_train=calculate_RMSE(scale*Y_linear_train,scale*Y_predicted)
  rmse_test=calculate_RMSE(scale*Y_linear_test,scale*Y_predicted_test)
  wmape_train=calculate_WMAPE(Y_linear_train,Y_predicted)
  wmape_test=calculate_WMAPE(Y_linear_test,Y_predicted_test)
  bias_train_scaled=calculate_bias(Y_linear_train,Y_predicted)
  bias_test_scaled=calculate_bias(Y_linear_test,Y_predicted_test)
  bias_train=calculate_bias(scale*Y_linear_train,scale*Y_predicted)
  bias_test=calculate_bias(scale*Y_linear_test,scale*Y_predicted_test)
  rmse_corrected_train=calculate_RMSE_errorcorrected(scale*Y_linear_train,scale*Y_predicted,epsilon_list[[1]])
  rmse_corrected_test=calculate_RMSE_errorcorrected(scale*Y_linear_test,scale*Y_predicted_test,epsilon_list_test[[1]])
  
  performance_table=data.table(Model="PCA_RLRHuber",Round=1,Site=1,R="vector"
                               ,PCA_percent=1,N_train=N[t],N_test=N_test[t]
                               ,RMSE_train_scaled=rmse_train_scaled,RMSE_test_scaled=rmse_test_scaled
                               ,RMSE_train=rmse_train,RMSE_test=rmse_test
                               ,WMAPE_train=wmape_train,WMAPE_test=wmape_test
                               ,BIAS_train_scaled=bias_train_scaled,BIAS_test_scaled=bias_test_scaled
                               ,BIAS_train=bias_train,BIAS_test=bias_test
                               ,RMSE_corrected_train=rmse_corrected_train
                               ,RMSE_corrected_test=rmse_corrected_test)
  
  preds=data.table(Actual_test=as.numeric(Y_linear_test),Predicted_test=as.numeric(Y_predicted_test))
  fwrite(preds,paste0("BENCHMARK_PCA_RLRHUBER_GAUSSIAN_predstest_",distribution,"_percent100_RunSet",RunSet,"_runID",runId,".csv"))
  
  ####################----- BENCHMARK EXP SQ LOSS -----####################
  number_of_prcomp=which(cumsum(pca_var)/sum(pca_var) >= 0.85)[1]
  X_pca=as.data.table(pca_train$x[,1:number_of_prcomp])
  X_linear_train=X_pca[1:N[t]]
  X_linear_test=X_pca[(N[t]+1):nrow(X_pca)]
  
  set.seed(num_of_folds+RunSet*runId)
  cv_indices=createFolds(Y_t[[t]],k=num_of_folds,list=TRUE,returnTrain = FALSE)
  Y_predicted_list=data.table()#vector("list",c(15))
  
  for(fold_iter in 1:num_of_folds){
    print(paste0("************************************** Fold:",fold_iter," **************************************"))
    X_linear_train_CVTRAIN=X_linear_train[setdiff(1:N[t],cv_indices[[fold_iter]])]
    X_linear_train_CVTEST=X_linear_train[cv_indices[[fold_iter]]]
    Y_linear_train_CVTRAIN=Y_linear_train[setdiff(1:N[t],cv_indices[[fold_iter]])]
    Y_linear_train_CVTEST=Y_linear_train[cv_indices[[fold_iter]]]
    # mad_list=data.table()
    for(gamma_iter in c(0.00001,0.0001,0.001,0.005,0.01,0.05,0.1,0.5)){
      print(paste0("************************************** gamma:",gamma_iter," **************************************"))
      robustlm_fit=robustlm(as.matrix(X_linear_train_CVTRAIN), as.numeric(Y_linear_train_CVTRAIN),gamma=gamma_iter)
      Y_predicted=predict(robustlm_fit, as.matrix(X_linear_train_CVTEST))
      Y_predicted_list=rbind(Y_predicted_list,data.table(fold=fold_iter,gamma=gamma_iter,predicted=as.numeric(Y_predicted),actual=as.numeric(Y_linear_train_CVTEST)))
    }
  }
  
  Y_predicted_list[,dev:=predicted-actual]
  Y_predicted_list[,median_dev:=median(dev),.(gamma)]
  Y_predicted_list[,dev_median_dev:=abs(dev-median_dev)]
  mad_list=Y_predicted_list[,.(MAD=median(dev_median_dev)),.(gamma)]
  gamma_selected=mad_list[which.min(MAD)]$gamma
  
  robustlm_fit=robustlm(as.matrix(X_linear_train), as.numeric(Y_linear_train),gamma=gamma_selected)
  Y_predicted=predict(robustlm_fit, as.matrix(X_linear_train))
  Y_predicted_test=predict(robustlm_fit, as.matrix(X_linear_test))
  
  performance_table=r_gsotr_add_iter_to_performance_table(performance_table,model_type="ExpSqLoss" 
                                                          ,Y_linear_train,Y_predicted
                                                          ,Y_linear_test,Y_predicted_test
                                                          ,summary_table_L_lastiters=NA
                                                          ,round=1,N,N_test,number_of_datasets
                                                          ,distribution = distribution
                                                          ,class_train=NA,class_test=NA,R="vector",alpha=NA
                                                          ,t,p_t=NA,p_t_test=NA,epsilon=epsilon_list[[1]]
                                                          ,epsilon_test=epsilon_list_test[[1]],scale=1
  )
  performance_table[,gamma:=gamma_selected]
  fwrite(performance_table,paste0("BENCHMARK_EXPSQLOSS_GAUSSIAN_perfomance_",distribution,"_RunSet",RunSet,"_runID",runId,".csv"))
  
}

