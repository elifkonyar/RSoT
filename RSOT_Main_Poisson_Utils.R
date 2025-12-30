# Package Management --------------------------------------------------------
pkgs <- c("rTensor", "data.table", "MASS", "caret", "epca", "parallel", "tictoc", "robustlm","here")
load_pkgs <- function(p) {
  if (!require(p, character.only = TRUE)) {
    install.packages(p, dependencies = TRUE)
    library(p, character.only = TRUE)
  }
}
invisible(lapply(pkgs, load_pkgs))

####################-----TENSOR UTILS-----#####################
reshape_factormat_to_tensor=function(U_dtr_list,number_of_modes,dimensions,R,t){
  #u_tensor=array(rep(0,prod(dimensions)),dim = dimensions)
  U_dtr_list_t=vector("list",c(number_of_modes))
  U_dtr_list_t_vec=vector("list",c(number_of_modes))
  for(mode_iter in 1:number_of_modes){
    U_dtr_list_t[[mode_iter]]=U_dtr_list[[mode_iter]][[t]]
  }
  kro_prod=rep(0,prod(dimensions))
  for(r in 1:R){
    for(mode_iter in 1:number_of_modes){
      U_dtr_list_t_vec[[mode_iter]]=U_dtr_list_t[[number_of_modes-mode_iter+1]][,r]
    }
    kro_prod=kro_prod+kronecker_list(U_dtr_list_t_vec)
  }
  u_tensor=as.tensor(array(kro_prod,dim=dimensions))
  return(u_tensor)
}

calculate_khatrirao_product=function(number_of_modes,U_dtr_list,d,t){
  #generation of K_U_dt
  d_ind=1
  khatrirao_set=rev(setdiff(seq(number_of_modes:1),d))
  if(length(khatrirao_set)>1){
    khatri_rao_product=khatri_rao(U_dtr_list[[khatrirao_set[d_ind]]][[t]],U_dtr_list[[khatrirao_set[d_ind+1]]][[t]])
    d_ind=d_ind+2
    if((number_of_modes-1)-(d_ind)>=0){
      for(d_ind in d_ind:(number_of_modes-1)){
        khatri_rao_product=khatri_rao(khatri_rao_product,U_dtr_list[[khatrirao_set[d_ind]]][[t]])
      }
    }
  }else{
    khatri_rao_product=copy(U_dtr_list[[khatrirao_set[d_ind]]][[t]])
  }
  return(khatri_rao_product)
}

####################-----DISTRIBUTION FUNCS-----#####################
poisson_h_mean=function(Y_t,t,i){
  return(1/factorial(Y_t[[t]][i]))
}
poisson_b=function(theta){
  return(exp(theta))
}
poisson_b_deriv=function(theta){
  return(exp(theta))
}
poisson_b_deriv2=function(theta){
  return(exp(theta))
}

####################-----LOSS CALCULATION-----#####################
calculate_r_gsotr_loss=function(h_y,b_theta,Y_t,TensorX_ti_train_list
                                ,U_dtr_list,Mu_t,d,t,number_of_modes,alpha,N){
  loss=0
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  for(i in 1:N[t]){
    matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
    inner_prod=sum(diag(t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product))
    m_idr=Mu_t[t]+inner_prod
    expon_term_i=(h_y(Y_t,t,i)*exp(Y_t[[t]][i]*m_idr-b_theta(m_idr)))^(alpha)
    loss_i=(1-expon_term_i)/alpha
    loss=loss+loss_i
  }
  return(loss)
}

calculate_gsotr_loss=function(b_theta,Y_t,TensorX_ti_train_list
                              ,U_dtr_list,Mu_t,d,t,number_of_modes,N){
  loss=0
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  for(i in 1:N[t]){
    matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
    inner_prod=sum(diag(t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product))
    m_idr=Mu_t[t]+inner_prod
    neg_loglike_iter=-Y_t[[t]][i]*m_idr+b_theta(m_idr)
    loss=loss+neg_loglike_iter
  }
  return(loss)
}

####################-----UPDATE: U-----#####################
#-----------Robust generalized scalar on tensor regression:
r_gsotr_update_u=function(h_y,b_theta,b_deriv,b_deriv2
                          ,U_dtr_list,TensorX_ti_train_list,Y_t,Mu_t
                          ,d,t,r,dimensions,alpha,stopping_cond
                          ,print,N,number_of_modes,calculate_r_gsotr_loss,learning_rate_init=1){
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  #getting factor matrix
  Udt_factormatrix=U_dtr_list[[d]][[t]]
  
  #newton search
  u_dt_r_prev=U_dtr_list[[d]][[t]][,r]
  U_temp=copy(U_dtr_list)
  #################loss prev calculation
  L_prev=calculate_r_gsotr_loss(h_y,b_theta,Y_t,TensorX_ti_train_list
                                ,U_dtr_list,Mu_t,d,t,number_of_modes,alpha,N)
  #################loss prev calculation ends
  iter=1
  epsilon=9999
  learning_rate=copy(learning_rate_init)
  while(epsilon>stopping_cond){
    #sum terms in gradient and hessian
    grad_sum_term=matrix(0,nrow=dimensions[d], ncol=1)
    hessian_sum_term=matrix(0,nrow=dimensions[d], ncol=dimensions[d])
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      W_itd=matricizedX_i@data%*%khatri_rao_product
      w_itd_r=matrix(W_itd[,r])
      T_itd_minusr=sum(diag(t(Udt_factormatrix[,-r])%*%W_itd[,-r]))
      
      m_idr=Mu_t[t]+T_itd_minusr+as.numeric(t(u_dt_r_prev)%*%w_itd_r)
      multip_term_i=-w_itd_r*Y_t[[t]][i]+w_itd_r*b_deriv(m_idr)
      expon_term_i=(h_y(Y_t,t,i)*exp(Y_t[[t]][i]*m_idr-b_theta(m_idr)))^alpha
      grad_sum_term_i=multip_term_i*expon_term_i
      grad_sum_term=grad_sum_term+grad_sum_term_i
      first_term_i=w_itd_r%*%t(w_itd_r)*b_deriv2(m_idr)
      second_term_i=(-1)*multip_term_i%*%t(multip_term_i)*alpha
      hessian_sum_term_i=first_term_i+second_term_i
      hessian_sum_term=hessian_sum_term+hessian_sum_term_i*expon_term_i
    }
    
    hessianinv_grad=tryCatch(
      expr = {
        chol2inv(chol(hessian_sum_term))%*%(grad_sum_term)
      },
      error = function(e){ 
        solve(hessian_sum_term)%*%(grad_sum_term)
      }
    )
    
    u_dt_r=u_dt_r_prev-learning_rate*hessianinv_grad
    
    U_temp[[d]][[t]][,r]=u_dt_r
    #################loss new calculation
    L_new=calculate_r_gsotr_loss(h_y,b_theta,Y_t,TensorX_ti_train_list
                                 ,U_temp,Mu_t,d,t,number_of_modes,alpha,N)
    #################loss new calculation ends
    update_iter=1
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      u_dt_r=u_dt_r_prev-learning_rate*hessianinv_grad
      
      U_temp[[d]][[t]][,r]=u_dt_r
      L_new=calculate_r_gsotr_loss(h_y,b_theta,Y_t,TensorX_ti_train_list
                                   ,U_temp,Mu_t,d,t,number_of_modes,alpha,N)
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=copy(learning_rate_init)
    
    L_prev=copy(L_new)
    
    norm_u_diff=norm(u_dt_r-u_dt_r_prev,type="2")
    
    epsilon=copy(norm_u_diff)
    u_dt_r_prev=copy(u_dt_r)
    if(print==T){
      print(paste0("******************** Iteration:",iter," ********************"))
      print("U vector:");print(as.numeric(u_dt_r))
      print(paste0("Norm of diff U: ",norm_u_diff))
      print("Gradient:");print(as.numeric(grad_sum_term))
    }
    if(iter==200){
      break
    }
    iter=iter+1
  }
  
  return(u_dt_r)
}

#-----------Generalized scalar on tensor regression:
gsotr_update_u=function(b_deriv,b_deriv2,U_dtr_list,TensorX_ti_train_list
                        ,Y_t,Mu_t,d,t,r,dimensions,stopping_cond,print
                        ,N,number_of_modes,calculate_gsotr_loss,learning_rate_init=1){
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  #getting factor matrix
  Udt_factormatrix=U_dtr_list[[d]][[t]]
  
  #newton search:
  #initialization:
  u_dt_r_prev=U_dtr_list[[d]][[t]][,r]
  U_temp=copy(U_dtr_list)
  #-----------------loss calculation
  L_prev=calculate_gsotr_loss(b_theta,Y_t,TensorX_ti_train_list
                              ,U_dtr_list,Mu_t,d,t,number_of_modes,N)
  
  iter=1
  epsilon=9999
  learning_rate=copy(learning_rate_init)
  while(epsilon>stopping_cond){
    #sum terms in gradient and hessian
    grad_sum_term=matrix(0,nrow=dimensions[d], ncol=1)
    hessian_sum_term=matrix(0,nrow=dimensions[d], ncol=dimensions[d])
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      W_itd=matricizedX_i@data%*%khatri_rao_product
      w_itd_r=matrix(W_itd[,r])
      T_itd_minusr=sum(diag(t(Udt_factormatrix[,-r])%*%W_itd[,-r]))
      
      m_idr=Mu_t[t]+T_itd_minusr+as.numeric(t(u_dt_r_prev)%*%w_itd_r)
      grad_sum_term_i=-w_itd_r*Y_t[[t]][i]+w_itd_r*b_deriv(m_idr)
      grad_sum_term=grad_sum_term+grad_sum_term_i
      
      hessian_sum_term_i=(w_itd_r%*%t(w_itd_r))*b_deriv2(m_idr)
      hessian_sum_term=hessian_sum_term+hessian_sum_term_i
    }
    
    hessianinv_grad=tryCatch(
      expr = {
        chol2inv(chol(hessian_sum_term))%*%(grad_sum_term)
      },
      error = function(e){ 
        solve(hessian_sum_term)%*%(grad_sum_term)
      }
    )
    
    u_dt_r=u_dt_r_prev-learning_rate*hessianinv_grad
    
    U_temp[[d]][[t]][,r]=u_dt_r
    #-----------------loss calculation
    L_new=calculate_gsotr_loss(b_theta,Y_t,TensorX_ti_train_list
                               ,U_temp,Mu_t,d,t,number_of_modes,N)
    
    update_iter=1
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      u_dt_r=u_dt_r_prev-learning_rate*hessianinv_grad
      
      U_temp[[d]][[t]][,r]=u_dt_r
      L_new=calculate_gsotr_loss(b_theta,Y_t,TensorX_ti_train_list
                                 ,U_temp,Mu_t,d,t,number_of_modes,N)
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=copy(learning_rate_init)
    L_prev=copy(L_new)
    norm_u_diff=norm(u_dt_r-u_dt_r_prev,type="2")
    
    epsilon=copy(norm_u_diff)#norm(f_u_dt_r,type="2")
    u_dt_r_prev=copy(u_dt_r)
    if(print==T){
      print(paste0("******************** Iteration:",iter," ********************"))
      print("U vector:");print(as.numeric(u_dt_r))
      print(paste0("Norm of diff U: ",norm_u_diff))
      print("Gradient:");print(as.numeric(grad_sum_term))
    }
    if(iter==200){
      break
    }
    iter=iter+1
  }
  
  return(u_dt_r)
}

####################-----UPDATE: MU-----#####################
r_gsotr_update_mu=function(h_y,b_theta,b_deriv,b_deriv2
                           ,Mu_t,number_of_datasets,number_of_modes
                           ,U_dtr_list,TensorX_ti_train_list
                           ,N,Y_t,t,stopping_cond,alpha,calculate_r_gsotr_loss,learning_rate_init=1){
  d=1
  #generation of K_d
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  mu_prev=Mu_t[t]
  
  #################loss prev calculation
  L_prev=calculate_r_gsotr_loss(h_y,b_theta,Y_t,TensorX_ti_train_list
                                ,U_dtr_list,mu_prev,d,t,number_of_modes,alpha,N)
  #################loss prev calculation ends
  iter=1
  epsilon=9999
  learning_rate=copy(learning_rate_init)
  while(epsilon>stopping_cond){
    deriv1=0
    deriv2=0
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      Z_it=t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product
      inner_product_i=sum(diag(Z_it))
      m_idr=mu_prev+inner_product_i
      multip_term_i=Y_t[[t]][i]-b_deriv(m_idr)
      expon_term_i=(h_y(Y_t,t,i)*exp(Y_t[[t]][i]*m_idr-b_theta(m_idr)))^(alpha)
      deriv1_i=-1*multip_term_i*expon_term_i
      if(is.nan(deriv1_i)){
        deriv1_i=0
      }
      deriv1=deriv1+deriv1_i
      
      first_term_i=b_deriv2(m_idr)*expon_term_i
      second_term_i=multip_term_i*multip_term_i*expon_term_i*alpha
      deriv2_i=first_term_i-second_term_i
      if(is.nan(deriv2_i)){
        deriv2_i=0
      }
      deriv2=deriv2+deriv2_i
    }
    #obtain new mu
    mu_new=mu_prev-learning_rate*(deriv1/deriv2)
    
    #################loss new calculation
    L_new=calculate_r_gsotr_loss(h_y,b_theta,Y_t,TensorX_ti_train_list
                                 ,U_dtr_list,mu_new,d,t,number_of_modes,alpha,N)
    #################loss new calculation ends
    update_iter=1
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      mu_new=mu_prev-learning_rate*(deriv1/deriv2)
      
      L_new=calculate_r_gsotr_loss(h_y,b_theta,Y_t,TensorX_ti_train_list
                                   ,U_dtr_list,mu_new,d,t,number_of_modes,alpha,N)
      
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=copy(learning_rate_init)
    
    L_prev=copy(L_new)
    
    epsilon=abs(mu_new-mu_prev)
    mu_prev=copy(mu_new)
    if(iter==200){
      break
    }
    iter=iter+1
  }
  Mu_t[t]=copy(mu_new)
  
  return(Mu_t)
}

gsotr_update_mu=function(b_theta,b_deriv,b_deriv2
                         ,Mu_t,number_of_datasets,number_of_modes
                         ,U_dtr_list,TensorX_ti_train_list,N,Y_t,t,stopping_cond
                         ,calculate_gsotr_loss,learning_rate_init=1){
  d=1
  #generation of K_d
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  mu_prev=Mu_t[t]
  #################loss prev calculation
  L_prev=calculate_gsotr_loss(b_theta,Y_t,TensorX_ti_train_list
                              ,U_dtr_list,Mu_t,d,t,number_of_modes,N)
  #################loss prev calculation ends
  iter=1
  epsilon=9999
  learning_rate=copy(learning_rate_init)
  while(epsilon>stopping_cond){
    #generation of Z_t_transpose
    deriv1=0
    deriv2=0
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      Z_it=t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product
      inner_product_i=sum(diag(Z_it))
      m_idr=mu_prev+inner_product_i
      deriv1_i=-Y_t[[t]][i]+b_deriv(m_idr)
      if(is.nan(deriv1_i)){
        deriv1_i=0
      }
      deriv1=deriv1+deriv1_i
      deriv2_i=b_deriv2(m_idr)
      if(is.nan(deriv2_i)){
        deriv2_i=0
      }
      deriv2=deriv2+deriv2_i
    }
    #obtain new mu
    mu_new=mu_prev-learning_rate*(deriv1/deriv2)
    
    #################loss new calculation
    L_new=calculate_gsotr_loss(b_theta,Y_t,TensorX_ti_train_list
                               ,U_dtr_list,mu_new,d,t,number_of_modes,N)
    #################loss new calculation ends
    update_iter=1
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      mu_new=mu_prev-learning_rate*(deriv1/deriv2)
      L_new=calculate_gsotr_loss(b_theta,Y_t,TensorX_ti_train_list
                                 ,U_dtr_list,mu_new,d,t,number_of_modes,N)
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=copy(learning_rate_init)
    L_prev=copy(L_new)
    
    epsilon=abs(mu_new-mu_prev)
    mu_prev=copy(mu_new)
    if(iter==200){
      break
    }
    iter=iter+1
  }
  
  Mu_t[t]=mu_new
  
  return(Mu_t)
}

####################-----LOCAL ITERATION-----#####################
r_gsotr_local_iteration=function(h_y,b_theta,b_deriv,b_deriv2,U_dtr_list,TensorX_ti_train_list
                                 ,TensorX_ti_test_list,t,Y_t,Y_t_test,Mu_t,N,N_test
                                 ,R,dimensions,stopping_cond_L,summary_table_L,total_iter
                                 ,local_iter_max,main_iter,alpha,distribution,is_scaled
                                 ,number_of_modes,number_of_datasets,prob_list,prob_list_test
                                 ,epsilon_list,epsilon_list_test,scale=1,calculate_r_gsotr_loss
                                 ,calculate_param_dev_tensor,U_dtr_groundtruth
                                 ,learning_rate_u=1,learning_rate_mu=1){
  local_iter=0
  L_prev=99999#100
  epsilon_L=99999
  while(epsilon_L>stopping_cond_L){
    error_count=0
    for(d in 1:number_of_modes){
      for(r in 1:R){
        u_error_flag=tryCatch({
          U_dtr_list[[d]][[t]][,r]=r_gsotr_update_u(h_y,b_theta,b_deriv,b_deriv2,U_dtr_list
                                                    ,TensorX_ti_train_list,Y_t,Mu_t,d,t,r,dimensions,alpha
                                                    ,stopping_cond=0.000001,print=F,N=N,number_of_modes
                                                    ,calculate_r_gsotr_loss,learning_rate_u)
        },error=function(e){
          return(1)
        })
        if(length(u_error_flag)==1){
          error_count=error_count+u_error_flag
        }
        rm(u_error_flag)
      }
    }
    
    #--------Update Mu_t
    Mu_t=r_gsotr_update_mu(h_y,b_theta,b_deriv,b_deriv2,Mu_t
                           ,number_of_datasets,number_of_modes,U_dtr_list
                           ,TensorX_ti_train_list,N,Y_t,t,stopping_cond=0.000001
                           ,alpha,calculate_r_gsotr_loss,learning_rate_mu)
    
    #--------Calculate Khatrirao product
    khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
    L=calculate_r_gsotr_loss(h_y,b_theta,Y_t,TensorX_ti_train_list
                             ,U_dtr_list,Mu_t,d,t,number_of_modes,alpha,N)
    # avg_reg_residual=reg_residual/N[t]
    epsilon_L=abs((L_prev-L)/L)#abs(L-L_prev)
    L_prev=copy(L)
    
    predictions_train_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_train_list
                                                 ,number_of_datasets,number_of_modes,N
                                                 ,distribution=distribution,t)
    predictions_test_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_test_list
                                                ,number_of_datasets,number_of_modes,N_test
                                                ,distribution=distribution,t)
    predictions_train=predictions_train_list[[1]]
    predictions_test=predictions_test_list[[1]]
    
    p_train=calculate_RMSE(Y_t[[t]],predictions_train)
    p_test=calculate_RMSE(Y_t_test[[t]],predictions_test)
    rmse_train=calculate_RMSE(scale*Y_t[[t]],scale*predictions_train)
    rmse_test=calculate_RMSE(scale*Y_t_test[[t]],scale*predictions_test)
    wmape_train=calculate_WMAPE(Y_t[[t]],predictions_train)
    wmape_test=calculate_WMAPE(Y_t_test[[t]],predictions_test)
    bias_train_scaled=calculate_bias(Y_t[[t]],predictions_train)
    bias_test_scaled=calculate_bias(Y_t_test[[t]],predictions_test)
    bias_train=calculate_bias(scale*Y_t[[t]],scale*predictions_train)
    bias_test=calculate_bias(scale*Y_t_test[[t]],scale*predictions_test)
    logarithmic_score_train=calculate_logscore(Y_t[[t]],predictions_train)
    logarithmic_score_test=calculate_logscore(Y_t_test[[t]],predictions_test)
    weightedrmse_train=calculate_weightedRMSE(Y_t[[t]],predictions_train)
    weightedrmse_test=calculate_weightedRMSE(Y_t_test[[t]],predictions_test)
    
    summary_table_L=rbind(summary_table_L,data.table(MainIter=main_iter,R=R,alpha=alpha
                                                     ,Site=t,LocalIter=local_iter
                                                     ,epsilon=epsilon_L,L=L
                                                     ,RMSE_train_scaled=p_train
                                                     ,RMSE_test_scaled=p_test
                                                     ,RMSE_train=rmse_train
                                                     ,RMSE_test=rmse_test
                                                     ,WMAPE_train=wmape_train
                                                     ,WMAPE_test=wmape_test
                                                     ,BIAS_train_scaled=bias_train_scaled
                                                     ,BIAS_test_scaled=bias_test_scaled
                                                     ,BIAS_train=bias_train
                                                     ,BIAS_test=bias_test
                                                     ,log_score_train=logarithmic_score_train
                                                     ,log_score_test=logarithmic_score_test
                                                     ,weightedRMSE_train=weightedrmse_train
                                                     ,weightedRMSE_test=weightedrmse_test
    )
    ,fill=TRUE)
    
    total_iter=total_iter+1
    local_iter=local_iter+1
    if(local_iter==local_iter_max){
      break
    }
  }
  
  return(list(summary_table_L,total_iter,L,U_dtr_list,Mu_t))
}

gsotr_local_iteration=function(b_theta,b_deriv,b_deriv2,U_dtr_list,TensorX_ti_train_list
                               ,TensorX_ti_test_list,t,Y_t,Y_t_test,Mu_t,N,N_test,R,dimensions
                               ,stopping_cond_L,summary_table_L,total_iter,local_iter_max
                               ,main_iter,distribution,is_scaled,number_of_modes,number_of_datasets
                               ,epsilon_list,epsilon_list_test,prob_list,prob_list_test,scale=1
                               ,calculate_gsotr_loss,calculate_param_dev_tensor,U_dtr_groundtruth
                               ,learning_rate_u=1,learning_rate_mu=1){
  local_iter=0
  L_prev=99999#100
  epsilon_L=99999
  while(epsilon_L>stopping_cond_L){
    #--------Update U
    error_count=0
    for(d in 1:number_of_modes){
      for(r in 1:R){
        u_error_flag=tryCatch({
          U_dtr_list[[d]][[t]][,r]=gsotr_update_u(b_deriv,b_deriv2,U_dtr_list
                                                  ,TensorX_ti_train_list,Y_t,Mu_t,d,t,r,dimensions
                                                  ,stopping_cond=0.000001,print=F,N=N
                                                  ,number_of_modes,calculate_gsotr_loss,learning_rate_u)
        },error=function(e){
          return(1)
        })
        if(length(u_error_flag)==1){
          error_count=error_count+u_error_flag
        }
        rm(u_error_flag)
      }
    }
    
    #--------Update Mu_t
    Mu_t=gsotr_update_mu(b_theta,b_deriv,b_deriv2,Mu_t,number_of_datasets
                         ,number_of_modes,U_dtr_list,TensorX_ti_train_list,N,Y_t,t
                         ,stopping_cond=0.000001,calculate_gsotr_loss,learning_rate_mu)
    
    #--------Calculate Khatrirao product
    khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
    L=calculate_gsotr_loss(b_theta,Y_t,TensorX_ti_train_list
                           ,U_dtr_list,Mu_t,d,t,number_of_modes,N)
    
    epsilon_L=abs((L_prev-L)/L)
    L_prev=copy(L)
    
    effective_param_iter=R*(sum(dimensions)-number_of_modes+1)
    aic_error_iter=2*L
    aic_param_iter=2*effective_param_iter
    aic_iter=aic_param_iter+aic_error_iter
    
    #BIC calculation iteration:
    bic_error_iter=2*L
    bic_param_iter=log(N[t])*effective_param_iter
    bic_iter=bic_param_iter+bic_error_iter
    
    
    predictions_train_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_train_list
                                                 ,number_of_datasets,number_of_modes,N
                                                 ,distribution=distribution,t)
    predictions_test_list=gsotr_get_predictions(U_dtr_list,Mu_t,TensorX_ti_test_list
                                                ,number_of_datasets,number_of_modes,N_test
                                                ,distribution=distribution,t)
    
    
    predictions_train=predictions_train_list[[1]]
    predictions_test=predictions_test_list[[1]]
    
    p_train=calculate_RMSE(Y_t[[t]],predictions_train)
    p_test=calculate_RMSE(Y_t_test[[t]],predictions_test)
    rmse_train=calculate_RMSE(scale*Y_t[[t]],scale*predictions_train)
    rmse_test=calculate_RMSE(scale*Y_t_test[[t]],scale*predictions_test)
    wmape_train=calculate_WMAPE(Y_t[[t]],predictions_train)
    wmape_test=calculate_WMAPE(Y_t_test[[t]],predictions_test)
    bias_train_scaled=calculate_bias(Y_t[[t]],predictions_train)
    bias_test_scaled=calculate_bias(Y_t_test[[t]],predictions_test)
    bias_train=calculate_bias(scale*Y_t[[t]],scale*predictions_train)
    bias_test=calculate_bias(scale*Y_t_test[[t]],scale*predictions_test)
    logarithmic_score_train=calculate_logscore(Y_t[[t]],predictions_train)
    logarithmic_score_test=calculate_logscore(Y_t_test[[t]],predictions_test)
    weightedrmse_train=calculate_weightedRMSE(Y_t[[t]],predictions_train)
    weightedrmse_test=calculate_weightedRMSE(Y_t_test[[t]],predictions_test)
    
    summary_table_L=rbind(summary_table_L,data.table(MainIter=main_iter,R=R
                                                     ,Site=t,LocalIter=local_iter
                                                     ,epsilon=epsilon_L,L=L
                                                     ,AIC=aic_iter
                                                     ,AIC_error=aic_error_iter
                                                     ,AIC_param=aic_param_iter
                                                     ,BIC=bic_iter
                                                     ,BIC_error=bic_error_iter
                                                     ,BIC_param=bic_param_iter
                                                     ,RMSE_train_scaled=p_train
                                                     ,RMSE_test_scaled=p_test
                                                     ,RMSE_train=rmse_train
                                                     ,RMSE_test=rmse_test
                                                     ,WMAPE_train=wmape_train
                                                     ,WMAPE_test=wmape_test
                                                     ,BIAS_train_scaled=bias_train_scaled
                                                     ,BIAS_test_scaled=bias_test_scaled
                                                     ,BIAS_train=bias_train
                                                     ,BIAS_test=bias_test
                                                     ,log_score_train=logarithmic_score_train
                                                     ,log_score_test=logarithmic_score_test
                                                     ,weightedRMSE_train=weightedrmse_train
                                                     ,weightedRMSE_test=weightedrmse_test
    )
    ,fill=TRUE)
    
    total_iter=total_iter+1
    local_iter=local_iter+1
    if(local_iter==local_iter_max){
      break
    }
  }
  
  effective_param=R*(sum(dimensions)-number_of_modes+1)
  aic_error=2*L
  aic_param=2*effective_param
  aic=aic_error+aic_param
  
  #BIC calculation iteration:
  bic_error=2*L
  bic_param=log(N[t])*effective_param
  bic=bic_param+bic_error
  
  return(list(summary_table_L,total_iter,L,U_dtr_list,Mu_t
              ,aic,aic_error,aic_param,bic,bic_error,bic_param))
}

####################----- DATA GENERATION-----#####################
####################----- > U -----#####################
sotr_generate_U_dtr=function(u_t_mean,u_t_sigma,dimensions
                             ,R,number_of_modes,number_of_datasets
                             ,seed){
  U_dtr_list=rep(list(vector("list",c(number_of_datasets))),number_of_modes)
  set.seed(seed)
  for(t in 1:number_of_datasets){
    for(d in 1:number_of_modes){
      for(r in 1:R){
        U_dtr_list[[d]][[t]]=cbind(U_dtr_list[[d]][[t]],rnorm(dimensions[d],mean = u_t_mean, sd=u_t_sigma))
      }
    }
  }
  return(U_dtr_list)
}

r_gsotr_generate_Y_t=function(sigma,mu,number_of_datasets,number_of_modes,U_dtr_list
                              ,TensorX_ti_list,N,seed,distribution,outlier_seed
                              ,outlier_q,outlier_mean,outlier_sd,add_outlier
                              ,dist_multip,outlier_lambda,poisson_true_intercept){
  p=vector("list",c(number_of_datasets))
  Y_t=vector("list",c(number_of_datasets))
  epsilon_t=vector("list",c(number_of_datasets))
  set.seed(seed)
  d=1
  for(t in 1:number_of_datasets){
    #--------Calculate Khatrirao product
    khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
    
    #generation of Z_t_transpose
    innerprod=numeric()
    outlier_sigma_input_list=c()
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_list[[t]][[i]],d)
      Z_it=t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product 
      innerprod=rbind(innerprod,sum(diag(Z_it))) 
    }
    
    #--add outliers:
    if(add_outlier){
      set.seed(outlier_seed)
      outlier_ind=sample(N[t],round(N[t]*outlier_q))
      outlier_indices=rep(0,N[t])#rbinom(N[t],1,outlier_q)
      outlier_indices[outlier_ind]=1
      #outlier_noise=rnorm(n=N[t],mean=outlier_mean,sd=outlier_sd)
      lambda=exp(poisson_multiplier*innerprod+poisson_true_intercept)#+outlier_indices*outlier_noise)
      lambda[outlier_ind]=lambda[outlier_ind]+lambda[outlier_ind]*outlier_lambda
      Y_t[[t]]=rpois(N[t],lambda)
      # # hist(lambda)
      # hist(Y_t[[t]])
      # boxplot(Y_t[[t]])
    }else{
      lambda=exp(poisson_multiplier*innerprod+poisson_true_intercept)
      Y_t[[t]]=rpois(N[t],lambda)
      # hist(Y_t[[t]])
      # hist(lambda)
    }
    p=c()
  }
  rm(matricizedX_i);rm(Z_it);rm(khatri_rao_product);rm(innerprod)
  return(list(Y_t,p,epsilon_t,lambda))
}

####################----- > X_it-----#####################
generate_X_it=function(tensorX_beta,rho,N,number_of_datasets,dimensions
                       ,seed,mean_vector_sigma,is_nonIID,IIDmean){
  TensorX_ti_list=vector("list",c(number_of_datasets))
  dim_X_prod=prod(dimensions) #vector length
  CovMatrix_X=matrix(NA, nrow=dim_X_prod, ncol=dim_X_prod)
  CovMatrix_X=outer(1:nrow(CovMatrix_X), 1:ncol(CovMatrix_X) , FUN=function(r,c) rho^(abs(r-c)))
  
  set.seed(seed)
  for(t in 1:number_of_datasets){
    if(is_nonIID==T){
      beta_t=rnorm(1,mean=0,sd=tensorX_beta)
      v_t=rnorm(dim_X_prod,mean=beta_t,sd=mean_vector_sigma)
    }else{
      v_t=rep(IIDmean,dim_X_prod)
    }
    TensorX_ti_vectorized=mvrnorm(n=N[t],mu=v_t,Sigma=CovMatrix_X)
    for(i in 1:N[t]){
      x_tensor=as.tensor(array(TensorX_ti_vectorized[i,],dim=dimensions)) 
      TensorX_ti_list[[t]][[i]]=x_tensor
    }
  }
  rm(CovMatrix_X);rm(x_tensor);rm(TensorX_ti_vectorized)
  return(TensorX_ti_list)
}

####################----- INITIALIZATION-GSOTR-----####################
sotr_initialize_U_dtr=function(u_t_mean,u_t_sigma,dimensions
                               ,R,number_of_modes,number_of_datasets,seed,type){
  #random initialization
  U_dtr_list=rep(list(vector("list",c(number_of_datasets))),number_of_modes)
  set.seed(seed)
  if(type=="random"){
    for(t in 1:number_of_datasets){
      for(d in 1:number_of_modes){
        for(r in 1:R[t]){
          U_dtr_list[[d]][[t]]=cbind(U_dtr_list[[d]][[t]],rnorm(dimensions[d],mean = u_t_mean, sd=u_t_sigma))
        }
      }
    }
  }
  return(U_dtr_list)
}

initialize_u_l2=function(u_t_mean,u_t_sigma,dimensions,init_tensor_R,R_decomp
                         ,number_of_modes,number_of_datasets
                         ,sotrl_generate_udtr_seed,hosvd_seed,type,TensorX_ti_list){
  U_dtr_list=rep(list(vector("list",c(number_of_datasets))),number_of_modes)
  
  for(t in 1:number_of_datasets){
    set.seed(t)
    selected_index=ceiling(runif(1,0,N[t]))
    tensor=TensorX_ti_list[[t]][[selected_index]]
    set.seed(hosvd_seed)
    hosvd_rank=min(dimensions)
    initial_cp_l2=hosvd(tensor,rep(R_decomp,number_of_modes))$U
    for(d in 1:number_of_modes){
      if(R_decomp>hosvd_rank){
        for(r in 1:(R_decomp-hosvd_rank)){
          initial_cp_l2[[d]]=cbind(initial_cp_l2[[d]],rnorm(dimensions[d],mean=u_t_mean,sd=u_t_sigma))
        }
      }
      U_dtr_list[[d]][[t]]=initial_cp_l2[[d]]
    }
  }
  
  return(U_dtr_list)
}

####################----- PERFORMANCE-----####################
gsotr_get_predictions=function(U_dtr_list,Mu_t,TensorX,number_of_datasets
                               ,number_of_modes,N,distribution,t){
  d=1
  #--------Calculate Khatrirao product
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  #generation of Z_t_transpose
  innerproduct=numeric()
  for(i in 1:N[t]){
    matricizedX_i=k_unfold(TensorX[[t]][[i]],d)
    Z_it=t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product 
    innerproduct=rbind(innerproduct,sum(diag(Z_it))) 
  }
  Predictions_t=exp(innerproduct+Mu_t[t])
  p=c()
  return(list(Predictions_t,p))
}

calculate_RMSE=function(actual,predicted){
  return(sqrt(sum((predicted-actual)^2)/length(actual)))
}

calculate_weightedRMSE=function(actual,predicted){
  return(sqrt(sum(((predicted-actual)^2)/predicted)/length(actual)))
}

calculate_lambdaRMSE=function(lambda,predicted){
  return(sqrt(sum(((predicted-lambda)^2))/length(lambda)))
}

calculate_lambdaweightedRMSE=function(lambda,predicted){
  return(sqrt(sum(((predicted-lambda)^2)/predicted)/length(lambda)))
}

calculate_logscore=function(actual,predicted){
  return(-1*sum(log(dpois(actual,predicted))))
}

calculate_WMAPE=function(actual,predicted){
  return(sum(abs(actual-predicted))/sum(abs(actual)))
}

calculate_accuracy=function(actual,predicted){
  return(sum(actual==predicted)/length(actual))
}

calculate_bias=function(actual,predicted){
  return(sum(actual-predicted)/length(actual))
}

calculate_RMSE_errorcorrected=function(actual,predicted,epsilon){
  return(sqrt(sum((actual-epsilon-predicted)^2)/length(actual)))
}

calculate_param_dev_tensor=function(groundtruth,estimate,number_of_modes,dimensions){
  gt_ten=reshape_factormat_to_tensor(groundtruth,number_of_modes,dimensions,R=ncol(groundtruth[[1]][[1]]),t=1)
  est_ten=reshape_factormat_to_tensor(estimate,number_of_modes,dimensions,R=ncol(estimate[[1]][[1]]),t=1)
  dev_norm=fnorm(gt_ten-est_ten)
  gt_norm=fnorm(gt_ten)
  est_norm=fnorm(est_ten)
  return(list(dev_norm,gt_norm,est_norm))
}

r_gsotr_add_iter_to_performance_table=function(performance_table,model_type,Y_t,predictions_train
                                               ,Y_t_test,predictions_test,summary_table_L_lastiters
                                               ,round,N,N_test,number_of_datasets,distribution
                                               ,class_train,class_test,R,alpha,t,p_t,p_t_test
                                               ,epsilon,epsilon_test,scale=1,lambda,lambda_test){
  
  rmse_train=calculate_RMSE(Y_t,predictions_train)
  rmse_test=calculate_RMSE(Y_t_test,predictions_test)
  wmape_train=calculate_WMAPE(Y_t,predictions_train)
  wmape_test=calculate_WMAPE(Y_t_test,predictions_test)
  bias_train=calculate_bias(Y_t,predictions_train)
  bias_test=calculate_bias(Y_t_test,predictions_test)
  logarithmic_score_train=calculate_logscore(Y_t,predictions_train)
  logarithmic_score_test=calculate_logscore(Y_t_test,predictions_test)
  weightedrmse_train=calculate_weightedRMSE(Y_t,predictions_train)
  weightedrmse_test=calculate_weightedRMSE(Y_t_test,predictions_test)
  lambdarmse_train=calculate_lambdaRMSE(lambda,predictions_train)
  lambdarmse_test=calculate_lambdaRMSE(lambda_test,predictions_test)
  lambdaweightedrmse_train=calculate_lambdaweightedRMSE(lambda,predictions_train)
  lambdaweightedrmse_test=calculate_lambdaweightedRMSE(lambda_test,predictions_test)
  # 
  summary_table_L_lastiters_t=summary_table_L_lastiters[Site==t]
  performance_table=rbind(performance_table
                          ,data.table(Model=model_type,Round=round,Site=t,R=R
                                      ,alpha=alpha,N_train=N[t],N_test=N_test[t]
                                      ,RMSE_train=rmse_train,RMSE_test=rmse_test
                                      ,WMAPE_train=wmape_train,WMAPE_test=wmape_test
                                      ,BIAS_train=bias_train,BIAS_test=bias_test
                                      ,log_score_train=logarithmic_score_train,log_score_test=logarithmic_score_test
                                      ,weightedRMSE_train=weightedrmse_train,weightedRMSE_test=weightedrmse_test
                                      ,lambdaRMSE_train=lambdarmse_train,lambdaRMSE_test=lambdarmse_test
                                      ,lambdaweightedRMSE_train=lambdaweightedrmse_train,lambdaweightedRMSE_test=lambdaweightedrmse_test
                                      ,L=summary_table_L_lastiters_t$L
                                      ,neg_loglike=summary_table_L_lastiters_t$neg_loglike)
                          ,fill=TRUE)
  return(performance_table)
}
