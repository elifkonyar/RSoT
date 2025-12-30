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


#gaussian: type: "mean" "variance"
gaussian_theta_vec=function(mu,sigma){
  return(list(mu/(sigma^2),-1/(2*(sigma^2))))
}

gaussian_h_mean=function(Y_t,t,i,sigma){
  return(exp((-1*(Y_t[[t]][i]^2)/(sigma^2))/2)/(sqrt(2*pi*sigma^2)))
}

gaussian_b_mean=function(theta){
  return(-(theta[[1]]^2)/(4*theta[[2]]))
}

gaussian_b_deriv_mean=function(theta){
  return(-theta[[1]]/(2*theta[[2]]))
}

gaussian_b_deriv2_mean=function(theta){
  return(-1/(2*theta[[2]]))
}

gaussian_eta_sigma=function(sigma){
  return(-1/(2*sigma^2))
}

gaussian_eta_deriv_sigma=function(sigma){
  return(sigma^(-3))
}

gaussian_eta_deriv2_sigma=function(sigma){
  return(-3*sigma^(-4))
}

gaussian_b_sigma=function(sigma){
  return(log(sigma))
}

gaussian_b_deriv_sigma=function(sigma){
  return(1/sigma)
}

gaussian_b_deriv2_sigma=function(sigma){
  return(-1/(sigma^2))
}

####################-----LOSS CALCULATION-----#####################
calculate_r_gsotr_loss=function(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                ,U_dtr_list,Mu_t,d,t,number_of_modes,alpha,N){
  loss=0
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  for(i in 1:N[t]){
    matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
    inner_prod=sum(diag(t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product))
    m_idr=Mu_t[t]+inner_prod
    theta_idr=theta_vec(m_idr,sigma)
    expon_term_i=(h_y(Y_t,t,i,sigma)*exp(Y_t[[t]][i]*theta_idr[[1]]-b_theta(theta_idr)))^(alpha)
    loss_i=(1-expon_term_i)/alpha
    loss=loss+loss_i
  }
  return(loss)
}

calculate_gsotr_loss=function(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                              ,U_dtr_list,Mu_t,d,t,number_of_modes,N){
  loss=0
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  for(i in 1:N[t]){
    matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
    inner_prod=sum(diag(t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product))
    m_idr=Mu_t[t]+inner_prod
    theta_idr=theta_vec(m_idr,sigma)
    neg_loglike_iter=-Y_t[[t]][i]*theta_idr[[1]]+b_theta(theta_idr)
    loss=loss+neg_loglike_iter
  }
  return(loss)
}

####################-----UPDATE: U-----#####################
#-----------Robust generalized scalar on tensor regression:
r_gsotr_update_u=function(h_y,b_theta,b_deriv,b_deriv2,theta_vec,sigma
                          ,U_dtr_list,TensorX_ti_train_list,Y_t,Mu_t
                          ,d,t,r,dimensions,alpha,stopping_cond
                          ,print,N,number_of_modes,calculate_r_gsotr_loss,u_dt_r_GSOT){
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  #getting factor matrix
  Udt_factormatrix=U_dtr_list[[d]][[t]]
  
  #newton search
  u_dt_r_prev=U_dtr_list[[d]][[t]][,r]
  U_temp=copy(U_dtr_list)
  #################loss prev calculation
  L_prev=calculate_r_gsotr_loss(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                ,U_dtr_list,Mu_t,d,t,number_of_modes,alpha,N)
  #################loss prev calculation ends
  
  iter=1
  epsilon=9999
  learning_rate=1
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
      theta_idr=theta_vec(m_idr,sigma)
      
      w_itd_r_tilde=w_itd_r/(sigma^2)
      multip_term_i=-w_itd_r_tilde*Y_t[[t]][i]+w_itd_r_tilde*b_deriv(theta_idr)
      expon_term_i=(h_y(Y_t,t,i,sigma)*exp(Y_t[[t]][i]*theta_idr[[1]]-b_theta(theta_idr)))^alpha
      grad_sum_term_i=multip_term_i*expon_term_i
      grad_sum_term=grad_sum_term+grad_sum_term_i
      first_term_i=w_itd_r_tilde%*%t(w_itd_r_tilde)*b_deriv2(theta_idr)
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
    L_new=calculate_r_gsotr_loss(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                 ,U_temp,Mu_t,d,t,number_of_modes,alpha,N)
    #################loss new calculation ends
    
    update_iter=1
    if(update_iter==1 & L_new>L_prev){
      learning_rate=1
    }
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      u_dt_r=u_dt_r_prev-learning_rate*hessianinv_grad
      
      U_temp[[d]][[t]][,r]=u_dt_r
      L_new=calculate_r_gsotr_loss(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                   ,U_temp,Mu_t,d,t,number_of_modes,alpha,N)
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=1
    
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
  
  #find c:
  c_dt_r=log(N)/(N*abs(u_dt_r_GSOT))
  u_dt_r_lasso=copy(u_dt_r)
  #soft thresholding
  indicator_zero=F
  for(d_soft in 1:length(u_dt_r)){
    if(u_dt_r[d_soft]>c_dt_r[d_soft]){
      u_dt_r_lasso[d_soft]=u_dt_r[d_soft]-c_dt_r[d_soft]
    }else if(u_dt_r[d_soft]<(-1*c_dt_r[d_soft])){
      u_dt_r_lasso[d_soft]=u_dt_r[d_soft]+c_dt_r[d_soft]
    }else{
      u_dt_r_lasso[d_soft]=0
    }
    if(sum(u_dt_r_lasso==0)==(length(u_dt_r)-1)){
      u_return=copy(u_dt_r_lasso)
      indicator_zero=T
    }
  }
  
  if(indicator_zero==T){
    return(u_return)
  }else{
    return(u_dt_r_lasso)
  }
}

gsotr_update_u_old=function(b_deriv,b_deriv2,theta_vec,sigma
                            ,U_dtr_list,TensorX_ti_train_list,Y_t,Mu_t
                            ,d,t,r,dimensions,stopping_cond,print
                            ,N,number_of_modes,calculate_gsotr_loss){
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  #getting factor matrix
  Udt_factormatrix=U_dtr_list[[d]][[t]]
  
  #newton search:
  #initialization:
  u_dt_r_prev=U_dtr_list[[d]][[t]][,r]
  U_temp=copy(U_dtr_list)
  #-----------------loss calculation
  L_prev=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                              ,U_dtr_list,Mu_t,d,t,number_of_modes,N)
  
  iter=1
  epsilon=9999
  learning_rate=1
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
      theta_idr=theta_vec(m_idr,sigma)
      w_itd_r_tilde=w_itd_r/(sigma^2)
      grad_sum_term_i=-w_itd_r_tilde*Y_t[[t]][i]+w_itd_r_tilde*b_deriv(theta_idr)
      grad_sum_term=grad_sum_term+grad_sum_term_i
      
      hessian_sum_term_i=(w_itd_r_tilde%*%t(w_itd_r_tilde))*b_deriv2(theta_idr)
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
    L_new=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                               ,U_temp,Mu_t,d,t,number_of_modes,N)
    
    update_iter=1
    if(update_iter==1 & L_new>L_prev){
      learning_rate=1
    }
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      u_dt_r=u_dt_r_prev-learning_rate*hessianinv_grad
      
      U_temp[[d]][[t]][,r]=u_dt_r
      L_new=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                 ,U_temp,Mu_t,d,t,number_of_modes,N)
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=1
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
gsotr_update_u=function(b_deriv,b_deriv2,theta_vec,sigma
                        ,U_dtr_list,TensorX_ti_train_list,Y_t,Mu_t
                        ,d,t,r,dimensions,stopping_cond,print
                        ,N,number_of_modes,calculate_gsotr_loss){
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  #getting factor matrix
  Udt_factormatrix=U_dtr_list[[d]][[t]]
  
  #newton search:
  #initialization:
  u_dt_r_prev=U_dtr_list[[d]][[t]][,r]
  U_temp=copy(U_dtr_list)
  #-----------------loss calculation
  L_prev=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                              ,U_dtr_list,Mu_t,d,t,number_of_modes,N)
  
  iter=1
  epsilon=9999
  learning_rate=1
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
      theta_idr=theta_vec(m_idr,sigma)
      w_itd_r_tilde=w_itd_r/(sigma^2)
      grad_sum_term_i=-w_itd_r_tilde*Y_t[[t]][i]+w_itd_r_tilde*b_deriv(theta_idr)
      grad_sum_term=grad_sum_term+grad_sum_term_i
      
      hessian_sum_term_i=(w_itd_r_tilde%*%t(w_itd_r_tilde))*b_deriv2(theta_idr)
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
    L_new=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                               ,U_temp,Mu_t,d,t,number_of_modes,N)
    
    update_iter=1
    if(update_iter==1 & L_new>L_prev){
      learning_rate=1
    }
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      u_dt_r=u_dt_r_prev-learning_rate*hessianinv_grad
      
      U_temp[[d]][[t]][,r]=u_dt_r
      L_new=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                 ,U_temp,Mu_t,d,t,number_of_modes,N)
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=1
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

####################-----UPDATE: MU-----#####################
r_gsotr_update_mu=function(h_y,b_theta,b_deriv,b_deriv2,theta_vec,sigma
                           ,Mu_t,number_of_datasets,number_of_modes
                           ,U_dtr_list,TensorX_ti_train_list
                           ,N,Y_t,t,stopping_cond,alpha,calculate_r_gsotr_loss){
  d=1
  #generation of K_d
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  mu_prev=Mu_t[t]
  
  #################loss prev calculation
  L_prev=calculate_r_gsotr_loss(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                ,U_dtr_list,mu_prev,d,t,number_of_modes,alpha,N)
  #################loss prev calculation ends
  iter=1
  epsilon=9999
  learning_rate=1
  while(epsilon>stopping_cond){
    deriv1=0
    deriv2=0
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      Z_it=t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product
      inner_product_i=sum(diag(Z_it))
      m_idr=mu_prev+inner_product_i
      theta_idr=theta_vec(m_idr,sigma)
      multip_term_i=Y_t[[t]][i]-b_deriv(theta_idr)
      expon_term_i=(h_y(Y_t,t,i,sigma)*exp(Y_t[[t]][i]*theta_idr[[1]]-b_theta(theta_idr)))^(alpha)
      deriv1_i=-1*multip_term_i*expon_term_i
      deriv1=deriv1+deriv1_i
      
      first_term_i=b_deriv2(theta_idr)*expon_term_i
      second_term_i=multip_term_i*multip_term_i*expon_term_i*alpha
      deriv2_i=first_term_i-second_term_i
      deriv2=deriv2+deriv2_i
    }
    deriv1=deriv1/(sigma^2)
    deriv2=deriv2/(sigma^4)
    #obtain new mu
    mu_new=mu_prev-learning_rate*(deriv1/deriv2)
    
    #################loss new calculation
    L_new=calculate_r_gsotr_loss(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                 ,U_dtr_list,mu_new,d,t,number_of_modes,alpha,N)
    #################loss new calculation ends
    
    update_iter=1
    if(update_iter==1 & L_new>L_prev){
      learning_rate=1
    }
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      mu_new=mu_prev-learning_rate*(deriv1/deriv2)
      
      L_new=calculate_r_gsotr_loss(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                   ,U_dtr_list,mu_new,d,t,number_of_modes,alpha,N)
      
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=1
    
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

gsotr_update_mu=function(b_theta,b_deriv,b_deriv2,theta_vec,sigma
                         ,Mu_t,number_of_datasets,number_of_modes
                         ,U_dtr_list,TensorX_ti_train_list,N,Y_t,t,stopping_cond
                         ,calculate_gsotr_loss){
  d=1
  #generation of K_d
  khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
  
  mu_prev=Mu_t[t]
  #################loss prev calculation
  L_prev=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                              ,U_dtr_list,Mu_t,d,t,number_of_modes,N)
  #################loss prev calculation ends
  
  iter=1
  epsilon=9999
  learning_rate=1
  while(epsilon>stopping_cond){
    #generation of Z_t_transpose
    deriv1=0
    deriv2=0
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      Z_it=t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product
      inner_product_i=sum(diag(Z_it))
      m_idr=mu_prev+inner_product_i
      theta_idr=theta_vec(m_idr,sigma)
      deriv1=deriv1+(-Y_t[[t]][i]+b_deriv(theta_idr))
      deriv2=deriv2+b_deriv2(theta_idr)
    }
    deriv1=deriv1/(sigma^2)
    deriv2=deriv2/(sigma^4)
    #obtain new mu
    mu_new=mu_prev-learning_rate*(deriv1/deriv2)
    
    #################loss new calculation
    L_new=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                               ,U_dtr_list,mu_new,d,t,number_of_modes,N)
    #################loss new calculation ends
    update_iter=1
    if(update_iter==1 & L_new>L_prev){
      learning_rate=1
    }
    while(L_new>L_prev){
      learning_rate=learning_rate*0.5
      mu_new=mu_prev-learning_rate*(deriv1/deriv2)
      L_new=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                                 ,U_dtr_list,mu_new,d,t,number_of_modes,N)
      if(update_iter==30){
        break
      }
      update_iter=update_iter+1
    }
    learning_rate=1
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
r_gsotr_local_iteration=function(h_y,b_theta,b_deriv,b_deriv2,theta_vec,sigma,b_sigma
                                 ,b_deriv_sigma,b_deriv2_sigma,eta_sigma,eta_deriv_sigma
                                 ,eta_deriv2_sigma,U_dtr_list,TensorX_ti_train_list
                                 ,TensorX_ti_test_list,t,Y_t,Y_t_test,Mu_t,N,N_test
                                 ,R,dimensions,stopping_cond_L,summary_table_L,total_iter
                                 ,local_iter_max,main_iter,alpha,distribution,is_scaled
                                 ,number_of_modes,number_of_datasets,prob_list,prob_list_test
                                 ,epsilon_list,epsilon_list_test,scale=1
                                 ,calculate_r_gsotr_loss,U_dtr_list_all){
  local_iter=0
  L_prev=99999
  epsilon_L=99999
  
  while(epsilon_L>stopping_cond_L){
    #--------Update U
    for(d in 1:number_of_modes){
      for(r in 1:R){
        u_dt_r_GSOT=U_dtr_list_all[[which(rank_list==R)]][[d]][[1]][,r]
        U_dtr_list[[d]][[t]][,r]=r_gsotr_update_u(h_y,b_theta,b_deriv,b_deriv2,theta_vec,sigma,U_dtr_list
                                                  ,TensorX_ti_train_list,Y_t,Mu_t,d,t,r,dimensions,alpha
                                                  ,stopping_cond=0.000001,print=F,N=N,number_of_modes
                                                  ,calculate_r_gsotr_loss,u_dt_r_GSOT)
      }
    }

    #--------Update Mu_t
    Mu_t=r_gsotr_update_mu(h_y,b_theta,b_deriv,b_deriv2,theta_vec,sigma,Mu_t
                           ,number_of_datasets,number_of_modes,U_dtr_list
                           ,TensorX_ti_train_list,N,Y_t,t,stopping_cond=0.000001
                           ,alpha,calculate_r_gsotr_loss)
    
    #--------Calculate Khatrirao product
    khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
    
    #--------Objective Function
    reg_residual=0
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      inner_prod=sum(diag(t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product))
      m_idr=Mu_t[t]+inner_prod
      reg_residual=reg_residual+(Y_t[[t]][i]-m_idr)^2
    }
    L=calculate_r_gsotr_loss(h_y,b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                             ,U_dtr_list,Mu_t,d,t,number_of_modes,alpha,N)
    avg_reg_residual=reg_residual/N[t]
    epsilon_L=abs((L_prev-L)/L)
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
    mape_train=calculate_MAPE(Y_t[[t]],predictions_train)
    mape_test=calculate_MAPE(Y_t_test[[t]],predictions_test)
    bias_train_scaled=calculate_bias(Y_t[[t]],predictions_train)
    bias_test_scaled=calculate_bias(Y_t_test[[t]],predictions_test)
    bias_train=calculate_bias(scale*Y_t[[t]],scale*predictions_train)
    bias_test=calculate_bias(scale*Y_t_test[[t]],scale*predictions_test)
    rmse_corrected_train=calculate_RMSE_errorcorrected(scale*Y_t[[t]],scale*predictions_train,epsilon_list[[t]])
    rmse_corrected_test=calculate_RMSE_errorcorrected(scale*Y_t_test[[t]],scale*predictions_test,epsilon_list_test[[t]])
    
    summary_table_L=rbind(summary_table_L,data.table(MainIter=main_iter,R=R,alpha=alpha
                                                     ,Site=t,LocalIter=local_iter
                                                     ,epsilon=epsilon_L,L=L
                                                     ,RMSE_train=rmse_train
                                                     ,RMSE_test=rmse_test
                                                     ,WMAPE_train=wmape_train
                                                     ,WMAPE_test=wmape_test
                                                     ,MAPE_train=mape_train
                                                     ,MAPE_test=mape_test
                                                     ,BIAS_train=bias_train
                                                     ,BIAS_test=bias_test
                                                     ,RMSE_corrected_train=rmse_corrected_train
                                                     ,RMSE_corrected_test=rmse_corrected_test)
                          ,fill=T)

    total_iter=total_iter+1
    local_iter=local_iter+1
    if(local_iter==local_iter_max){
      break
    }
  }
  
  return(list(summary_table_L,total_iter,L,U_dtr_list,Mu_t,sigma))
}

gsotr_local_iteration=function(b_theta,b_deriv,b_deriv2,theta_vec,sigma,eta_deriv_sigma,eta_deriv2_sigma
                               ,b_deriv_sigma,b_deriv2_sigma,U_dtr_list,TensorX_ti_train_list
                               ,TensorX_ti_test_list,t,Y_t,Y_t_test,Mu_t,N,N_test,R,dimensions
                               ,stopping_cond_L,summary_table_L,total_iter,local_iter_max
                               ,main_iter,distribution,is_scaled,number_of_modes,number_of_datasets
                               ,epsilon_list,epsilon_list_test,prob_list,prob_list_test,scale=1
                               ,calculate_gsotr_loss){
  local_iter=0
  L_prev=99999
  epsilon_L=99999
  while(epsilon_L>stopping_cond_L){
    #--------Update U
    error_count=0
    for(d in 1:number_of_modes){
      for(r in 1:R){
        u_error_flag=tryCatch({
          U_dtr_list[[d]][[t]][,r]=gsotr_update_u(b_deriv,b_deriv2,theta_vec,sigma,U_dtr_list
                                                  ,TensorX_ti_train_list,Y_t,Mu_t,d,t,r,dimensions
                                                  ,stopping_cond=0.000001,print=F,N=N
                                                  ,number_of_modes,calculate_gsotr_loss)
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
    Mu_t=gsotr_update_mu(b_theta,b_deriv,b_deriv2,theta_vec,sigma,Mu_t,number_of_datasets
                         ,number_of_modes,U_dtr_list,TensorX_ti_train_list,N,Y_t,t
                         ,stopping_cond=0.000001,calculate_gsotr_loss)
    
    #--------Calculate Khatrirao product
    khatri_rao_product=calculate_khatrirao_product(number_of_modes,U_dtr_list,d,t)
    
    #--------Objective Function
    #--loglikelihood:
    reg_residual=0
    for(i in 1:N[t]){
      matricizedX_i=k_unfold(TensorX_ti_train_list[[t]][[i]],d)
      inner_prod=sum(diag(t(U_dtr_list[[d]][[t]])%*%matricizedX_i@data%*%khatri_rao_product))
      reg_residual=reg_residual+(Y_t[[t]][i]-Mu_t[t]-inner_prod)^2
    }
    L=calculate_gsotr_loss(b_theta,theta_vec,sigma,Y_t,TensorX_ti_train_list
                           ,U_dtr_list,Mu_t,d,t,number_of_modes,N)
    avg_reg_residual=reg_residual/(N[t])
    
    epsilon_L=abs((L_prev-L)/L)
    L_prev=copy(L)
    
    #AIC calculation iteration:
    if(distribution=="gaussian"){
      effective_param_iter=R*(sum(dimensions)-number_of_modes+1)
      aic_error_iter=log(avg_reg_residual)*N[t]
      aic_param_iter=2*effective_param_iter
      aic_iter=aic_error_iter+aic_param_iter
      
      #BIC calculation iteration:
      bic_error_iter=log(avg_reg_residual)*N[t]
      bic_param_iter=log(N[t])*effective_param_iter
      bic_iter=bic_param_iter+bic_error_iter
    }else{
      effective_param_iter=R*(sum(dimensions)-number_of_modes+1)
      aic_error_iter=2*L
      aic_param_iter=2*effective_param_iter
      aic_iter=aic_param_iter+aic_error_iter
      
      #BIC calculation iteration:
      bic_error_iter=2*L
      bic_param_iter=log(N[t])*effective_param_iter
      bic_iter=bic_param_iter+bic_error_iter
    }
    
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
    mape_train=calculate_MAPE(Y_t[[t]],predictions_train)
    mape_test=calculate_MAPE(Y_t_test[[t]],predictions_test)
    bias_train_scaled=calculate_bias(Y_t[[t]],predictions_train)
    bias_test_scaled=calculate_bias(Y_t_test[[t]],predictions_test)
    bias_train=calculate_bias(scale*Y_t[[t]],scale*predictions_train)
    bias_test=calculate_bias(scale*Y_t_test[[t]],scale*predictions_test)
    rmse_corrected_train=calculate_RMSE_errorcorrected(scale*Y_t[[t]],scale*predictions_train,epsilon_list[[t]])
    rmse_corrected_test=calculate_RMSE_errorcorrected(scale*Y_t_test[[t]],scale*predictions_test,epsilon_list_test[[t]])
    
    summary_table_L=rbind(summary_table_L,data.table(MainIter=main_iter,R=R
                                                     ,Site=t,LocalIter=local_iter
                                                     ,epsilon=epsilon_L,L=L
                                                     ,AIC=aic_iter
                                                     ,AIC_error=aic_error_iter
                                                     ,AIC_param=aic_param_iter
                                                     ,BIC=bic_iter
                                                     ,BIC_error=bic_error_iter
                                                     ,BIC_param=bic_param_iter
                                                     ,RMSE_train=rmse_train
                                                     ,RMSE_test=rmse_test
                                                     ,WMAPE_train=wmape_train
                                                     ,WMAPE_test=wmape_test
                                                     ,MAPE_train=mape_train
                                                     ,MAPE_test=mape_test
                                                     ,BIAS_train=bias_train
                                                     ,BIAS_test=bias_test
                                                     ,RMSE_corrected_train=rmse_corrected_train
                                                     ,RMSE_corrected_test=rmse_corrected_test)
                          ,fill=T)

    total_iter=total_iter+1
    local_iter=local_iter+1
    if(local_iter==local_iter_max){
      break
    }
  }
  
  #AIC calculation:
  if(distribution=="gaussian"){
    effective_param=R*(sum(dimensions)-number_of_modes+1)
    aic_error=log(avg_reg_residual)*N[t]
    aic_param=2*effective_param
    aic=aic_error+aic_param
    
    #BIC calculation iteration:
    bic_error=log(avg_reg_residual)*N[t]
    bic_param=log(N[t])*effective_param
    bic=bic_param+bic_error
  }else{
    effective_param=R*(sum(dimensions)-number_of_modes+1)
    aic_error=2*L
    aic_param=2*effective_param
    aic=aic_error+aic_param
    
    #BIC calculation iteration:
    bic_error=2*L
    bic_param=log(N[t])*effective_param
    bic=bic_param+bic_error
  }
  
  return(list(summary_table_L,total_iter,L,U_dtr_list,Mu_t,sigma
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

sotr_generate_B_sparse=function(u_t_mean,u_t_sigma,dimensions
                                ,R,number_of_modes,number_of_datasets
                                ,seed){
  zero_mat=matrix(0,nrow=dimensions[1],ncol=dimensions[2])
  for(i in 6:10){
    for(j in 6:10){
      zero_mat[i,j]=10
    }
  }
  u_tensor=as.tensor(array(cbind(zero_mat,zero_mat),dim=dimensions))
  
  return(u_tensor)
}

####################----- > Y_t-----#####################
r_gsotr_generate_Y_t=function(sigma,mu,number_of_datasets,number_of_modes
                              ,U_dtr_list,TensorX_ti_list,N,seed,distribution
                              ,outlier_seed,outlier_q,outlier_mean,outlier_sd
                              ,add_outlier,bernoulli_multiplier,poisson_multiplier){
  Y_t=vector("list",c(number_of_datasets))
  epsilon_t=vector("list",c(number_of_datasets))
  set.seed(seed)
  d=1
  for(t in 1:number_of_datasets){
    #--------Calculate Khatrirao product
    #generation of Z_t_transpose
    innerprod=numeric()
    outlier_sigma_input_list=c()
    for(i in 1:N[t]){
      innerprod=rbind(innerprod,rTensor::innerProd(B_groundtruth,TensorX_ti_list[[t]][[i]])) 
    }
    epsilon=rnorm(N[t],mu,sigma)
    epsilon_t[[t]]=epsilon
    #--y values:
    Y_t[[t]]=innerprod+epsilon
    #--probability values: NA (for bernoulli)
    p=c()
    #--add outliers:
    if(add_outlier){
      set.seed(outlier_seed)
      outlier_ind=sample(N[t],round(N[t]*outlier_q))
      outlier_indices=rep(0,N[t])
      outlier_indices[outlier_ind]=1
      
      outlier_noise=rnorm(n=N[t],mean=outlier_mean,sd=outlier_sd)
      Y_t[[t]]=Y_t[[t]]+sign(Y_t[[t]])*outlier_indices*Y_t[[t]]*outlier_mean
      outlier_noise_return=outlier_indices*outlier_noise
    }else{
      outlier_noise_return=NA
    }
    
  }
  return(list(Y_t,p,epsilon_t,innerprod,epsilon,outlier_noise_return))
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

hadamard_list=function (L) 
{
  isvecORmat <- function(x) {
    is.matrix(x) || is.vector(x)
  }
  stopifnot(all(unlist(lapply(L, isvecORmat))))
  retmat <- L[[1]]
  if(length(L)>1){
    for (i in 2:length(L)) {
      retmat <- retmat * L[[i]]
    }
  }
  retmat
}

khatri_rao_list=function (L, reverse = FALSE) 
{
  stopifnot(all(unlist(lapply(L, is.matrix))))
  ncols <- unlist(lapply(L, ncol))
  stopifnot(length(unique(ncols)) == 1)
  ncols <- ncols[1]
  nrows <- unlist(lapply(L, nrow))
  retmat <- matrix(0, nrow = prod(nrows), ncol = ncols)
  if (reverse) 
    L <- rev(L)
  for (j in 1:ncols) {
    Lj <- lapply(L, function(x) x[, j])
    retmat[, j] <- kronecker_list(Lj)
  }
  retmat
}

kronecker_list=function (L) 
{
  isvecORmat <- function(x) {
    is.matrix(x) || is.vector(x)
  }
  stopifnot(all(unlist(lapply(L, isvecORmat))))
  retmat <- L[[1]]
  if(length(L)>1){
    for (i in 2:length(L)) {
      retmat <- kronecker(retmat, L[[i]])
    }
  }
  retmat
}

cp=function (tnsr, num_components = NULL, max_iter = 25, tol = 1e-05) 
{
  if (is.null(num_components)) 
    stop("num_components must be specified")
  stopifnot(is(tnsr, "Tensor"))
  num_modes <- tnsr@num_modes
  modes <- tnsr@modes
  U_list <- vector("list", num_modes)
  unfolded_mat <- vector("list", num_modes)
  tnsr_norm <- fnorm(tnsr)
  for (m in 1:num_modes) {
    unfolded_mat[[m]] <- rs_unfold(tnsr, m = m)@data
    U_list[[m]] <- matrix(rnorm(modes[m] * num_components), 
                          nrow = modes[m], ncol = num_components)
  }
  est <- tnsr
  curr_iter <- 1
  converged <- FALSE
  fnorm_resid <- rep(0, max_iter)
  CHECK_CONV <- function(est) {
    curr_resid <- fnorm(est - tnsr)
    fnorm_resid[curr_iter] <<- curr_resid
    if (curr_iter == 1) 
      return(FALSE)
    if (abs(curr_resid - fnorm_resid[curr_iter - 1])/tnsr_norm < 
        tol) 
      return(TRUE)
    else {
      return(FALSE)
    }
  }
  pb <- txtProgressBar(min = 0, max = max_iter, style = 3)
  norm_vec <- function(vec) {
    norm(as.matrix(vec))
  }
  while ((curr_iter < max_iter) && (!converged)) {
    setTxtProgressBar(pb, curr_iter)
    for (m in 1:num_modes) {
      V <- hadamard_list(lapply(U_list[-m], function(x) {
        t(x) %*% x
      }))
      V_inv <- solve(V)
      tmp <- unfolded_mat[[m]] %*% khatri_rao_list(U_list[-m], 
                                                   reverse = TRUE) %*% V_inv
      # lambdas <- apply(tmp, 2, norm_vec)
      # U_list[[m]] <- sweep(tmp, 2, lambdas, "/")
      Z <- .superdiagonal_tensor(num_modes = num_modes, 
                                 len = num_components, elements = 1L)
      est <- ttl(Z, U_list, ms = 1:num_modes)
    }
    if (CHECK_CONV(est)) {
      converged <- TRUE
      setTxtProgressBar(pb, max_iter)
    }
    else {
      curr_iter <- curr_iter + 1
    }
  }
  if (!converged) {
    setTxtProgressBar(pb, max_iter)
  }
  close(pb)
  fnorm_resid <- fnorm_resid[fnorm_resid != 0]
  norm_percent <- (1 - (tail(fnorm_resid, 1)/tnsr_norm)) * 
    100
  invisible(list(lambdas = 1L, U = U_list, conv = converged, 
                 est = est, norm_percent = norm_percent, fnorm_resid = tail(fnorm_resid, 
                                                                            1), all_resids = fnorm_resid))
}

.superdiagonal_tensor <- function(num_modes,len,elements=1L){
  modes <- rep(len,num_modes)
  arr <- array(0, dim = modes)
  if(length(elements)==1) elements <- rep(elements,len)
  for (i in 1:len){
    txt <- paste("arr[",paste(rep("i", num_modes),collapse=","),"] <- ", elements[i],sep="")
    eval(parse(text=txt))
  }
  as.tensor(arr)
}


initialize_u_l2=function(u_t_mean,u_t_sigma,dimensions,init_tensor_R,R_decomp
                         ,number_of_modes,number_of_datasets
                         ,sotrl_generate_udtr_seed,hosvd_seed,type,TensorX_ti_list){
  U_dtr_list=rep(list(vector("list",c(number_of_datasets))),number_of_modes)
  
  if(type=="random"){
    U_dtr_init=sotrl_generate_U_dtr(u_t_mean=u_t_mean,u_t_sigma=u_t_sigma
                                    ,dimensions,R=init_tensor_R
                                    ,number_of_modes,number_of_datasets
                                    ,seed=sotrl_generate_udtr_seed)
    for(t in 1:number_of_datasets){
      tensor=reshape_factormat_to_tensor(U_dtr_init,number_of_modes,dimensions,init_tensor_R,t)
      set.seed(hosvd_seed)
      hosvd_rank=min(dimensions)
      initial_cp_l2=hosvd(tensor,rep(hosvd_rank,number_of_modes))$U
      for(d in 1:number_of_modes){
        for(r in 1:(R_decomp-hosvd_rank)){
          initial_cp_l2[[d]]=cbind(initial_cp_l2[[d]],rnorm(dimensions[d],mean=u_t_mean,sd=u_t_sigma))
        }
        U_dtr_list[[d]][[t]]=initial_cp_l2[[d]]
      }
    }
  }else if(type=="decomposition"){
    for(t in 1:number_of_datasets){
      set.seed(t)
      selected_index=ceiling(runif(1,0,N[t]))
      tensor=TensorX_ti_list[[t]][[selected_index]]
      set.seed(hosvd_seed)
      hosvd_rank=min(dimensions)
      initial_cp_l2=cp(tensor,num_components = R_decomp)$U
      
      # initial_cp_l2=hosvd(tensor,rep(hosvd_rank,number_of_modes))$U
      for(d in 1:number_of_modes){
        # if(R_decomp>hosvd_rank){
        #   for(r in 1:(R_decomp-hosvd_rank)){
        #     initial_cp_l2[[d]]=cbind(initial_cp_l2[[d]],rnorm(dimensions[d],mean=u_t_mean,sd=u_t_sigma))
        #   }
        # }
        U_dtr_list[[d]][[t]]=initial_cp_l2[[d]]
      }
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
  if(distribution=="gaussian"){
    Predictions_t=innerproduct+Mu_t[t]
    p=c()
  }else if(distribution=="poisson"){
    Predictions_t=exp(innerproduct+Mu_t[t])#round(exp(innerproduct+Mu_t[t]))
    p=c()
  }
  return(list(Predictions_t,p))
}

calculate_RMSE=function(actual,predicted){
  return(sqrt(sum((predicted-actual)^2)/length(actual)))
}

calculate_WMAPE=function(actual,predicted){
  return(sum(abs(actual-predicted))/sum(abs(actual)))
}

calculate_MAPE=function(actual,predicted){
  return(sum(abs((actual-predicted)/actual))/length(actual))
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

r_gsotr_add_iter_to_performance_table=function(performance_table,model_type 
                                               ,Y_t,predictions_train,Y_t_test,predictions_test
                                               ,summary_table_L_lastiters
                                               ,round,N,N_test,number_of_datasets,distribution
                                               ,class_train,class_test,R,alpha,t,p_t,p_t_test
                                               ,epsilon,epsilon_test,scale=1){
  rmse_train_scaled=calculate_RMSE(Y_t,predictions_train)
  rmse_test_scaled=calculate_RMSE(Y_t_test,predictions_test)
  rmse_train=calculate_RMSE(scale*Y_t,scale*predictions_train)
  rmse_test=calculate_RMSE(scale*Y_t_test,scale*predictions_test)
  wmape_train=calculate_WMAPE(Y_t,predictions_train)
  wmape_test=calculate_WMAPE(Y_t_test,predictions_test)
  mape_train=calculate_MAPE(Y_t[[t]],predictions_train)
  mape_test=calculate_MAPE(Y_t_test[[t]],predictions_test)
  bias_train_scaled=calculate_bias(Y_t,predictions_train)
  bias_test_scaled=calculate_bias(Y_t_test,predictions_test)
  bias_train=calculate_bias(scale*Y_t,scale*predictions_train)
  bias_test=calculate_bias(scale*Y_t_test,scale*predictions_test)
  rmse_corrected_train=calculate_RMSE_errorcorrected(scale*Y_t,scale*predictions_train,epsilon)
  rmse_corrected_test=calculate_RMSE_errorcorrected(scale*Y_t_test,scale*predictions_test,epsilon_test)
  
  summary_table_L_lastiters_t=summary_table_L_lastiters[Site==t]
  performance_table=rbind(performance_table
                          ,data.table(Model=model_type,Round=round,Site=t,R=R
                                      ,alpha=alpha,N_train=N[t],N_test=N_test[t]
                                      ,RMSE_train=rmse_train,RMSE_test=rmse_test
                                      ,WMAPE_train=wmape_train,WMAPE_test=wmape_test
                                      ,MAPE_train=mape_train,MAPE_test=mape_test
                                      ,BIAS_train=bias_train,BIAS_test=bias_test
                                      ,RMSE_corrected_train=rmse_corrected_train
                                      ,RMSE_corrected_test=rmse_corrected_test
                                      ,L=summary_table_L_lastiters_t$L
                                      ,neg_loglike=summary_table_L_lastiters_t$neg_loglike)
                          ,fill=T)
  
  return(performance_table)
}

