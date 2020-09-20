function [ co,kg,cls_co] = multicls_compute_mape(train_n,k,all_x,feature_pos,data_cls_pos,data_idx_pos,kmeans_pos,prename)
%COMPUTE_MAPE 此处显示有关此函数的摘要
%   此处显示详细说明
rand_num=20;
addpath('dace');

temp_sum=ones(rand_num,2);
temp_cls=ones(rand_num,k);
for i=1:rand_num
    all_Mval_y=all_x(all_x(:,data_cls_pos)==3,feature_pos+1);
    temp_cok=zeros(size(all_Mval_y));
    temp_kg=zeros(size(all_Mval_y));
    
    for k_idx=(1:k)
        Ms=all_x(all_x(:,kmeans_pos)==k_idx & all_x(:,data_cls_pos)==1,:);
        Mx=all_x(all_x(:,kmeans_pos)==k_idx & all_x(:,data_cls_pos)==2,:);
        Mval=all_x(all_x(:,kmeans_pos)==k_idx & all_x(:,data_cls_pos)==3,:);
        
        idx=randperm(size(Mx,1));
        nd=min(ceil(train_n/k),size(idx,2));
        train_idx=idx(1:nd);
        Mtrain=Mx(train_idx,1:feature_pos+1);
        
        Ms_x=Ms(:,1:feature_pos);
        Ms_y=Ms(:,feature_pos+1);
        
        Mtrain_x=Mtrain(:,1:feature_pos);
        Mtrain_y=Mtrain(:,feature_pos+1);
        Mval_x=Mval(:,1:feature_pos);
        Mval_y=Mval(:,feature_pos+1);
        Mval_ori_y=Mval(:,kmeans_pos+1);
        
        [dmodel,dmc,dmd]=cokriging2(Ms_x,Ms_y,Mtrain_x,Mtrain_y,@regpoly0,@corrspline,1e-6,1e2);
        cok=predict_cok2(Mval_x,dmodel);
        temp_idx=Mval(:,data_idx_pos);
        temp_cok(temp_idx,1)=cok;

        krig = dacefit(Mtrain_x,Mtrain_y,@regpoly0,@corrspline,1e-6,1e-3,20);
        kg = predictor(Mval_x,krig);
        temp_kg(temp_idx,1)=kg;
        
        mco=mean(abs((Mval_y-cok)./Mval_y))*100;
        cok=[mco;cok];
        Mval_y=[mco;Mval_y];
        Mval_ori_y=[mco;Mval_ori_y];
        if i==1
            various={'co_kg','Mval_y','Mval_ori_y'};
            %表的内容
            result_table=table(cok(:,1),Mval_y(:,1),Mval_ori_y(:,1),'VariableNames',various);
            %创建csv表格
            writetable(result_table, [prename,num2str(train_n),'_',num2str(k),'_',num2str(k_idx),'_cok.csv'])
        end
        temp_cls(i,k_idx)=mco;
    end
    meap_co=mean(abs((all_Mval_y-temp_cok)./all_Mval_y))*100;
    meap_k=mean(abs((all_Mval_y-temp_kg)./all_Mval_y))*100;
    temp_sum(i,1)=meap_co;
    temp_sum(i,2)=meap_k;
end
co=mean(temp_sum(:,1))
kg=mean(temp_sum(:,2))
cls_co=mean(temp_cls,1)
end

