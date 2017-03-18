
function[gbest,gbestval,pop]=myfunc(fhd,D,N,Max_Gen,VRmin,VRmax,varargin);
%------给定初始化条件----------------------------------------------
%问题要么在邻域部分，要么在速度更新。我在只更换速度公式的时候是可以得到一个明显的改善的

clc;
rand('state',sum(100*clock));
c=[0.5+log(2),0.5+log(2)];
%c=[2 2];
%w=1/(2*log(2));
w=0.9-(1:Max_Gen).*(0.5./Max_Gen)
Vmax=50;
Vmin=-50;
%------初始化种群的个体------------
%pbest 个体最佳，lbest 邻域最佳，gbest，全局最佳,VRmax,粒子范围，Vmin速度范围。
pop=VRmin+(VRmax-VRmin).*rand(N,D);
vel=0.5*(Vmin+(Vmax-Vmin).*rand(N,D)-pop);
%vel=Vmin+2.*Vmax.*rand(N,D)
e=feval(fhd,pop',varargin{:});


%% 个体极值和群体极值
pbest=pop; %个体最佳
pbestval=e;%个体最佳适应度值
[gbestval bestindex]=min(e);
gbest=pbest(bestindex,:);%全局最佳
%gbestrep=repmat(gbest,N,1);
n(1)=gbestval;

k=3;
%拓扑结构
for i=1:N;
   a1=[1:i-1,i+1:N];
   a1= a1(randperm(N-1));
   a2=a1(1:k);
   a3(i,:)=[a2,i];
   a4=a3(i,:);
   [lbestval(i),a5]=min(pbestval(a3(i,:)));
    a6=a4(a5);
   lbest(i,:)=pbest(a6,:);%邻域
end
%在更换速度公式的时候，可以更换变量
%gbestrep=repmat(gbest,N,1);
%aa=c(1).*rand(N,D).*(pbest-pop)+c(2).*rand(N,D).*(gbestrep-pop);
%vel=w(j).*vel+aa;
  

%% 迭代寻优

for j=2:Max_Gen;
    %更新速度
       l=pop+c(2)*rand(N,D).*(lbest-pop);%邻域
       p=pop+c(1)*rand(N,D).*(pbest-pop);%个体最佳
       G=(p+l+pop)/3;
       %aa=c(1).*rand(N,D).*(pbest-pop)+c(2).*rand(N,D).*(gbestrep-pop);
       %vel=w(j).*vel+aa;
       R=abs(G-pop);
       vel=w(j).*vel+G+(-1+rand(N,D)*2).*R
       vel=(vel>Vmax).*Vmax+(vel<=Vmax).*vel;
       vel=(vel<Vmin).*Vmin+(vel>=Vmin).*vel;
      %更新位置 
       pop=vel+pop;
       pop=((pop>=VRmin)&(pop<=VRmax)).*pop...
            +(pop<VRmin).*(VRmin+0.25.*(VRmax-VRmin).*rand(N,D))+(pop>VRmax).*(VRmax-0.25.*(VRmax-VRmin).*rand(N,D));
      %pop=((pop>=VRmin)&(pop<=VRmax)).*pop...
          %+(pop<VRmin).*(VRmin)+(pop>VRmax).*(VRmax);
       vel=((pop>=Vmin)&(pop<=Vmax)).*vel+(pop<Vmin).*0+(pop>Vmax).*0;
      %获取适应值
       e=feval(fhd,pop',varargin{:});
       %更新pbest，历史最佳
       t=(e<pbestval);
       m=repmat(t',1,D);
       pbest=m.*pop+(1-m).*pbest;
       pbestval=t.*e+(1-t).*pbestval;
       %更新lbest，局部最佳
       for i=1:N;
        a7=a3(i,:);
       [lbestval(i),m1]=min(pbestval(a3(i,:)))
       a8=a7(m1);
       lbest(i,:)=pbest(a8,:);
       end
      %得到全局最佳
       [gbestval id]=min(pbestval);
        n(j)=gbestval;
        gbest=pbest(id,:);
        gbestrep=repmat(gbest,N,1)
        %变更拓扑结构
       if isequal(n(j-1),n(j));
          for i=1:N;
          a1=[1:i-1,i+1:N];
          a1= a1(randperm(N-1));
          a2=a1(1:k);
          a3(i,:)=[a2,i];
          a4=a3(i,:);
          [lbestval(i),a5]=min(pbestval(a3(i,:)));
           a6=a4(a5);
           lbest(i,:)=pop(a6,:);
          end
       end
end
      
           
   
end




         

    
        
    
 

       
          
     
       


    