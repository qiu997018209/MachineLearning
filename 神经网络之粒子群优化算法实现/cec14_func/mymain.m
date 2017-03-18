 clear 
% mex cec13_func.cpp -DWINDOWS
func_num=1;
D=10;
VRmin=-100;
VRmax=100;
N=40;
Max_Gen=5000;
runs=1
fhd=str2func('cec14_func');



for i=1
    func_num=i;
    for j=1:runs
    
    [gbest,gbestval,pop]=myfunc(fhd,D,N,Max_Gen,VRmin,VRmax,func_num);
    xbest(i,:)=gbest
    fbest(i,j)=gbestval
   % plot(pbest(:,1)
   % plot(x,y)
    %plot(x,yy)
   % hold(a1,a2);
    %axis([500 5000,10^2 10^5]);
    %title(['适应度曲线  ' '终止代数＝' num2str(Max_Gen)]);
     %xlabel('进化代数');ylabel('适应度');
    end
    f_mean(i)=mean(fbest(i,j))
   
end

