function p = predict(theta, X1)
a=load('data4');
lk=size(a,2)
x=a(:,[3:lk]);
y=a(:,2);
m=size(x,1);
x=[ones(m,1),x];
sd=size(x,1);
l=size(x,2)
for i=1:l
    op=std(x(:,i));
    if(op~=0)
        as=mean(x(:,i));
        x(:,i)=(x(:,i)-mean(x(:,i)))/op;
        X1(1,i)=(X1(1,i)-as)/op;
    end
end
initial_theta=zeros(l,1);
initial_theta(1)=1;
options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 1000);
[optthetareg,funcval,exitflag]=fminunc(@(t)costFunctionReg(t,x,y,3),initial_theta,options);
m1 = size(X1,1);
p = zeros(m1, 1);
s=sigmoid(X1*optthetareg);
for i=1:m1
    if(s(i)>0.5)
        p(i)=1;
    else
        p(i)=0;
    end
end




% =========================================================================


end
