function r=randg(a)
    d=a-1/3;
    c=1/sqrt(9*d);
    while true
        v=-1;
        while  v<=0
            x=randn();
            v=(1+c*x);
            v=v.*v.*v;
        end        
        u=rand();
        x2=x*x; x4=x2*x2;
        if (u<1-0.0331*x4 || log(u)<0.5*x2+d*(1-v+log(v)))
            r=d*v;
            return;
        end
    end
end
