function  value=calR2(x,y)
[P,S] = polyfit(x, y, 1);
yfit = P(1)*x + P(2);
R2 = norm(yfit -mean(y))^2/norm(y - mean(y))^2;
value=R2 ;
end
