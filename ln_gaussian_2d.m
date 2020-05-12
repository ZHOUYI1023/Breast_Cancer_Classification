function val = ln_gaussian_2d(mu, cov, p)
val = -1/2* log(det(cov)) - 1/2*transpose(p-mu)*inv(cov)*(p-mu);